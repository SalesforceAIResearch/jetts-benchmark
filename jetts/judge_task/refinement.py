import uuid
from ..data import Response, ModelResponseData
from ..judge import Judge, JudgeOutput
from ..judge_task import JudgeTask
from ..judge_task.rerank import RerankTask
from ..refiner import Refiner, SpecialRefinementOutput

from tqdm import trange

from copy import deepcopy as copy

class TwoStageRefineTask(JudgeTask):
    def __init__(self, task_data: ModelResponseData, max_iterations=None, rerank_params=None):
        super().__init__(task_data)
        self.max_iterations = max_iterations
        self.rerank_params = rerank_params

    def run(self, judge: Judge, refiner: Refiner, max_iterations=None, return_all_refinements=True, 
            refine_tqdm='dataset', rerank_tqdm='dataset', reranking_output_fn=None):
        '''
        compute a list of max_iterations number of ModelResponseData instances, each of the same length as the original task_data, 
        representing each iteration of refinement. Refinement stops when the judgment parsing fails, the refiner finds no refinement needed, 
        or refined response parsing fails, and the corresponding status object will be returned. Subsequent iterations will have None in 
        the corresponding Response slot.
        return a ResponseOutputData object representing the final responses, and if return_all_refinements is True, also return the list 
        described above as the 2nd element of a tuple
        '''
        all_refinements = self.run_first_stage(judge, refiner, max_iterations, refine_tqdm=refine_tqdm)
        return self.run_second_stage(all_refinements, judge, return_all_refinements, rerank_tqdm=rerank_tqdm, reranking_output_fn=reranking_output_fn)
    
    def run_second_stage(self, all_refinements, judge: Judge, return_all_refinements=True, rerank_tqdm='dataset', reranking_output_fn=None):
        merged_data = self.merge_all_iterations(all_refinements).prune_nonresponse()
        rerank_task = RerankTask(merged_data, self.rerank_params)
        output = rerank_task.run(judge, use_tqdm=rerank_tqdm, output_fn=reranking_output_fn, force_return_all=True)
        if return_all_refinements:
            return output, all_refinements
        else:
            return output

class SingleInstanceRefineTask(TwoStageRefineTask):

    def __init__(self, task_data: ModelResponseData, max_iterations=None, rerank_params=None):
        super().__init__(task_data, max_iterations, rerank_params)

    def batch_refine(self, query_lst, response_lst, judge, refiner, use_tqdm=True):
        judgment_lst = judge.single_instance_rate_batch(query_lst, response_lst, explain=True)
        refine_query_lst = []
        refine_response_lst = []
        refine_score_lst = []
        refine_explanation_lst = []
        output_lst = []
        for query, response, (score, explanation) in zip(query_lst, response_lst, judgment_lst):
            if score is None or explanation is None:
                output_lst.append(JudgeOutput.PARSING_FAILURE)
                continue
            refine_query_lst.append(query)
            refine_response_lst.append(response)
            refine_score_lst.append(score)
            refine_explanation_lst.append(explanation)
            output_lst.append(-1)
        refine_max_score_lst = [judge.single_instance_rate_max_score] * len(refine_query_lst)
        refine_response_new_lst = refiner.single_instance_refine_batch(
            refine_query_lst, refine_response_lst, refine_explanation_lst, refine_score_lst, refine_max_score_lst, use_tqdm=use_tqdm)
        cur = 0
        for i in range(len(output_lst)):
            if output_lst[i] == -1:
                output_lst[i] = refine_response_new_lst[cur]
                cur += 1
        assert cur == len(refine_response_new_lst)
        return output_lst

    def assemble_refined_response(self, refined_response_lst, template: ModelResponseData):
        new_data = copy(template)
        cur = 0
        for qr in new_data:
            for i in range(len(qr.responses)):
                if isinstance(qr.responses[i], Response):
                    qr.responses[i] = refined_response_lst[cur]
                    cur += 1
                else:
                    qr.responses[i] = None
        assert cur == len(refined_response_lst)
        return new_data

    def merge_all_iterations(self, all_refinements: list[ModelResponseData]):
        assert len(set(map(len, all_refinements))) == 1, 'every refinement needs to have the same number of instances'
        merged = copy(all_refinements[0])
        for refinement in all_refinements[1:]:
            for qrs_merged, qrs_new in zip(merged, refinement):
                qrs_merged.merge(qrs_new)
        return merged

    def run_first_stage(self, judge: Judge, refiner: Refiner, max_iterations=None, refine_tqdm='dataset'):
        if refine_tqdm is None:
            refine_tqdm == 'none'
        assert refine_tqdm in ['iter', 'dataset', 'both', 'none'], 'Unrecognized tqdm option for refinement:' + refine_tqdm
        if max_iterations is None:
            assert self.max_iterations is not None, 'max_iterations needs to be specified either at initialization or in the run() function'
            max_iterations = self.max_iterations
        cur_refinement = copy(self.task_data)
        for qrs in cur_refinement:
            for response in qrs.responses:
                response.metadata['id'] = str(uuid.uuid4())
                response.metadata['parent'] = None
        all_refinements: list[ModelResponseData] = [cur_refinement]
        if refine_tqdm in ['iter', 'both']:
            iter_range = trange(max_iterations, desc='Refinement')
        else:
            iter_range = range(max_iterations)
        for _ in iter_range:
            query_lst, response_lst = cur_refinement.flatten()
            refined_response_lst = self.batch_refine(query_lst, response_lst, judge, refiner, use_tqdm=(refine_tqdm in ['dataset', 'both']))
            refinement = self.assemble_refined_response(refined_response_lst, cur_refinement)
            assert len(cur_refinement) == len(refinement)
            for old_qrs, new_qrs in zip(cur_refinement, refinement):
                assert len(old_qrs.responses) == len(new_qrs.responses)
                for old_response, new_response in zip(old_qrs.responses, new_qrs.responses):
                    if isinstance(new_response, Response):
                        new_response.metadata = dict()
                        new_response.metadata['id'] = str(uuid.uuid4())
                        new_response.metadata['parent'] = old_response.metadata['id']
                        old_response.metadata['child'] = new_response.metadata['id']
                    elif isinstance(new_response, SpecialRefinementOutput):
                        old_response.metadata['child'] = new_response.value
                    elif isinstance(new_response, JudgeOutput):
                        old_response.metadata['child'] = str(new_response)
            all_refinements.append(refinement)
            cur_refinement = refinement
        for qrs in cur_refinement:
            for response in qrs.responses:
                if isinstance(response, Response):
                    response.metadata['child'] = None
        return all_refinements

class PairwiseCompareRefineTask(TwoStageRefineTask):

    def __init__(self, task_data: ModelResponseData, max_iterations=None, rerank_params=None, use_first_2=False):
        super().__init__(task_data, max_iterations, rerank_params)
        if use_first_2:
            warned = False
            self.task_data = copy(self.task_data)
        for qrs in self.task_data:
            if use_first_2:
                if len(qrs.responses) > 2 and not warned:
                    print('Warning: PairwiseCompareRefineTask only supports pairwise comparison, truncating responses to the first 2.')
                    warned = True
                qrs.responses = qrs.responses[:2]
            else:
                assert len(qrs.responses) <= 2, f'PairwiseCompareRefineTask only supports pairwise comparison but found more than 2 responses for the query {qrs.query.content}'

    def batch_refine(self, query_lst, response_pair_lst, judge: Judge, refiner: Refiner, use_tqdm=True):
        assert len(query_lst) == len(response_pair_lst), 'unequal data list length'
        response_0_lst, response_1_lst = zip(*response_pair_lst)
        judge_query_lst = []
        judge_response_0_lst = []
        judge_response_1_lst = []
        judge_idx_lst = []
        output_lst = [-1] * len(query_lst)
        for i, (query, response_0, response_1) in enumerate(zip(query_lst, response_0_lst, response_1_lst)):
            if isinstance(response_0, Response) and isinstance(response_1, Response):
                judge_query_lst.append(query)
                judge_response_0_lst.append(response_0)
                judge_response_1_lst.append(response_1)
                judge_idx_lst.append(i)
            else:
                output_lst[i] = (None, None)
        refine_query_lst = []
        refine_response_0_lst = []
        refine_response_1_lst = []
        refine_choice_lst = []
        refine_explanation_lst = []
        refine_idx_lst = []
        judgment_lst = judge.pairwise_compare_batch(judge_query_lst, judge_response_0_lst, judge_response_1_lst, explain=True, use_tqdm=use_tqdm)
        for i, query, response_0, response_1, (choice, explanation) in zip(
            judge_idx_lst, judge_query_lst, judge_response_0_lst, judge_response_1_lst, judgment_lst):
            if choice is None or explanation is None:
                output_lst[i] = (JudgeOutput.PARSING_FAILURE, None)
            else:
                refine_query_lst.append(query)
                refine_response_0_lst.append(response_0)
                refine_response_1_lst.append(response_1)
                refine_choice_lst.append(choice)
                refine_explanation_lst.append(explanation)
                refine_idx_lst.append(i)
        refine_response_new_lst = refiner.pairwise_refine_batch(
            refine_query_lst, refine_response_0_lst, refine_response_1_lst, refine_explanation_lst, refine_choice_lst, use_tqdm=use_tqdm)
        for ii, (i, refined) in enumerate(zip(refine_idx_lst, refine_response_new_lst)):
            choice = refined.metadata['judge_choice']
            better_prev_response = [refine_response_0_lst[ii], refine_response_1_lst[ii]][choice]
            output_lst[i] = (refined, better_prev_response)
        assert -1 not in output_lst
        return output_lst

    def merge_all_iterations(self, all_refinements: list[ModelResponseData]):
        assert len(set(map(len, all_refinements))) == 1, 'every refinement needs to have the same number of instances'
        merged = copy(all_refinements[0])
        for refinement in all_refinements[1:]:
            for qrs_merged, qrs_new in zip(merged, refinement):
                qrs_new_cp = copy(qrs_new)
                qrs_new_cp.responses = [qrs_new_cp.responses[0]]
                qrs_merged.merge(qrs_new_cp)
        return merged

    def run_first_stage(self, judge: Judge, refiner: Refiner, max_iterations=None, refine_tqdm='dataset'):
        if refine_tqdm is None:
            refine_tqdm == 'none'
        assert refine_tqdm in ['iter', 'dataset', 'both', 'none'], 'Unrecognized tqdm option for refinement:' + refine_tqdm
        if max_iterations is None:
            assert self.max_iterations is not None, 'max_iterations needs to be specified either at initialization or in the run() function'
            max_iterations = self.max_iterations
        response_pair_lst = [qrs.responses for qrs in self.task_data]
        for p in response_pair_lst:
            if len(p) == 1:
                p.append(None)
        if refine_tqdm in ['iter', 'both']:
            iter_range = trange(max_iterations, desc='Refinement')
        else:
            iter_range = range(max_iterations)
        query_lst = [qrs.query for qrs in self.task_data]
        all_refinements: list[ModelResponseData] = [copy(self.task_data)]
        for _ in iter_range:
            refined_response_pair_lst = self.batch_refine(query_lst, response_pair_lst, judge, refiner, use_tqdm=(refine_tqdm in ['dataset', 'both']))
            new_refinement = copy(all_refinements[-1])
            for qrs, refined_pair in zip(new_refinement, refined_response_pair_lst):
                qrs.responses = refined_pair
            all_refinements.append(new_refinement)
            response_pair_lst = refined_response_pair_lst
        return all_refinements
