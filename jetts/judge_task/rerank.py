
from ..judge_task import JudgeTask
from ..data import ModelResponseData, QueryWithResponses
from ..judge import Judge

import json
import numpy as np
from tqdm import tqdm

class RerankTask(JudgeTask):
    
    def __init__(self, task_data: ModelResponseData, rerank_params=None):
        super().__init__(task_data)
        if rerank_params is None:
            self.rerank_params = dict()
        else:
            self.rerank_params = rerank_params

    def run(self, judge: Judge, output_fn=None, rerank_params=None, use_tqdm='dataset', batch_size=None, force_return_all=False):
        if use_tqdm is None:
            use_tqdm == 'none'
        assert use_tqdm in ['dataset', 'instance', 'both', 'none'], 'Unrecognized tqdm option for reranking: ' + use_tqdm
        if rerank_params is None:
            rerank_params = self.rerank_params
        if batch_size is None:
            method = rerank_params.get('method', None)
            if method in [None, 'pairwise-rr']:
                batch_size = 1
            elif method == 'pairwise-ko':
                batch_size = 60
            elif method.startswith('single-'):
                batch_size = 10
        if output_fn is not None:
            f = open(output_fn, 'w')
        output = []
        num_batches = int(np.ceil(len(self.task_data) / batch_size))
        if use_tqdm in ['dataset', 'both']:
            pbar = tqdm(total=len(self.task_data), desc='Reranking', ncols=70)
        for b in range(num_batches):
            query_lst = [qrs.query for qrs in self.task_data.data[b * batch_size : (b + 1) * batch_size]]
            responses_lst = [qrs.responses for qrs in self.task_data.data[b * batch_size : (b + 1) * batch_size]]
            result_batch = judge.rerank_batch(query_lst=query_lst, responses_lst=responses_lst, use_tqdm=(tqdm in ['instance', 'both']), force_return_all=force_return_all, **rerank_params)
            ranked_idxs_lst, raw_scores_lst, noises_lst = zip(*result_batch)
            ranked_responses_lst = [[responses[idx] for idx in ranked_idxs] for responses, ranked_idxs in zip(responses_lst, ranked_idxs_lst)]
            ranked_raw_scores_lst = [[raw_scores[idx] for idx in ranked_idxs] for raw_scores, ranked_idxs in zip(raw_scores_lst, ranked_idxs_lst)]
            ranked_noises_lst = [[noises[idx] for idx in ranked_idxs] for noises, ranked_idxs in zip(noises_lst, ranked_idxs_lst)]
            output.extend([QueryWithResponses(query, top_response) for query, top_response in zip(query_lst, ranked_responses_lst)])
            if output_fn is not None:
                for query, top_responses, raw_scores, noises in zip(query_lst, ranked_responses_lst, ranked_raw_scores_lst, ranked_noises_lst):
                    out_data = {'query': query.to_dict(), 
                                'responses': [response.to_dict() for response in top_responses], 
                                'raw_scores': raw_scores, 
                                'noises': noises}
                    f.write(json.dumps(out_data) + '\n')
                f.flush()
            if use_tqdm in ['dataset', 'both']:
                pbar.update(len(query_lst))

        if use_tqdm in ['dataset', 'both']:
            pbar.close()
        if output_fn is not None:
            f.close()
        return ModelResponseData(output)
