from ..data import Query, Response
from ..utils import next_pow_of_2, retrieve_idxs

import random
from enum import Enum
import numpy as np

class JudgeOutput(Enum):
    PARSING_FAILURE = 0

class Judge():

    def rerank(self, query: Query, responses: list[Response], use_tqdm=False, partial_response=False, **kwargs):
        return self.rerank_batch(query_lst=[query], responses_lst=[responses], use_tqdm=use_tqdm, partial_response=partial_response, **kwargs)[0]

    def pairwise_compare(self, query: Query, response0: Response, response1: Response, explain=False, use_tqdm=False, partial_response=False):
        return self.pairwise_compare_batch([query], [response0], [response1], explain, use_tqdm, partial_response)[0]

    def single_instance_rate(self, query: Query, response: Response, explain=False, use_tqdm=False, partial_response=False):
        return self.single_instance_rate_batch([query], [response], explain, use_tqdm, partial_response)[0]

    def rerank_batch(self, method: str, query_lst: list[Query], responses_lst: list[list[Response]], use_tqdm=True, partial_response=False, force_return_all=False):
        assert len(query_lst) == len(responses_lst)
        assert method in ['pairwise-rr', 'pairwise-ko', 'single-likert', 'single-additive']

        if method == 'pairwise-ko':
            return self.rerank_pairwise_ko_batch(query_lst, responses_lst, use_tqdm, partial_response, force_return_all=force_return_all)
        
        if method == 'pairwise-rr':
            tot_scores_lst = self.get_pairwise_rr_scores_batch(query_lst, responses_lst, use_tqdm, partial_response)
        elif 'single' in method:
            tot_scores_lst = self.get_single_scores_batch(query_lst, responses_lst, use_tqdm, partial_response=partial_response, render_kwargs={'which': method})
        ranked_idxs_lst = []
        noises_lst = []
        for tot_scores in tot_scores_lst:
            noises = np.random.random(tot_scores.shape) * 1e-6
            noised_scores = tot_scores + noises
            ranked_idxs = np.argsort(-noised_scores).tolist()
            noises_lst.append(noises.tolist())
            ranked_idxs_lst.append(ranked_idxs)
        return list(zip(ranked_idxs_lst, [e.tolist() for e in tot_scores_lst], noises_lst))

    def rerank_pairwise_ko_batch(self, query_lst, responses_lst, use_tqdm, partial_response, force_return_all=False):
        bracket_lst = []
        for query, responses in zip(query_lst, responses_lst):
            num_pad = next_pow_of_2(len(responses)) - len(responses)
            responses_padded = responses + [None] * num_pad
            random.shuffle(responses_padded)
            bracket_lst.append(responses_padded)
        while True:
            query_batch = []
            response0_batch = []
            response1_batch = []
            for query, responses in zip(query_lst, bracket_lst):
                if len(responses) == 1:
                    continue
                for r0, r1 in zip(responses[0::2], responses[1::2]):
                    if (r0 is not None) and (r1 is not None):
                        query_batch.append(query)
                        response0_batch.append(r0)
                        response1_batch.append(r1)
            if len(response0_batch) == 0:
                break
            scores = self.pairwise_compare_batch(query_batch, response0_batch, response1_batch, use_tqdm=use_tqdm, partial_response=partial_response)
            score_idx = 0
            for i in range(len(bracket_lst)):
                bracket = bracket_lst[i]
                new_paired_responses = []
                for r0, r1 in zip(bracket[0::2], bracket[1::2]):
                    if r0 is None:
                        new_paired_responses.append(r1)
                    elif r1 is None:
                        new_paired_responses.append(r0)
                    else:
                        score = scores[score_idx]
                        if score <= 0.5:  # it is fine to tie-break toward r0 arbitrarily since all responses are initially shuffled
                            new_paired_responses.append(r0)
                        elif score > 0.5:
                            new_paired_responses.append(r1)
                        score_idx += 1
                bracket_lst[i] = new_paired_responses
            assert score_idx == len(scores)
        ranked_idxs_lst = [retrieve_idxs(responses, bracket) for responses, bracket in zip(responses_lst, bracket_lst)]
        if not force_return_all:
            return [(ranked_idxs, np.array([0.0]), np.array([0.0])) for ranked_idxs in ranked_idxs_lst]
        else:
            result = []
            for ranked_idxs, responses in zip(ranked_idxs_lst, responses_lst):
                assert len(ranked_idxs) == 1
                scores = np.zeros(len(responses)) - 100
                scores[ranked_idxs[0]] = 100
                noises = np.random.random(len(responses)) * 1e-6
                new_ranked_idxs = np.argsort(-(scores + noises))
                result.append((new_ranked_idxs, scores, noises))
            return result

    def get_single_scores_batch(self, query_lst, responses_lst, use_tqdm, **kwargs):
        query_batch = []
        response_batch = []
        for query, responses in zip(query_lst, responses_lst):
            if len(responses) == 1:
                continue
            response_batch.extend(responses)
            query_batch.extend([query] * len(responses))
        if len(query_batch) != 0:
            scores = self.single_instance_rate_batch(query_batch, response_batch, use_tqdm=use_tqdm, **kwargs)
            scores = [s if s is not None else random.randint(1, self.prompter.single_instance_rate_max_score) for s in scores] # if failed to parse, assign random score
        tot_scores_lst = []
        ct = 0
        for query, responses in zip(query_lst, responses_lst):
            if len(responses) == 1:
                tot_scores_lst.append(np.array([1.0])) # if only one response, doesn't matter what the score is.
                continue
            tot_scores = np.zeros(len(responses))
            for i in range(len(responses)):
                tot_scores[i] = scores[ct]
                ct += 1
            tot_scores_lst.append(tot_scores)
        if ct != 0:
            assert ct == len(scores)
        return tot_scores_lst
    
    def get_pairwise_rr_scores_batch(self, query_lst: list[Query], responses_lst: list[list[Response]], use_tqdm=True, partial_response=False):
        query_batch = []
        response0_batch = []
        response1_batch = []
        for query, responses in zip(query_lst, responses_lst):
            if len(responses) == 1:
                continue
            N = len(responses)
            r0_lst, r1_lst = zip(*[(responses[i], responses[j]) for i in range(N) for j in range(i+1, N)])
            response0_batch.extend(r0_lst)
            response1_batch.extend(r1_lst)
            query_batch.extend([query] * len(r0_lst))

        if len(query_batch) != 0:
            scores = self.pairwise_compare_batch(query_batch, response0_batch, response1_batch, use_tqdm=use_tqdm, partial_response=partial_response)
        tot_scores_lst = []
        ct = 0
        for query, responses in zip(query_lst, responses_lst):
            if len(responses) == 1:
                tot_scores_lst.append(np.array([1.0]))
                continue
            N = len(responses)
            tot_scores = np.zeros(len(responses))
            for i in range(N):
                for j in range(i+1, N):
                    score = scores[ct]
                    tot_scores[i] += 1 - score
                    tot_scores[j] += score
                    ct += 1
            tot_scores_lst.append(tot_scores)
        if ct != 0:
            assert ct == len(scores)
        return tot_scores_lst
        
    def pairwise_compare_batch(self, query_lst: list[Query], response0_lst: list[Response], response1_lst: list[Response], explain=False, use_tqdm=True, partial_response=False):
        assert len(query_lst) == len(response0_lst) == len(response1_lst)
        if explain:
            raise Exception('Explanation is not supported for the default pairwise comparison method')
        query_batch = query_lst + query_lst
        response_batch = response0_lst + response1_lst
        score_batch = self.single_instance_rate_batch(query_batch, response_batch, explain, use_tqdm, partial_response)
        scores0 = score_batch[:len(query_lst)]
        scores1 = score_batch[len(query_lst):]
        all_scores = []
        for score0, score1 in zip(scores0, scores1):
            if score0 > score1:
                all_scores.append(0)
            elif score0 < score1:
                all_scores.append(1)
            else:
                all_scores.append(0.5)
        return all_scores
    
    def single_instance_rate_batch(self, query_lst: list[Query], response_lst: list[Response], explain=False, use_tqdm=True, partial_response=False):
        raise NotImplementedError
