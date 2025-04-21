from jetts.judge import Judge

class PromptedJudge(Judge):

    def __init__(self, prompter):
        super().__init__()
        self.prompter = prompter
        self.single_instance_rate_max_score = prompter.single_instance_rate_max_score

    def generate_batch(self, conversations, use_tqdm=True):
        raise NotImplementedError

    def aggregate_pairwise_scores(self, score_AB, score_BA):
        if (score_AB is None) and (score_BA is None):
            return 0.5
        elif score_AB is None:
            return 1 - score_BA
        elif score_BA is None:
            return score_AB
        else:
            return (score_AB + (1 - score_BA)) / 2

    def pairwise_compare_batch(self, query_lst, response0_lst, response1_lst, explain=False, use_tqdm=True, partial_response=False):
        assert len(query_lst) == len(response0_lst) == len(response1_lst)
        if not explain:
            prompt_AB_lst = [self.prompter.render_pairwise_prompt(q, r0, r1, partial_response=partial_response) for q, r0, r1 in zip(query_lst, response0_lst, response1_lst)]
            prompt_BA_lst = [self.prompter.render_pairwise_prompt(q, r1, r0, partial_response=partial_response) for q, r0, r1 in zip(query_lst, response0_lst, response1_lst)]

            all_prompts = prompt_AB_lst + prompt_BA_lst
            all_judgments = self.generate_batch(all_prompts, use_tqdm=use_tqdm)
            all_scores = [self.prompter.parse_pairwise_judgment(judgment, return_critique=False) for judgment in all_judgments]
            score_AB_lst = all_scores[:len(query_lst)]
            score_BA_lst = all_scores[len(query_lst):]
            return [self.aggregate_pairwise_scores(score_AB, score_BA) for score_AB, score_BA in zip(score_AB_lst, score_BA_lst)]
        else:
            all_prompts = [self.prompter.render_pairwise_prompt(q, r0, r1, partial_response=partial_response) for q, r0, r1 in zip(query_lst, response0_lst, response1_lst)]

            all_judgments = self.generate_batch(all_prompts, use_tqdm=use_tqdm)
            return [self.prompter.parse_pairwise_judgment(judgment, return_critique=True) for judgment in all_judgments]

    def single_instance_rate_batch(self, query_lst, response_lst, explain=False, render_kwargs=None, use_tqdm=True, partial_response=False):
        if render_kwargs is None:
            render_kwargs = dict()
        prompt_lst = [self.prompter.render_single_instance_rate_prompt(q, r, partial_response=partial_response, **render_kwargs) for q, r in zip(query_lst, response_lst)]
        judgment_lst = self.generate_batch(prompt_lst, use_tqdm=use_tqdm)
        return [self.prompter.parse_single_instance_rate_judgment(judgment, return_critique=explain) for judgment in judgment_lst]