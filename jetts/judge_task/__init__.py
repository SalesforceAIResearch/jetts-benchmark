from ..data import ModelResponseData

class JudgeTask():
    '''
    A JudgeTask is the fundamental object in inference-time judging, in which a judge is used to help select, guide or refine model generations.
    Each judging task is implemented as a subclass of the JudgeTask.
    A JudgeTask object holds, at the very minimum, a list of queries to be answered by a certain generator model.
    The main method is run(self, judge), which takes a Judge instance and produces a list of responses, one for each query.
    '''

    def __init__(self, task_data: ModelResponseData):
        self.task_data = task_data

    def run(self, judge):
        raise NotImplementedError

from jetts.judge_task.rerank import RerankTask
from jetts.judge_task.refinement import SingleInstanceRefineTask, PairwiseCompareRefineTask