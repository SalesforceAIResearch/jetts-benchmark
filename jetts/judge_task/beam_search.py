import os, random, json, warnings
from dataclasses import dataclass
from tqdm import tqdm

from ..judge import Judge
from ..judge_task import JudgeTask
from ..data.tree import SearchTree

@dataclass
class BeamSearchTask(JudgeTask):
    input_dir: str
    output_dir: str
    num_instances: int
    lookahead: bool
    rerank_params: dict = None
    final_rerank_params: dict = None
    only_first_n: int = None

    def __post_init__(self):
        if self.rerank_params is None:
            self.rerank_params = dict()
        if self.final_rerank_params is None:
            self.final_rerank_params = dict()
        assert os.path.exists(self.input_dir), f'Input tree folder {self.input_dir} does not exist?'
        self.all_idxs = self.check_num_instances()
        if self.only_first_n is not None:
            self.all_idxs = self.all_idxs[:self.only_first_n]

    def check_num_instances(self):
        if self.num_instances is not None:
            missing_files = set([f'{idx}.jsonl' for idx in range(self.num_instances)]) - set(os.listdir(self.input_dir))
            assert missing_files == set([]), f'Missing input file(s): {sorted(list(missing_files))}'
            return list(range(self.num_instances))
        else:
            warnings.warn('num_instances not provided.')
            fns = [fn[:-5] for fn in os.listdir(self.input_dir) if fn.endswith('.jsonl')]
            idxs = sorted([int(fn) for fn in fns if fn.isdigit()])
            assert idxs == list(range(len(idxs))), f'Input files are not consecutively numbered starting from 0.jsonl'
            return idxs
    def get_input_fn(self, idx):
        return os.path.join(self.input_dir, f'{idx}.jsonl')

    def get_output_fn(self, idx):
        return os.path.join(self.output_dir, f'{idx}.jsonl')
    
    def create_tmp_file(self, idx):
        fn = self.get_output_fn(idx) + '.tmp'
        with open(fn, 'w') as f:
            f.write('placeholder')

    def delete_tmp_file(self, idx):
        fn = self.get_output_fn(idx) + '.tmp'
        try:
            os.remove(fn)
        except:
            pass

    def is_unjudged(self, idx):
        fn = self.get_output_fn(idx)
        tmp_fn = fn + '.tmp'
        return (not os.path.exists(fn)) and (not os.path.exists(tmp_fn))

    def run(self, judge: Judge, use_tqdm: str = 'both'):
        assert use_tqdm in ['both', 'dataset', 'instance', 'none']
        unjudged_idxs = [idx for idx in self.all_idxs if self.is_unjudged(idx)]
        random.shuffle(unjudged_idxs)
        if use_tqdm in ['both', 'dataset']:
            unjudged_idxs = tqdm(unjudged_idxs, ncols=70)
        for idx in unjudged_idxs:
            if not self.is_unjudged(idx):
                continue
            self.create_tmp_file(idx)
            tree = SearchTree.deserialize(self.get_input_fn(idx))
            selected_node, decisions = tree.run_beam_search(judge, self.lookahead, use_tqdm in ['both', 'instance'], return_decisions=True, 
                                                            rerank_params=self.rerank_params, final_rerank_params=self.final_rerank_params)
            judgment = {'idx': idx, 'node_id': selected_node.id, 'decisions': decisions, 
                        'query': tree.query.content, 'query_metadata': tree.query.metadata, 
                        'response': tree.tokenizer.decode(selected_node.get_generated_tokens_so_far(), skip_special_tokens=True), 
                        'response_metadata': selected_node.metadata}
            with open(self.get_output_fn(idx), 'w') as f:
                f.write(json.dumps(judgment) + '\n')
            self.delete_tmp_file(idx)
