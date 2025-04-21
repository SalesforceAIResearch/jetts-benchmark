from __future__ import annotations

from ..data import Query, Response
from ..judge import Judge

import uuid, json
from collections import UserList
from tqdm import tqdm
from transformers import AutoTokenizer as AT, PreTrainedTokenizerBase

class ActiveBeam(UserList):

    def __init__(self, nodes: list[TreeNodeLike], M: int):
        self.data: list[TreeNodeLike] = nodes
        self.M: int = M

    def run_judge(self, judge: Judge, query: Query, top_k: int, tokenizer: PreTrainedTokenizerBase, lookahead: bool, **rerank_params):
        if len(self.data) <= top_k:  # if there are top_k or fewer nodes in the beam, return all of them
            return list(range(len(self.data)))
        if not lookahead:
            all_tokens = [n.get_generated_tokens_so_far() for n in self.data]
        else:
            all_tokens = [n.get_lookahead_tokens() for n in self.data]
        all_response_texts = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in all_tokens]
        responses = [Response(query=query, content=text) for text in all_response_texts]
        idxs = judge.rerank(query=query, responses=responses, partial_response=(not lookahead), **rerank_params)[0]
        return idxs[:top_k]

    def expand(self, idxs):
        result = []
        expansion_info = dict()
        for idx in idxs:
            node = self.data[idx]
            if node.is_leaf():
                result.append(node)
                continue
            if isinstance(node, TreeNode):
                # when expanding a true node, just get the first M children, which consists of one greedy response and M-1 sampled responses
                result.extend(node.children[:self.M])
            elif isinstance(node, TreeNodePointer):
                true_node = node.node
                id = true_node.id
                if id not in expansion_info:
                    expansion_info[id] = 0
                expansion_info[id] += 1
                greedy = true_node.children[0]
                sampled = true_node.children[1 + expansion_info[id] * (self.M - 1) : 1 + (expansion_info[id] + 1) * (self.M - 1)]
                result.append(greedy)
                result.extend(sampled)
        return ActiveBeam(result, self.M)
    
    def all_finished(self):
        return all(n.is_leaf() for n in self.data)

class TreeNodeLike():

    def __init__(self, id=None):
        if id is None:
            self.id = str(uuid.uuid4())
        else:
            self.id = id

    def is_leaf(self):
        '''
        return True if the current tree node cannot be further expanded, or False otherwise
        '''
        raise NotImplementedError

    def serialize(self, fo):
        raise NotImplementedError
    
    def get_generated_tokens_so_far(self, include_self=True, include_prompt=False):
        raise NotImplementedError
    
    def get_lookahead_tokens(self, include_prompt=False):
        raise NotImplementedError

class TreeNodePointer(TreeNodeLike):
    
    def __init__(self, node: TreeNode, id=None):
        super().__init__(id=id)
        self.node = node

    def is_leaf(self):
        return self.node.is_leaf()
    
    def serialize(self, fo):
        data = {'type': 'tree_node_pointer', 'id': self.id, 'node': self.node.id}
        fo.write(json.dumps(data) + '\n')

    def get_generated_tokens_so_far(self, *args, **kwargs):
        return self.node.get_generated_tokens_so_far(*args, **kwargs)
    
    def get_lookahead_tokens(self, *args, **kwargs):
        return self.node.get_lookahead_tokens(*args, **kwargs)

    @property
    def metadata(self):
        return self.node.metadata

class TreeNode(TreeNodeLike):

    def __init__(self, value: list[int], children=None, id=None, metadata=None):
        super().__init__(id=id)
        if metadata is None:
            self.metadata = dict()
        else:
            self.metadata = metadata
        self.value = value
        if children is None:
            self.children = []
        else:
            self.children = children

    def serialize(self, fo):
        data = {'type': 'tree_node', 'id': self.id, 'value': self.value, 'metadata': self.metadata, 'children': [ch.id for ch in self.children]}
        fo.write(json.dumps(data) + '\n')
        for ch in self.children:
            ch.serialize(fo)

    def is_leaf(self):
        return len(self.children) == 0

    def set_parent(self):
        self.parent = None
        for ch in self.children:
            if isinstance(ch, TreeNode):
                ch.set_parent()
                ch.parent = self

    def get_all_leaves(self, lst=None):
        if lst is None:
            lst = []
        assert isinstance(lst, list)
        if self.is_leaf():
            lst.append(self)
        else:
            for ch in self.children:
                if isinstance(ch, TreeNode):
                    ch.get_all_leaves(lst)
        return lst

    def get_generated_tokens_so_far(self, include_self=True, include_prompt=False, return_as_list=False):
        try:
            self.parent
        except AttributeError:
            raise Exception('This node\'s parent has not been set. You should call .set_parent() on the root node first.')
        cur_tokens = []
        if include_self:
            cur_tokens.append(self.value)
        cur_node = self.parent
        while cur_node is not None:
            assert isinstance(cur_node, TreeNode), f'Parent is not a TreeNode? (It is a {cur_node.__class__})'
            cur_tokens.append(cur_node.value)
            cur_node = cur_node.parent
        if not include_prompt:
            cur_tokens.pop()
        cur_tokens = cur_tokens[::-1]

        if not return_as_list:
            cur_tokens = [t for tks in cur_tokens for t in tks]
        else:
            cur_tokens = [tks for tks in cur_tokens]
        return cur_tokens
    
    def get_lookahead_tokens(self, include_prompt=False, return_as_list=False):
        cur = self
        while not cur.is_leaf():
            cur = cur.children[0]
            assert isinstance(cur, TreeNode), f'Greedy child not being a TreeNode? It is a {cur.__class__}'
        return cur.get_generated_tokens_so_far(include_self=True, include_prompt=include_prompt, return_as_list=return_as_list)

    def validate_children_multiplicity(self, M=2, K=5):
        if self.is_leaf():
            return
        children_dict = dict()
        for ch in self.children:
            if isinstance(ch, TreeNode):
                children_dict[ch.id] = [ch, 1]
            elif isinstance(ch, TreeNodePointer):
                children_dict[ch.node.id][1] += 1
        for ch, mul in children_dict.values():
            if ch.is_leaf():
                continue
            assert len(ch.children) == 1 + min(K, mul) * (M - 1)
        for ch in self.children:
            if isinstance(ch, TreeNode):
                ch.validate_children_multiplicity(M, K)

class SearchTree():

    def __init__(self, N, M, root: TreeNode, model_name: str, query: Query=None, none_query='warn'):
        assert none_query in ['warn', 'raise', 'none']
        self.N = N
        self.M = M
        self.K = int(self.N / self.M)
        self.root = root
        self.model_name = model_name
        self.tokenizer = AT.from_pretrained(model_name)
        if query is None:
            if none_query == 'warn':
                print('Warning: query is not provided for tree construction')
            elif none_query == 'raise':
                raise Exception('Query is not provided for tree construction')
        self.query = query

    def run_beam_search(self, judge, lookahead=False, use_tqdm=True, return_decisions=False, rerank_params=None, final_rerank_params=None):
        if rerank_params is None:
            rerank_params = dict()
        if final_rerank_params is None:
            final_rerank_params = dict()
        cur_beam = ActiveBeam(self.root.children, self.M)
        if use_tqdm:
            pbar = tqdm(ncols=70)
        if return_decisions:
            decisions = []
        while not cur_beam.all_finished():
            sel_idxs = cur_beam.run_judge(judge=judge, query=self.query, top_k=self.K, tokenizer=self.tokenizer, lookahead=lookahead, **rerank_params)
            if return_decisions:
                decisions.append(sel_idxs)
            cur_beam = cur_beam.expand(sel_idxs)
            if use_tqdm:
                pbar.update()
        # we use lookahead=True to make sure the judge prompt does not include any instruction about partial response for the final reranking
        idx = cur_beam.run_judge(judge=judge, top_k=1, query=self.query, tokenizer=self.tokenizer, lookahead=True, **final_rerank_params)[0]
        if return_decisions:
            decisions.append(idx)
        if use_tqdm:
            pbar.update()
            pbar.close()
        if not return_decisions:
            return cur_beam[idx]
        else:
            return cur_beam[idx], decisions
    
    def serialize(self, fn):
        with open(fn, 'w') as fo:
            metadata = {'type': 'metadata', 'N': self.N, 'M': self.M, 'model_name': self.model_name}
            if self.query is not None:
                metadata['query'] = self.query.to_dict()
            fo.write(json.dumps(metadata) + '\n')
            self.root.serialize(fo)

    @classmethod
    def deserialize(cls, fn, none_query='warn'):
        all_node_objects = dict()
        all_node_ids = []
        with open(fn) as f:
            metadata = json.loads(f.readline())
            assert metadata['type'] == 'metadata'
            query = metadata.get('query', None)
            if query is not None:
                query = Query.from_dict(query)
            for l in f:
                data = json.loads(l)
                assert data['type'] in ['tree_node', 'tree_node_pointer']
                id = data['id']
                all_node_ids.append(id)
                if data['type'] == 'tree_node':
                    node = TreeNode(value=data['value'], metadata=data.get('metadata', None), children=data['children'], id=id)
                else:
                    node = TreeNodePointer(node=data['node'], id=id)
                all_node_objects[id] = node
        for node in all_node_objects.values():
            if isinstance(node, TreeNode):
                node.children = [all_node_objects[id] for id in node.children]
            elif isinstance(node, TreeNodePointer):
                node.node = all_node_objects[node.node]
        root = all_node_objects[all_node_ids[0]]
        root.set_parent()
        return cls(N=metadata['N'], M=metadata['M'], root=root, model_name=metadata['model_name'], query=query, none_query=none_query)
