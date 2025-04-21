from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict
from collections import UserList
from copy import deepcopy as copy

@dataclass
class Query():
    content: str
    metadata: dict = None

    def __post_init__(self):
        assert self.content is not None
        if self.metadata is None:
            self.metadata = dict()

    @classmethod
    def from_dict(cls, d):
        content = d['content']
        metadata = d.get('metadata', None)
        return cls(content=content, metadata=metadata)
    
    def to_dict(self):
        return asdict(self)

@dataclass
class Response():
    query: Query
    content: str
    metadata: dict = None

    def __post_init__(self):
        assert self.query is not None
        assert self.content is not None
        if self.metadata is None:
            self.metadata = dict()

    @classmethod
    def from_dict(cls, d, query=None):
        if query is None:
            query = Query.from_dict(d['query'])
        else:
            assert 'query' not in d, '"query" should not be provided in the dictionary when it is provided as a parameter'
        content = d['content']
        metadata = d.get('metadata', None)
        return cls(query=query, content=content, metadata=metadata)

    def to_dict(self, include_query=False):
        data = {'content': self.content, 'metadata': self.metadata}
        if include_query:
            data['query'] = self.query.to_dict()
        return data

@dataclass
class QueryWithResponses():
    query: Query
    responses: list[Response]

    def __post_init__(self):
        assert self.query is not None
        assert self.responses is not None and len(self.responses) != 0

    def merge(self, other: QueryWithResponses):
        assert self.query.content == other.query.content
        for response in other.responses:
            cp = copy(response)
            if isinstance(cp, Response):
                cp.query = self.query
            self.responses.append(cp)

    def to_dict(self, response_include_query=False):
        data = {'query': self.query.to_dict(), 'responses': [response.to_dict(include_query=response_include_query) for response in self.responses]}
        return data
    
    @classmethod
    def from_dict(cls, d, max_keep=None):
        query = Query.from_dict(d['query'])
        responses = [Response.from_dict(r, query=query) for r in d['responses']]
        if max_keep is not None:
            responses = responses[:max_keep]
        return cls(query, responses)

@dataclass
class ModelResponseData(UserList):
    data: list[QueryWithResponses]

    def __post_init__(self):
        assert self.data is not None and len(self.data) != 0

    def flatten(self, class_check=True):
        query_lst = []
        response_lst = []
        for qrs in self.data:
            query = qrs.query
            responses = qrs.responses
            for response in responses:
                if class_check and not isinstance(response, Response):
                    continue
                query_lst.append(query)
                response_lst.append(response)
        return query_lst, response_lst
    
    def prune_nonresponse(self):
        new_data = []
        for qrs in self.data:
            responses = [r for r in qrs.responses if isinstance(r, Response)]
            if len(responses) != 0:
                new_data.append(QueryWithResponses(qrs.query, responses))
        return ModelResponseData(new_data)
    
    def to_jsonl(self, fo):
        '''
        each line is a json string for {'query': query_json, 'responses': [response_json list]}
        '''
        with open(fo, 'w') as f:
            for qrs in self.data:
                f.write(json.dumps(qrs.to_dict(response_include_query=False)) + '\n')

    @classmethod
    def from_jsonl(cls, fn, max_keep=None):
        data = []
        with open(fn) as f:
            for line in f:
                qrs = QueryWithResponses.from_dict(json.loads(line), max_keep=max_keep)
                data.append(qrs)
        return cls(data)
