"""
The source code is based on:
Scallop: From Probabilistic Deductive Databases to Scalable Differentiable Reasoning
Jiani Huang, Ziyang Li, Binghong Chen, Karan Samel, Mayur Naik, Le Song, Xujie Si
Advances in Neural Information Processing Systems 34 (NeurIPS 2021)
https://proceedings.neurips.cc/paper/2021/hash/d367eef13f90793bd8121e2f675f0dc2-Abstract.html
"""
import json


class Idx2Word():

    def __init__(self, meta_info, use_canon=False):
        self.setup(meta_info)
        self.attr_canon = meta_info['attr']['canon']
        self.name_canon = meta_info['name']['canon']
        self.rela_canon = meta_info['rel']['canon']

        self.attr_alias = meta_info['attr']['alias']
        self.rela_alias = meta_info['rel']['alias']

        self.attr_to_idx_dict = meta_info['attr']['idx']
        self.rela_to_idx_dict = meta_info['rel']['idx']
        self.name_to_idx_dict = meta_info['name']['idx']
        self.use_canon = use_canon
        # print("here")

    def setup(self, meta_info):

        attr_to_idx = meta_info['attr']['idx']
        rela_to_idx = meta_info['rel']['idx']
        name_to_idx = meta_info['name']['idx']

        attr_freq = meta_info['attr']['freq']
        rela_freq = meta_info['rel']['freq']
        name_freq = meta_info['name']['freq']

        # attr_group = meta_info['attr']['group']

        def setup_single(to_idx, freq, group=None):
            idx_to_name = {}
            for name in freq:
                if name not in to_idx:
                    continue
                idx = to_idx[name]
                if type(idx) == list:
                    if not idx[0] in idx_to_name.keys():
                        idx_to_name[idx[0]] = {}
                    idx_to_name[idx[0]][idx[1]] = name
                else:
                    idx_to_name[idx] = name
            return idx_to_name

        self.idx_to_name_dict = setup_single(name_to_idx, name_freq)
        self.idx_to_rela_dict = setup_single(rela_to_idx, rela_freq)
        self.idx_to_attr_dict = setup_single(attr_to_idx, attr_freq)
        # self.idx_to_attr_dict = setup_single(attr_to_idx, attr_freq, attr_group)

    def get_name_ct(self):
        return len(self.idx_to_name_dict)

    def get_rela_ct(self):
        return len(self.idx_to_rela_dict)

    def get_attr_ct(self):
        return len(self.idx_to_attr_dict)

    def get_names(self):
        return list(self.idx_to_name_dict.values())

    def idx_to_name(self, idx):
        if idx is None:
            return None
        if type(idx) == str:
            return idx
        if len(self.idx_to_name_dict) == idx:
            return None
        if idx == -1:
            return None
        return self.idx_to_name_dict[idx]

    def idx_to_rela(self, idx):
        if idx is None:
            return None
        if idx == -1:
            return None
        if type(idx) == str:
            return idx
        if len(self.idx_to_rela_dict) == idx:
            return None
        return self.idx_to_rela_dict[idx]

    def idx_to_attr(self, idx):
        if idx is None:
            return None
        if type(idx) == str:
            return idx
        if len(self.idx_to_attr_dict) == idx:
            return None
        if idx == -1:
            return None
        # return self.idx_to_attr_dict[idx[0]][idx[1]]
        return self.idx_to_attr_dict[idx]

    def attr_to_idx(self, attr):
        if attr is None:
            return attr

        if self.use_canon:
            if attr in self.attr_canon.keys():
                attr = self.attr_canon[attr]

        if attr in self.attr_alias.keys():
            attr = self.attr_alias[attr]

        if attr not in self.attr_to_idx_dict.keys():
            return None

        return self.attr_to_idx_dict[attr]

    def name_to_idx(self, name):

        if name is None:
            return name

        if self.use_canon:
            if name in self.name_canon.keys():
                name = self.name_canon[name]

        if name not in self.name_to_idx_dict.keys():
            return None

        return self.name_to_idx_dict[name]

    def rela_to_idx(self, rela):
        if rela is None:
            return rela

        if self.use_canon:
            if rela in self.rela_canon.keys():
                rela = self.rela_canon[rela]

        if rela in self.rela_alias.keys():
            rela = self.rela_alias[rela]

        if rela not in self.rela_to_idx_dict.keys():
            return None

        return self.rela_to_idx_dict[rela]


def process_program(program, meta_info):

    new_program = []

    for clause in program:
        new_clause = {}
        new_clause['function'] = clause['function']

        if 'output' in clause.keys():
            new_clause['output'] = clause['output']

        if clause['function'] == "Hypernym_Find":
            name = clause['text_input'][0]
            attr = clause['text_input'][1]

            new_clause['text_input'] = [name, attr] + clause['text_input'][2:]

        elif clause['function'] == "Find":
            name = clause['text_input'][0]
            new_clause['text_input'] = [name] + clause['text_input'][1:]

        elif clause['function'] == "Relate_Reverse":
            relation = clause['text_input']
            new_clause['text_input'] = relation

        elif clause['function'] == "Relate":
            relation = clause['text_input']
            new_clause['text_input'] = relation

        else:
            if 'text_input' in clause.keys():
                new_clause['text_input'] = clause['text_input']

        new_program.append(new_clause)

    return new_program


def process_questions(questions_path, new_question_path, meta_info):
    new_questions = {}

    with open(questions_path, 'r') as questions_file:
        questions = json.load(questions_file)

    # process questions
    for question in questions:

        image_id = question["image_id"]

        # process questions
        if image_id not in new_questions.keys():
            new_questions[image_id] = {}
        new_question = new_questions[image_id]

        new_question['question_id'] = question['question_id']
        new_question['program'] = process_program(
            question["program"], meta_info)

        program = question['program']
        new_question['target'] = program[-2]["output"]
        new_question['question'] = question['question']
        new_question['answer'] = question['answer']

    with open(new_question_path, 'w') as new_question_file:
        json.dump(new_questions, new_question_file)

