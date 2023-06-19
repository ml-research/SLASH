"""
The source code is based on:
Scallop: From Probabilistic Deductive Databases to Scalable Differentiable Reasoning
Jiani Huang, Ziyang Li, Binghong Chen, Karan Samel, Mayur Naik, Le Song, Xujie Si
Advances in Neural Information Processing Systems 34 (NeurIPS 2021)
https://proceedings.neurips.cc/paper/2021/hash/d367eef13f90793bd8121e2f675f0dc2-Abstract.html
"""

# Prog = Conj | Logic
# Logic = biOp Prog Prog
# Conj = and Des Conj | Des
# Des = rela Relaname O1 O2 | attr Attr O

class Variable():
    def __init__(self, id):
        self.var_id = f"O{id}"
        self.name_id = f"N{id}"
        self.name = []
        self.hypernyms = []
        self.attrs = []
        self.kgs = []
        # The relations where this object functions as a subject
        self.sub_relas = []
        self.obj_relas = []

    def has_rela(self):
        if len(self.sub_relas) == 0 and len(self.obj_relas) == 0:
            return False
        return True

    def get_name_id(self):
        # if (not len(self.hypernyms) == 0) and len(self.name) == 0:
        #     return True, self.name_id
        # if (not len(self.kgs) == 0) and len(self.name) == 0:
        #     return True, self.name_id
        return False, self.name_id

    def set_name(self, name):
        if name in self.name:
            return

        self.name.append(name)

    def set_kg(self, kg):
        if kg not in self.kgs:
            self.kgs.append(kg)

    def set_hypernym(self, hypernym):
        if hypernym not in self.hypernyms:
            self.hypernyms.append(hypernym)

    def set_attr(self, attr):
        if attr not in self.attrs:
            self.attrs.append(attr)

    def set_obj_relas(self, obj_rela):
        if obj_rela not in self.obj_relas:
            self.obj_relas.append(obj_rela)

    def set_sub_relas(self, sub_rela):
        if sub_rela not in self.sub_relas:
            self.sub_relas.append(sub_rela)

    def get_neighbor(self):
        neighbors = []
        for rela in self.sub_relas:
            neighbors.append(rela.obj)
        for rela in self.obj_relas:
            neighbors.append(rela.sub)
        return neighbors

    def update(self, other):

        self.hypernyms = list(set(self.name + other.name))
        self.hypernyms = list(set(self.hypernyms + other.hypernyms))
        self.attrs = list(set(self.attrs + other.attrs))
        self.kgs = list(set(self.kgs + other.kgs))


    def to_datalog(self, with_name=False, with_rela=True):

        name_query = []

        if (len(self.name) == 0) and with_name:
            name_query.append("name({}, {})".format(self.name_id.replace(" ","_").replace(".","_"), self.var_id.replace(" ","_").replace(".","_")))

        if (not len(self.name) == 0):
            for n in self.name:
                #name_query.append(f"name(\"{n}\", {self.var_id})")
                n = n.replace(" ","_").replace(".","_")
                #name_query.append(f"name(0,1,{self.var_id},{n})")
                name_query.append(f"name({self.var_id},{n})")


        #attr_query = [ f"attr(\"{attr}\", {self.var_id})" for attr in self.attrs]
        #attr_query = [ "attr(0,1,{}, {})".format(self.var_id.replace(" ","_"),attr.replace(" ","_")) for attr in self.attrs]
        attr_query = [ "attr({}, {})".format(self.var_id.replace(" ","_"),attr.replace(" ","_")) for attr in self.attrs]

        #hypernym_query = [f"name(\"{hypernym}\", {self.var_id})" for hypernym in self.hypernyms]
        #hypernym_query = ["name(0,1,{}, {})".format(self.var_id.replace(" ","_").replace(".","_"),hypernym.replace(" ","_").replace(".","_")) for hypernym in self.hypernyms]
        hypernym_query = ["name({}, {})".format(self.var_id.replace(" ","_").replace(".","_"),hypernym.replace(" ","_").replace(".","_")) for hypernym in self.hypernyms]

        kg_query = []

        for kg in self.kgs:
            restriction = list(filter(lambda x: not x == 'BLANK' and not x == '', kg))
            assert (len(restriction) == 2)
            rel = restriction[0].replace(" ","_")
            usage = restriction[1].replace(" ","_")
            #kg_query += [f"name({self.name_id}, {self.var_id}), oa_rel({rel}, {self.name_id}, {usage})"]
            #kg_query +=     ["name({}, {}), oa_rel({}, {}, {})".format(self.name_id.replace(" ","_").replace(".","_") ,self.var_id.replace(" ","_").replace(".","_") ,rel.replace(" ","_").replace(".","_"), self.name_id.replace(" ","_").replace(".","_"), usage.replace(" ","_").replace(".","_"))]
            #kg_query +=     ["name(0,1,{}, {}), oa_rel({}, {}, {})".format(self.var_id.replace(" ","_").replace(".","_") ,self.name_id.replace(" ","_").replace(".","_"), rel.replace(" ","_").replace(".","_"), self.name_id.replace(" ","_").replace(".","_"), usage.replace(" ","_").replace(".","_"))]
            kg_query +=     ["name({}, {}), oa_rel({}, {}, {})".format(self.var_id.replace(" ","_").replace(".","_") ,self.name_id.replace(" ","_").replace(".","_"), rel.replace(" ","_").replace(".","_"), self.name_id.replace(" ","_").replace(".","_"), usage.replace(" ","_").replace(".","_"))]

        if with_rela:
            rela_query = [rela.to_datalog() for rela in self.sub_relas]
        else:
            rela_query = []

        program = name_query + attr_query + hypernym_query + kg_query + rela_query

        #print(program)
        return program

class Relation():
    def __init__(self, rela_name, sub, obj):
        self.rela_name = rela_name
        self.sub = sub
        self.obj = obj
        self.sub.set_sub_relas(self)
        self.obj.set_obj_relas(self)

    def substitute(self, v1, v2):
        if self.sub == v1:
            self.sub = v2
        if self.obj == v1:
            self.obj = v2

    def to_datalog(self):
        #rela_query = f"relation(\"{self.rela_name}\", {self.sub.var_id},  {self.obj.var_id})"
        #rela_query = "relation(0,1,{}, {},  {})".format( self.sub.var_id.replace(" ","_"), self.obj.var_id.replace(" ","_"),self.rela_name.replace(" ","_"))
        rela_query = "relation({}, {},  {})".format( self.sub.var_id.replace(" ","_"), self.obj.var_id.replace(" ","_"),self.rela_name.replace(" ","_"))

        return rela_query


# This is for binary operations on variables
class BiOp():
    def __init__(self, op_name, v1, v2):
        self.op_name = op_name
        self.v1 = v1
        self.v2 = v2

    def to_datalog(self):
        raise NotImplementedError

class Or(BiOp):
    def __init__(self, v1, v2):
        super().__init__('or', v1, v2)

    def to_datalog(self):
        pass

class And(BiOp):
    def __init__(self, v1, v2):
        super().__init__('and', v1, v2)

    def to_datalog(self):
        pass

class Query():

    def __init__(self, query):
        self.vars = []
        self.relations = []
        self.operations = []
        self.stack = []
        self.preprocess(query)


    def get_target(self):
        pass

    def get_new_var(self):
        self.vars.append(Variable(len(self.vars)))

    def preprocess(self, query):

        # for clause in question["program"]:
        for clause in query:

            if clause['function'] == "Initial":
                if not len(self.vars) == 0:
                    self.stack.append(self.vars[-1])
                self.get_new_var()
                self.root = self.vars[-1]

            # logic operations
            elif clause['function'] == "And":
                v = self.stack.pop()
                self.operations.append(And(v, self.vars[-1]))
                self.root = self.operations[-1]

            elif clause['function'] == "Or":
                v = self.stack.pop()
                self.operations.append(Or(v, self.vars[-1]))
                self.root = self.operations[-1]

            # find operations
            elif clause['function'] == "KG_Find":
                self.vars[-1].set_kg(clause['text_input'])

            elif clause['function'] == "Hypernym_Find":
                self.vars[-1].set_hypernym(clause['text_input'])

            elif clause['function'] == "Find_Name":
                self.vars[-1].set_name(clause['text_input'])

            elif clause['function'] == "Find_Attr":
                self.vars[-1].set_attr(clause['text_input'])

            elif clause['function'] == "Relate_Reverse":
                self.get_new_var()
                self.root = self.vars[-1]
                obj = self.vars[-2]
                sub = self.vars[-1]
                rela_name = clause['text_input']
                relation = Relation(rela_name, sub, obj)
                self.relations.append(relation)

            elif clause['function'] == "Relate":
                self.get_new_var()
                self.root = self.vars[-1]
                sub = self.vars[-2]
                obj = self.vars[-1]
                rela_name = clause['text_input']
                relation = Relation(rela_name, sub, obj)
                self.relations.append(relation)

            else:
                raise Exception(f"Not handled function: {clause['function']}")


# Optimizers for optimization
class QueryOptimizer():

    def __init__(self, name):
        self.name = name

    def optimize(self, query):
        raise NotImplementedError

# This only works for one and operation at the end
# This is waited for update
# class AndQueryOptimizer(QueryOptimizer):

#     def __init__(self):
#         super().__init__("AndQueryOptimizer")

#     # For any and operation, this can be rewritten as a single object
#     def optimize(self, query):

#         if len(query.operations) == 0:
#             return query

#         assert(len(query.operations) == 1)

#         operation = query.operations[0]
#         # merge every subtree into one
#         if operation.name == "and":
#             v1 = operation.v1
#             v2 = operation.v2
#             v1.merge(v2)

#             for relation in query.relations:
#                 relation.substitute(v2, v1)

#             query.vars.remove(v2)

#             if query.root == operation:
#                 query.root = v1

#         return query


class HypernymOptimizer(QueryOptimizer):

    def __init__(self):
        super().__init__("HypernymOptimizer")

    def optimize(self, query):

        if (query.name is not None and not query.hypernyms == []):
            query.hypernyms = []

        return query


class KGOptimizer(QueryOptimizer):

    def __init__(self):
        super().__init__("HypernymOptimizer")

    def optimize(self, query):

        if (query.name is not None and not query.kgs == []):
            query.kgs = []

        return query
