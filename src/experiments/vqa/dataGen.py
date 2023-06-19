import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pickle
import re
import itertools

from trainer import ClauseNTrainer  # the class for training, testing, etc.
from preprocess import Query  # the abstract class to transform the strings of VQA to an abstraction of a query

from tqdm import tqdm

def get_SLASH_npp(outcomes:list, tag:str) -> str:
    """ Function to generate a valid NPP given a list of outcomes
        Inputs:
            outcomes<list>: list of strings entailing the outcomes
            tag<str>: the name tag of the npp
        Outputs:
            npp<str>: string descibing a valid NPP
    """
    if tag == "relation":
        str_outcomes = ", "
        str_outcomes = str_outcomes.join([f'{outcome.replace(" ", "_").replace("-", "_")}' for outcome in outcomes])
        npp = "".join([f"npp({tag}(1,X), ", "[", str_outcomes , "]) :- object(X1,X2)."])

    else: 
        str_outcomes = ", "
        str_outcomes = str_outcomes.join([f'{outcome.replace(" ", "_").replace("-", "_")}' for outcome in outcomes])
        npp = "".join([f"npp({tag}(1,X), ", "[", str_outcomes , "]) :- object(X)."])



    return npp

def get_SLASH_query(query_content:str , target_objects= None,non_target_objects=None, show_filter=None, num_objects=0) -> str:
    """ Given the string of SCALLOP query rewrite it to a SLASH variant.
        Inputs:
            query_content<str>: string entailing SCALLOP's version of a query.
        Outputs:
            main_rule_and_query<str>: SLASH' version of the given query
    """

    #adjust VQAR scallop format to SLASH by deleting the query(target(O)). entry.
    tmp = query_content.split("\n")
    main_rule = "\n".join([t for t in tmp[:-1] if t!=''])
    
    #store attributes and relations from the query here
    attr_rel_filter = ""

    #add target constraints for training.
    query = ":- target(X), target(Y), X != Y.\n" #display only one target per solution
    if target_objects is not None:

        #if an answer exists add them as a constraint. One of them has to be in a solution
        if len(target_objects) > 0:
            query += ":-"

            for target_object in target_objects:
                query += " not target({});".format(int(target_object))

            query= query[:-1] +".\n"

        #if non answer objects exist add them as a constraint.
        #Solution is UNSAT if it contains one of these answers.
        for non_target_object in non_target_objects: 
            query += ":- target({}).".format(int(non_target_object))
        
        query += "\n"

        #show statement to disable all printing
        show_statements = "#show. \n"

    #add show statements for testing
    else: 
        show_statements = "#show target/1.\n"


    #add show statements for training and testing
    #to build showstatements we use a list of tuples with target head and body as a tuple
    # example: ('O2', ['attr(dirty, O2)', 'relation(left, O1, O2)',...]) as an intermediate 
    # we call this intermediate representation show filter and transform it afterwards to the show statements

    #if we have two different rules (and/or)
    if len(show_filter)== 2:
        if show_filter[0][2] != show_filter[1][2]:
            show_filter[0][1].extend(show_filter[1][1])
            show_filter = [show_filter[0]]


    #iterate over all show filters
    for filter in show_filter: #filter[0] is the target variable

        #iterate over the body
        for var_descr in filter[1]:
            
            #filter attributes and relation predicates
            matches_attr_rel = re.findall(r'(relation|attr)\(([ a-zA-Z0-9]*|[ a-zA-Z0-9]*,[ a-zA-Z0-9]*),([-_ a-z]*)\)',var_descr )

            #if we found a relation or attribute iterate over them
            for match in matches_attr_rel:

                #every show statement/filter corresponds to one subgoal of the target query.
                #We want to have them seperate so we use a choice rule to show subgoals seperatly

                attr_rel_filter = attr_rel_filter + "///" +str(match[2]).replace(" ","")#extend the attribute/relation filter

                #for relations we only want one relation combination to fulfills the target requirement
                #we add a count constraint for this
                if match[0] == 'relation' :#and target_objects is not None:
                    query += ":- #count {{ {}(0,1,{},{}): target({}), {}}} > 1.\n".format(match[0],match[1],match[2],filter[0], ", ".join(filter[1]))

                #now build the ASP readable show statement
                show_statements += "#show {}(0,1,{},{}): target({}), {}.\n".format(match[0],match[1],match[2],filter[0], ", ".join(filter[1]))

            #filter names and add the showstatements
            matches_name = re.findall(r'name\(([ a-zA-Z0-9]*),([ -_a-zA-Z0-9]*)\)',var_descr)
            for match in matches_name:
                show_statements += "#show name(0,1,{}, X) : target({}), name(0,1,{},X), {}, {}.\n".format(match[0],  filter[0], match[0], ", ".join(filter[1]), "s(0)")


    #correct wrong KG translations
    def correct_kg_intput(str):
        return str.replace("vitamin_B", "vitamin_b").replace("even-toed","even_toed").replace("sub-event","sub_event").replace("odd-toed","odd_toed").replace('t-shirt','t_shirt').replace("chain-link","chain_link").replace("-ungulate","_ungulate")

    query = correct_kg_intput(query)

    #join all together in one query
    query_rule_and_show_statements = "".join(["%\t Target rule: \n", main_rule,"\n",query,  "\n%\tShow statement:\n", show_statements,"\n"] )

    query_rule_and_show_statements = correct_kg_intput(query_rule_and_show_statements)

    return query_rule_and_show_statements, attr_rel_filter


meta_f = "dataset/gqa_info.json"  # Meta file - Graph Question Answering information
with open (meta_f, 'r') as meta_f:
    meta_info = json.load(meta_f)



#load the set of names, attributes and relations
names = list(meta_info['name']['canon'].keys())
attributes = list(meta_info['attr']['canon'].keys())
relations = list(meta_info['rel']['canon'].keys())

#names= ["giraffe","tigger","lion"] #debug
#relations = ["left", "right","above"] #debug
#attributes = ["gren", "blue","striped","shiny"] #debug


#NPPS
name_npp = get_SLASH_npp(names, "name")
relation_npp = get_SLASH_npp(relations, "relation")
attribute_npp = get_SLASH_npp(attributes, "attr")


class VQA(Dataset):
    '''
    VQA dataset consisting of:
    objects
    flag_addinf: The flag for getting dictionary with additional informations for visualisations.
    '''


    def __init__(self, dataset,  task_file, num_objects =None, flag_addinf=False):
        self.data = list()
        self.dataset = dataset
        self.flag_addinf = flag_addinf
        self.max_objects = 0

        self.dataset_size = 0

        with open (task_file, 'rb') as tf:
            tasks = pickle.load(tf)

        print("dataloading done")
        

        self.object_list = []
        self.query_list = []
        self.feature_and_bb_list = []
        self.num_true_objects = []
        self.targets = []
        self.additional_information_list = []
        
        self.attr_rel_filter_list = []

        #if num_objects is not None:
        self.num_objects = num_objects
        print("taskfile len:{}, working with {} objects".format(len(tasks), self.num_objects))

        #some statistics
        self.num_rela=0
        self.num_attr=0
        #hist = np.zeros(self.num_objects+1)
        obj_list = np.arange(0, self.num_objects)


        #CREATE and empty scene dict 
        empty_scene_dict = {}

        #add empty objects for relations
        for id1 in obj_list:
            for id2 in obj_list:
                key = str(id1)+","+str(id2)
                if key not in empty_scene_dict:
                    empty_scene_dict[key] = torch.zeros([4104])

        #add empty objects for names and attributes
            for id in obj_list:
                empty_scene_dict[str(id)] = torch.zeros([2048])

        no_ans = 0
        for tidx, task in enumerate(tqdm(tasks)):

            # if task['image_id'] != 2337789:
            #    continue


            target = np.zeros(self.num_objects)
            clause_n_trainer = ClauseNTrainer(train_data_loader=tasks, val_data_loader=tasks, test_data_loader=tasks)

            #OBJECTS
            all_oid = task['question']['input']
            all_oid = [int(oid) for oid in all_oid]
            len_all_oid = len(all_oid)


            #only consider images with less then num_objects
            if self.num_objects is not None:
                if len_all_oid > self.num_objects:
                    continue
                else:
                    self.dataset_size +=1
            

            #map object ids to ids lower than num_objects
            correct_oids = task['question']['output']

            if len(correct_oids) == 0 :
                no_ans +=1
            len_correct_oid = len(correct_oids)
            not_correct_oids = [x for x in all_oid if (x not in correct_oids)]
            all_oid = correct_oids + not_correct_oids #this list contains all objects but puts the true objects first


            #map object ids to the inverval [0,25]
            oid_map = {}
            for idx, oid in enumerate(all_oid):
                oid_map[oid] = idx

            #map the correct answers
            correct_oids_mapped = [oid_map[correct_oid] for correct_oid in correct_oids]
            not_correct_oids_mapped = [oid_map[not_correct_oid] for not_correct_oid in not_correct_oids]
            #print("the correct answers are:", correct_oids)


            #set gt vector true for correct objects
            for oid in correct_oids_mapped:
                target[oid] = 1

            #create the QUERY
            query = task["question"]["clauses"]
            query = Query(query)
            query_content = clause_n_trainer.query_manager.transformer.transform(query)


            
            if 'rela'  in clause_n_trainer.query_manager.transformer.show_filter[0][1][0]:
               self.num_rela += 1
               #continue

            if 'attr'  in clause_n_trainer.query_manager.transformer.show_filter[0][1][0]:
               self.num_attr += 1
               #continue
            
            #if "hand" not in query_content and self.dataset=="test":
            #    self.dataset_size -=1
            #    continue

            #if ("controlling_flows_of_traffic" not in query_content) and self.dataset=="test":
            #    self.dataset_size -=1
            #    continue



            #collect bounding boxes and object features 
            #scene_dict stores all combinations while data_dict contains only the query-relevant ones
            scene_dict = empty_scene_dict.copy()


            # Put objects into scene dict
            #feature maps for attributes and names are of size 2048, bounding boxes are of size 4 
            for  oid in all_oid:
                scene_dict[str(oid_map[oid])] = torch.tensor(task['object_feature'][task['object_ids'].index(oid)])
            
            #feature maps for relations
            for  oid1 in all_oid:
                for  oid2 in all_oid:
                    key = str(oid_map[oid1])+","+str(oid_map[oid2])
                    scene_dict[key] = torch.cat([torch.tensor(task['object_feature'][task['object_ids'].index(oid1)]), torch.tensor(task['scene_graph']['bboxes'][oid1]),
                                                torch.tensor(task['object_feature'][task['object_ids'].index(oid2)]), torch.tensor(task['scene_graph']['bboxes'][oid2])])    


            
            #use only objects in the query for training
            if dataset == "train":
                query_rule_and_show_statement, attr_rel_filter = get_SLASH_query(query_content, np.array(correct_oids_mapped), np.array(not_correct_oids_mapped), clause_n_trainer.query_manager.transformer.show_filter,num_objects = len_all_oid) # Translate the query to SLASH
                

            #use all objects for evaluation
            else:
                query_rule_and_show_statement, attr_rel_filter= get_SLASH_query(query_content, show_filter =clause_n_trainer.query_manager.transformer.show_filter, num_objects = len_all_oid) # Translate the query to SLASH
            


            # if 'relation' in query_rule_and_show_statement:
            #    self.dataset_size -=1
            #    continue

            # if 'rule_1' in query_rule_and_show_statement:
            #    self.dataset_size -=1
            #    continue

            #queries without any relations or rule work up to 25 objects

            # if 'attr' in query_rule_and_show_statement:
            #    self.dataset_size -=1
            #    continue

            self.attr_rel_filter_list.append(attr_rel_filter)

            #we add the number of true objects for each image
            #later we can then remove NPPs which are not true objects
            self.num_true_objects.append(np.array(len_all_oid))



            #if 'flag_addinf' is True, generate "additional_information_dict"
            #and fill it with entries from "task" dictionary.
            if flag_addinf:
                additional_information_dict = {}
                #add image_id
                additional_information_dict['image_id'] = task['image_id']
                #add image's URL
                additional_information_dict['image_URL'] = task['url']
                #add original object ids
                additional_information_dict['original_object_ids'] = task['question']['input']
                #add original answer object's id
                additional_information_dict['original_answer_object_id'] = task['question']['output']
                #add the mapping from obj_ids to obj*
                additional_information_dict['mapping'] = oid_map
                #add the answers in form of obj*
                additional_information_dict['correct_oids_mapped'] = correct_oids_mapped
                #add the rest of obj*, which are not answers to the question
                additional_information_dict['not_correct_oids_mapped'] = not_correct_oids_mapped
                #add bounding boxes of every object in the image
                additional_information_dict['bboxes'] = task['scene_graph']['bboxes']
                #add query in the FOL format
                additional_information_dict['query'] = query_rule_and_show_statement
                additional_information_dict['scene_graph'] = task['scene_graph']
                self.additional_information_list.append(additional_information_dict)
                # print(self.additional_information_list[0]['image_id'])

            # add data, objects, queries and query rules to the dataset
            self.feature_and_bb_list.append(scene_dict)
            #self.object_list.append(object_string)
            self.query_list.append(query_rule_and_show_statement)
            self.targets.append(target)

            # if self.dataset_size >= 100 and self.dataset == "train":
            #   break
            # elif self.dataset_size >= 100 and self.dataset == "test":
            #   break
            # elif self.dataset_size >= 100 and self.dataset == "val":
            #   break
            

        # temp = np.cumsum(hist)
        # for hidx, h in enumerate(hist):
        #     if h != 0:
        #         print(hidx, h, temp[hidx]/tasks.__len__())




        print(len(self.query_list),len(self.targets))
        print("dataset size ", self.dataset,":",self.dataset_size)
        print("queries with no answer", no_ans)
        print("Num relations", self.num_rela)
        print("Num attr", self.num_attr)




    
    def __getitem__(self, index):

        if self.flag_addinf:
            return self.feature_and_bb_list[index], self.query_list[index], self.num_true_objects[index], self.targets[index] , self.additional_information_list[index]
        else:
            return self.feature_and_bb_list[index], self.query_list[index]+self.attr_rel_filter_list[index] , self.num_true_objects[index], self.targets[index]
        

    def __len__(self):
        return self.dataset_size
        #return len(self.feature_and_bb_list)