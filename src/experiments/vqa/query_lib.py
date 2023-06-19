"""
The source code is based on:
Scallop: From Probabilistic Deductive Databases to Scalable Differentiable Reasoning
Jiani Huang, Ziyang Li, Binghong Chen, Karan Samel, Mayur Naik, Le Song, Xujie Si
Advances in Neural Information Processing Systems 34 (NeurIPS 2021)
https://proceedings.neurips.cc/paper/2021/hash/d367eef13f90793bd8121e2f675f0dc2-Abstract.html
"""

import pickle
import os
import subprocess

from transformer import DetailTransformer, SimpleTransformer
from preprocess import Query
from knowledge_graph import KG, RULES

# animals = ["giraffe", "cat", "kitten", "dog", "puppy", "poodle", "bull", "cow", "cattle", "bison", "calf", "pig", "ape", "monkey", "gorilla", "rat", "squirrel", "hamster", "deer", "moose", "alpaca", "elephant", "goat", "sheep", "lamb", "antelope", "rhino", "hippo",  "zebra", "horse", "pony", "donkey", "camel", "panda", "panda bear", "bear", "polar bear", "seal", "fox", "raccoon", "tiger", "wolf", "lion", "leopard", "cheetah", "badger", "rabbit", "bunny", "beaver", "kangaroo", "dinosaur", "dragon", "fish", "whale", "dolphin", "crab", "shark", "octopus", "lobster", "oyster", "butterfly", "bee", "fly", "ant", "firefly", "snail", "spider", "bird", "penguin", "pigeon", "seagull", "finch", "robin", "ostrich", "goose", "owl", "duck", "hawk", "eagle", "swan", "chicken", "hen", "hummingbird", "parrot", "crow", "flamingo", "peacock", "bald eagle", "dove", "snake", "lizard", "alligator", "turtle", "frog", "animal"]
class QueryManager():
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.transformer = SimpleTransformer()

    def save_file(self, file_name, content):
        save_path = os.path.join (self.save_dir, file_name)
        with open (save_path, "w") as save_file:
            save_file.write(content)

    def delete_file(self, file_name):
        save_path = os.path.join (self.save_dir, file_name)
        if os.path.exists(save_path):
            os.remove(save_path)

    def fact_prob_to_file(self, fact_tps, fact_probs):
        scene_tps = []

        (name_tps, attr_tps, rela_tps) = fact_tps
        (name_probs, attr_probs, rela_probs) = fact_probs

        cluster_ntp = {}
        for (oid, name), prob in zip(name_tps, name_probs):
            if not oid in cluster_ntp:
                cluster_ntp[oid] = [(name, prob)]
            else:
                cluster_ntp[oid].append((name, prob))


        for oid, name_list in cluster_ntp.items():
            name_tps = []
            for (name, prob) in name_list:
                # if not name in animals[:5]:
                #     continue
                name_tps.append(f'{prob}::name("{name}", {int(oid)})')
            name_content = ";\n".join(name_tps) + "."
            scene_tps.append(name_content)

        for attr_tp, prob in zip(attr_tps, attr_probs):
            # if not attr_tp[1] == "tall":
            #     continue
            scene_tps.append(f'{prob}::attr("{attr_tp[1]}", {int(attr_tp[0])}).')

        for rela_tp, prob in zip(rela_tps, rela_probs):
            # if not rela_tp[0] == "left":
            #     continue
            scene_tps.append(f'{prob}::relation("{rela_tp[0]}", {int(rela_tp[1])}, {int(rela_tp[2])}).')

        return "\n".join(scene_tps)

    def process_result(self, result):
        output = result.stdout.decode()
        lines = output.split("\n")
        targets = {}
        for line in lines:
            if line == '':
                continue
            if not '\t' in line:
                continue
            info = line.split('\t')
            # No target found
            if 'X' in info[0]:
                break
            target_name = int(info[0][7:-2])
            target_prob = float(info[1])
            targets[target_name] = target_prob
        return targets

    def get_result(self, task, fact_tps, fact_probs):
        timeout = False
        question = task["question"]["clauses"]
        file_name = f"{task['question']['question_id']}.pl"
        save_path = os.path.join (self.save_dir, file_name)
        query = Query(question)
        query_content = self.transformer.transform(query)
        scene_content = self.fact_prob_to_file(fact_tps, fact_probs)

        content = KG+ "\n" + RULES + "\n" + scene_content + "\n" + query_content
        self.save_file(file_name, content)
        try:
            result = subprocess.run(["problog", save_path], capture_output=True, timeout=10)
            targets = self.process_result(result)
        except:
            # time out here
            timeout = True
            targets = {}

        # self.delete_file(file_name)
        return targets, timeout

    def get_relas(self, query):
        relations = []
        for clause in query:
            if 'Relate' in clause['function']:
                relations.append(clause['text_input'])
        return relations



