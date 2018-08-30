# CS114 Spring 2018 Homework 4
# Part-of-speech Tagging with Hidden Markov Models
# Alex Reese
# Acc = 0.8805749452728099

import os
from collections import defaultdict
from operator import itemgetter
import math
import threading

class POSTagger():

    def __init__(self):
        self.transition = defaultdict(dict)
        self.emission = defaultdict(dict)
        self.words = defaultdict(int)
        self.results = defaultdict(dict)

    """
    Parses a given line into stuff needed for training. For parallelized train
    """
    def parse_line(self, line, pos_bigrams, emissions, pos_counts):
        line = line.split()
        if len(line) > 0:
            line_parts = [(part.rsplit("/",1)[0],part.rsplit("/",1)[1]) for part in line]
            pos_counts["<S>"] += 1
            pos_counts["</S>"] += 1
            pos_bigrams[("<S>",line_parts[0][1])] += 1
            pos_bigrams[(line_parts[-1][1],"</S>")] += 1
            for i in range(0,len(line_parts) - 1):
                self.words[line_parts[i][0]] += 1
                pos_counts[line_parts[i][1]] += 1
                pos_bigrams[(line_parts[i][1],line_parts[i+1][1])] += 1
                emissions[(line_parts[i][1],line_parts[i][0])] += 1

            self.words[line_parts[-1][0]] += 1
            pos_counts[line_parts[-1][1]] += 1
            emissions[(line_parts[-1][1],line_parts[-1][0])] += 1
    """
    Populates transition dictionaries
    """
    def populate_transitions(self, pos_counts, pos_bigrams): 
        for key in pos_counts:
            for key2 in pos_counts:
                #Make sure start and end states aren't smoothed
                if key == "</S>" or key2 == "<S>" or (key == "<S>" and key2 == "</S>"):
                    self.transition[key][key2] = float('-inf')
                else:
                    self.transition[key][key2] = math.log((pos_bigrams[(key,key2)] + 1) / (pos_counts[key] + len(pos_counts)),2)
    
    """
    Populates emission dictionaries
    """
    def populate_emissions(self, pos_counts, emissions):
        for key in pos_counts:
            for key2 in self.words:
                if key == "</S>" or key == "<S>":
                        self.emission[key][key2] = float('-inf')
                else:
                    self.emission[key][key2] = math.log((emissions[(key,key2)] + 1) / (pos_counts[key] + len(self.words)),2)
    
    '''
    Trains a supervised hidden Markov model on a training set.
    Transition probabilities P(s|p) = C(p,s) / C(p)
    Emission probabilities P(w|s) = C(s,w) / C(s)
    '''
    def train(self, train_set):
        print("Training started.")
        pos_bigrams = defaultdict(int)
        emissions = defaultdict(int)
        pos_counts = defaultdict(int)
        threads = []
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # be sure to split documents into sentences here
                    print("Parsing",f)
                    for line in f:
                        self.parse_line(line,pos_bigrams,emissions,pos_counts)
        self.pos = pos_counts
        # Split off populating each dictionary into its own thread to speed up training
        t_thread = threading.Thread(target=self.populate_transitions, args=[pos_counts, pos_bigrams])
        e_thread = threading.Thread(target=self.populate_emissions, args=[pos_counts, emissions])
        t_thread.start()
        e_thread.start()
        t_thread.join()
        e_thread.join()
        
        # For testing whether probs sum to 1 (only <S> and </S> should appear here):
        for key in pos_counts.keys():
            prob = 0.0
            for key2 in pos_counts.keys():
                prob += 2**self.transition[key][key2]
            if abs(prob - 1) > 0.1:
                print("Transition: " + key + "\t" + str(prob))
            prob = 0.0
            for key2 in self.words.keys():
                prob += 2**self.emission[key][key2]
            if abs(prob - 1) > 0.1:
                print("Emission: " + key + "\t" + str(prob))
        
        print("Training complete")

    '''
    Implements the Viterbi algorithm.
    Use v and backpointer to find the best_path.
    '''
    def viterbi(self, sentence, s_id):
        v = defaultdict(dict)
        backpointer = defaultdict(dict)
        # initialization step
        for tag in self.transition.keys():
            if self.words[sentence[0]] != 0:
                v[tag][0] = self.transition["<S>"][tag] + self.emission[tag][sentence[0]]
            else:
                #UNK handled here
                v[tag][0] = self.transition["<S>"][tag] + math.log((1 / (self.pos["<S>"] + len(self.words))),2)
            backpointer[tag][0] = "<S>"

        # recursion step
        for t in range(1,len(sentence)):
            for tag in self.transition:
                current_max_key = None
                current_max = float('-inf')
                for tag_prime in self.transition:
                    # Find largest tag'
                    if self.words[sentence[t]] != 0:
                        num = v[tag_prime][t-1] + self.transition[tag_prime][tag] + self.emission[tag][sentence[t]]
                    else:
                        #UNK handled here
                        num = v[tag_prime][t-1] + self.transition[tag_prime][tag] + math.log((1 / (self.pos[tag] + len(self.words))),2)
                    if num > current_max:
                        current_max = num
                        current_max_key = tag_prime

                v[tag][t] = current_max             
                backpointer[tag][t] = current_max_key
        
        # termination steps
        current_max_key = None
        current_max = float('-inf')
        for tag_prime in self.transition:
            num = v[tag_prime][len(sentence)-1] + self.transition[tag_prime]["</S>"]
            if num > current_max:
                current_max = num
                current_max_key = tag_prime
        v["</S>"][len(sentence)] = current_max
        backpointer["</S>"][len(sentence)] = current_max_key
        
        best_path = []
        best_path.insert(0,backpointer["</S>"][len(sentence)])
        tag = backpointer["</S>"][len(sentence)]
        t = len(sentence) - 1
        while tag != "<S>":
            tag = backpointer[tag][t]
            best_path.insert(0,tag)
            t = t - 1
        # Was keeping <S> so chop it off here
        self.results[s_id]['predicted'] = best_path[1:]
        print(s_id + " completed.")

    '''
    Tests the tagger on a development or test set.
    Returns a dictionary of sentence_ids mapped to their correct and predicted
    sequences of POS tags such that:
    results[sentence_id]['correct'] = correct sequence of POS tags
    results[sentence_id]['predicted'] = predicted sequence of POS tags
    '''
    def test(self, dev_set):
        print("Testing.")
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            file_counter = 0
            for name in files:
                file_counter += 1
                line_counter = 0
                with open(os.path.join(root, name)) as f:
                    for line in f:
                        if len(line.split()) > 0:
                            line_counter += 1
                with open(os.path.join(root, name)) as f:
                    counter = 0
                    threads = []
                    for line in f:
                        line = line.split()
                        if len(line) > 0:
                            s_id = name + "_" + str(counter)
                            pos = [part.split("/")[1] for part in line]

                            words = [part.split("/")[0] for part in line]
                            self.results[s_id]['correct'] = pos
                            self.viterbi(words,s_id)
                            counter += 1
                    print("File " + str(file_counter) + " completed.")
        return self.results

    '''
    Given results, calculates overall accuracy
    '''
    def evaluate(self, results):
        total = 0
        correct = 0
        for sentence in results:
            if len(results[sentence]['correct']) == len(results[sentence]['predicted']):
                for i in range(0,len(results[sentence]['correct'])):
                    if results[sentence]['correct'][i] == results[sentence]['predicted'][i]:
                        correct += 1
                    total += 1
            else:
                print("Something went wrong!")
        return correct / total

if __name__ == '__main__':
    pos = POSTagger()
    # make sure these point to the right directories
    pos.train('brown/train')
    results = pos.test('brown/dev')
    print(pos.evaluate(results))
