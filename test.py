import torch
import dataset
import queue
from functools import reduce
import math
import torch.nn.functional as F
import pythonparser
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class decode_list:
    def __init__(self,hs):
        self.parent=[hs.PL_voc['root']]
        self.name=[hs.PL_voc['root']]
        self.trg=[hs.PL_voc[dataset.SOS]]
        self.pending_node=[]
        self.pending_list=None
        self.cur_idx=0
        self.priority=0
        self.p_list=[]
        self.hs=hs

    def get_pending_node(self):
        node=self.pending_node[0]
        self.pending_node=self.pending_node[1:]
        return node

    def copy(self):
        ret=decode_list(hs)
        ret.parent=self.parent.copy()
        ret.name=self.name.copy()
        ret.trg=self.trg.copy()
        ret.pending_node=self.pending_node.copy()
        if self.pending_list==None:
            ret.pending_list=None
        else:
            ret.pending_list=self.pending_list.copy()
        ret.cur_idx=self.cur_idx
        ret.priority=self.priority
        ret.p_list=self.p_list.copy()
        return ret

    def update(self,outputi):
        output=hs.PL_dict[outputi]
        self.trg.append(outputi)
        if self.pending_list!=None:
            if output=='EOL':
                for node in reversed(self.pending_list):
                    if node in self.hs.grammar:
                        fields=self.hs.grammar[node].keys()
                        self.pending_node=[(node,field) for field in fields]+self.pending_node
                self.pending_list=None
            else:
                self.pending_list.append(output)
        else:
            if output in self.hs.grammar:
                fields=self.hs.grammar[output].keys()
                self.pending_node=[(output,field) for field in fields]+self.pending_node
        self.cur_idx+=1

    def update_priority(self,p):
        self.p_list.append(p)
        self.priority = sum(self.p_list) / len(self.p_list)


    def next_step(self):
        if self.cur_idx != 0:
            if self.pending_list==None:
                if len(self.pending_node)==0:
                    # finish decoding
                    return False
                next_node=self.get_pending_node()
                self.parent.append(self.hs.PL_voc[next_node[0]])
                self.name.append(self.hs.PL_voc[next_node[1]])
                if self.hs.grammar[next_node[0]][next_node[1]]==list:
                    self.pending_list=[]
            else:
                self.parent.append(self.parent[-1])
                self.name.append(self.name[-1])
        return True

    def __lt__(self, other):
        return self.priority < other.priority

def decode(model, src, hs, window_size=2):
    model.eval()
    ret=queue.PriorityQueue()
    with torch.no_grad():
        src=torch.tensor([src]).to(device)
        pad_idx=hs.PL_voc[dataset.PAD]

        dl=decode_list(hs)
        candidate = None
        priority_list = [[None] * window_size, [None] * window_size]
        priority_list[0][0] = dl
        round = 0
        while(True):
            #print(1)
            #print(round)
            if(round>=hs.max_PL_len+1):
                break
            for i in range(window_size):
                priority_list[(round + 1) % 2][i] = None
            for i in range(window_size):
                if priority_list[round % 2][i] == None:
                    continue
                dl = priority_list[round % 2][i]
                #print(dl.cur_idx)
                if dl.cur_idx>=hs.max_PL_len+1 or not dl.next_step():
                    if candidate == None or candidate.priority < dl.priority:
                        candidate = dl.copy()
                    continue
                output = model(src, torch.tensor([dl.parent]).to(device), torch.tensor([dl.name]).to(device), torch.tensor([dl.trg]).to(device))
                output=output[0][dl.cur_idx]
                output=F.log_softmax(output, dim=0)
                _, indices=torch.sort(output, descending=True)
                for j in range(window_size):
                    dl2 = dl.copy()
                    index=indices[j].item()
                    dl2.update(index)
                    dl2.update_priority(output[index].item())
                    mininum_index = 0
                    for k in range(window_size):
                        if priority_list[(round + 1) % 2][mininum_index] == None:
                            break
                        if priority_list[(round + 1) % 2][k] == None or priority_list[(round + 1) % 2][k].priority < priority_list[(round + 1) % 2][mininum_index].priority:
                            mininum_index = k
                    if priority_list[(round + 1) % 2][mininum_index] == None or priority_list[(round + 1) % 2][mininum_index].priority < dl2.priority:
                        priority_list[(round + 1) % 2][mininum_index] = dl2
            round += 1
        return candidate

def seq2tree(trg, hs):
    pass

if __name__=="__main__":
    import nltk
    hs=dataset.hearthstone()
    model=torch.load('model.weights')
    X_test,Y_test=hs.dataset('test')
    failcnt=0
    blues=[]
    for i in range(len(X_test)):
        X=X_test[i]
        dl=decode(model,X,hs)
        trg=dl.trg
        #print(trg)
        ref=[c for c in Y_test[i][2] if c!=0]
        #print(oracle)
        blue=nltk.translate.bleu_score.sentence_bleu([ref],trg)
        print(blue)
        blues.append(blue)
        #input()
        continue
        if trg[0]==3:
            trg=trg[1:]
        node=pythonparser.seq2tree(trg, hs)
        import ast
        import astunparse
        #print(ast.dump(node))
        try:
            code=astunparse.unparse(node)
            print(code)
        except:
            failcnt+=1
            print(ast.dump(node))
        #print(astunparse.unparse(node))
        
        input()
    print(failcnt)
    print(sum(blues)/len(blues))
