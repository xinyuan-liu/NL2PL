import torch
import dataset
import queue
from functools import reduce
import math
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
        #for i in self.p_list:
            
        self.priority= sum(map(math.log,self.p_list)) / len(self.p_list)
        

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
        while(True):
            if dl.cur_idx>=hs.max_PL_len+1 or not dl.next_step():
                return dl
            output = model(src, torch.tensor([dl.parent]).to(device), torch.tensor([dl.name]).to(device), torch.tensor([dl.trg]).to(device))
            output=output[0][dl.cur_idx]
            _, indices=torch.sort(output, descending=True)
            index=indices[i].item()
            dl.update(index)
            dl.update_priority(output[index].item())

def seq2tree(trg, hs):
    pass

if __name__=="__main__":
    hs=dataset.hearthstone()
    model=torch.load('model.weights')
    X_test,Y_test=hs.dataset('test')
    for i in range(len(X_test)):
        X=X_test[i]
        dl=decode(model,X,hs)
        #decode(model,X,hs)
        print(dl.trg)
        input()
