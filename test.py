import torch
import dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class decode_list:
    def __init__(self,hs):
        self.parent=[hs.PL_voc['root']]
        self.name=[hs.PL_voc['root']]
        self.trg=[hs.PL_voc[dataset.SOS]]
        self.pending_node=[]
        self.pending_list=None
        self.cur_idx=0
        self.hs=hs

    def get_pending_node(self):
        node=self.pending_node[0]
        self.pending_node=self.pending_node[1:]
        return node
    
    def update(self,output):
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

def decode(model, src, hs):
    model.eval()
    with torch.no_grad():
        src=torch.tensor([src]).to(device)
        pad_idx=hs.PL_voc[dataset.PAD]
        
        dl=decode_list(hs)
        
        while(dl.cur_idx<hs.max_PL_len+1):
            if not dl.next_step():
                break
            output = model(src, torch.tensor([dl.parent]).to(device), torch.tensor([dl.name]).to(device), torch.tensor([dl.trg]).to(device))
            # TODO: Beam-Search
            _, indices=torch.sort(output[0][dl.cur_idx], descending=True)
            maxi=indices[0].item()
            output=hs.PL_dict[maxi]
            dl.trg.append(maxi)
            print(output)
            dl.update(output)
            input()
        

if __name__=="__main__":
    hs=dataset.hearthstone()
    model=torch.load('model.weights')
    X_test,Y_test=hs.dataset('test')
    for i in range(len(X_test)):
        X=X_test[i]
        print(X)
        decode(model,X,hs)
        input()
