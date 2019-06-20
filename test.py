import torch
import dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def decode(model, src, hs):
    model.eval()
    with torch.no_grad():
        src=torch.tensor([src]).to(device)
        pad_idx=hs.PL_voc[dataset.PAD]
        parent=[hs.PL_voc['root']]
        name=[hs.PL_voc['root']]
        trg=[hs.PL_voc[dataset.SOS]]
        
        pending_node=[]
        pending_list=None
        for i in range(hs.max_PL_len+1):
            if i != 0:
                if pending_list==None:
                    if len(pending_node)==0:
                        break
                    next_node=pending_node[0]
                    pending_node=pending_node[1:]
                    parent.append(hs.PL_voc[next_node[0]])
                    name.append(hs.PL_voc[next_node[1]])
                    if hs.grammar[next_node[0]][next_node[1]]==list:
                        pending_list=[]
                else:
                    parent.append(parent[-1])
                    name.append(name[-1])
            output = model(src, torch.tensor([parent]).to(device), torch.tensor([name]).to(device), torch.tensor([trg]).to(device))
            # TODO: Beam-Search
            maxi=output[0][i].max(-1)[1].item()
            _, indices=torch.sort(output[0][i], descending=True)
            #print(maxi)
            #print(indices[0])
            output=hs.PL_dict[maxi]
            trg.append(maxi)
            print(output)
            if pending_list!=None:
                if output=='EOL':
                    for node in reversed(pending_list):
                        if node in hs.grammar:
                            fields=hs.grammar[node].keys()
                            pending_node=[(node,field) for field in fields]+pending_node
                    pending_list=None
                else:
                    pending_list.append(output)
            else:
                if output in hs.grammar:
                    fields=hs.grammar[output].keys()
                    pending_node=[(output,field) for field in fields]+pending_node
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
