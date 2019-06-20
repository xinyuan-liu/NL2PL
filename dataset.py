import pythonparser
import nltk
import ast
import numpy as np
def split(string):
    pattern =r'''(?x)
    <[^>]+>
    | \w+
    | [][.,;"'?():-_`]
    '''
    return nltk.regexp_tokenize(string, pattern)

UNK='_UNK_'
PAD='_PAD_'
SOS='_SOS_'
EOS='_EOS_'
preserve=[SOS,EOS,UNK,PAD]
def build_voc(tokens,min_count=0):
    voc_cnt={}
    for l in tokens:
        for token in l:
            if type(token)==str:
                if token in voc_cnt:
                    voc_cnt[token]+=1
                else:
                    voc_cnt[token]=1
            elif type(token)==tuple:
                for word in token:
                    if word in voc_cnt:
                        voc_cnt[word]+=1
                    else:
                        voc_cnt[word]=1

    voc_cnt=list(voc_cnt.items())
    voc_cnt.sort(key=lambda x:x[1])
    voc_list=[pair[0] for pair in voc_cnt if pair[1]>=min_count]
    voc_list+=preserve
    voc_list.reverse()
    voc_dict=dict(zip(voc_list,range(len(voc_list))))
    #print(voc_list[:15])
    return voc_dict

def token2index(token,voc):
    if token in voc:
        return voc[token]
    else:
        return voc[UNK]

def maxlen(X):
    maxlen=0
    for x in X:
        if len(x)>maxlen:
            maxlen=len(x)
    return maxlen

bug_fix={'''if self.target.health <= player.effective_spell_damage(2) and \ (isinstance(self.target, Minion) and not self.target.divine_shield):''':
        '''if self.target.health <= player.effective_spell_damage(2) and (isinstance(self.target, Minion) and not self.target.divine_shield):'''
        }

class hearthstone:
    def __init__(self,
            train_in='card2code/third_party/hearthstone/train_hs.in',
            test_in='card2code/third_party/hearthstone/test_hs.in',
            dev_in='card2code/third_party/hearthstone/dev_hs.in',
            train_out='card2code/third_party/hearthstone/train_hs.out',
            test_out='card2code/third_party/hearthstone/test_hs.out',
            dev_out='card2code/third_party/hearthstone/dev_hs.out'):
        
        self.train_in_token=self.parse_in(train_in)
        self.test_in_token=self.parse_in(test_in)
        self.dev_in_token=self.parse_in(dev_in)
        self.NL_voc=build_voc(self.train_in_token, min_count=2)
        print("len(NL_voc)=%s"%len(self.NL_voc))

        self.train_out_token=self.parse_out(train_out)
        self.test_out_token=self.parse_out(test_out)
        self.dev_out_token=self.parse_out(dev_out)
        #print(maxlen(self.train_in_token))
        print(maxlen(self.train_out_token))
        self.PL_voc=build_voc(self.train_out_token, min_count=2)
        print("len(PL_voc)=%s"%len(self.PL_voc))

        self.PL_dict={}
        for key in self.PL_voc:
            self.PL_dict[self.PL_voc[key]]=key

        self.shift=True
        self.max_NL_len=50
        self.max_PL_len=300

    def dataset(self,name):
        if name=="train":
            X=self.train_in_token
            Y=self.train_out_token
        if name=="dev":
            X=self.dev_in_token
            Y=self.dev_out_token
        if name=="test":
            X=self.test_in_token
            Y=self.test_out_token
        print("len(%s)=%s"%(name,len(X)))
        assert len(X)==len(Y)
        X_arr=[]
        for NL in X:
            NL=NL[:self.max_NL_len]
            NL=[SOS]+NL+[EOS]
            NL=NL+[PAD]*(self.max_NL_len+2-len(NL))
            NL=[token2index(word,self.NL_voc) for word in NL]
            X_arr.append(NL)
        X_arr=np.array(X_arr)
        print(X_arr.shape)
        Y_arr=[]
        PL_cut_cnt=0
        for PL in Y:
            if len(PL)>self.max_PL_len:
                PL_cut_cnt+=1
            PL=PL[:self.max_PL_len]
            PL=[(SOS,SOS,SOS)]+PL+[(EOS,EOS,EOS)]
            #PL=PL+[(PAD,PAD,PAD)]+(self.max_PL_len+2-len(PL))
            #print (PL)
            parent=[node[0] for node in PL]
            name=[node[1] for node in PL]
            trg=[node[2] for node in PL]
            if self.shift:
                parent=parent[1:]
                name=name[1:]
                trg=trg[:-1]
                assert len(name)==len(trg)
            parent=parent+[PAD]*(self.max_PL_len+2-len(parent))
            name=name+[PAD]*(self.max_PL_len+2-len(name))
            trg=trg+[PAD]*(self.max_PL_len+2-len(trg))
            
            parent=[token2index(word,self.PL_voc) for word in parent]
            name=[token2index(word,self.PL_voc) for word in name]
            trg=[token2index(word,self.PL_voc) for word in trg]
            Y_arr.append([parent,name,trg])
        Y_arr=np.array(Y_arr)
        print(Y_arr.shape)
        print("overlong PL:%s"%PL_cut_cnt)
        return X_arr,Y_arr

    def parse_out(self,path):
        outputs=[]
        bug_cnt=0
        with open(path) as f:
            for line in f:
                src=line.strip()
                src=src.replace('''§''','\n')
                for key in bug_fix:
                    if key in src:
                        bug_cnt+=1
                        src=src.replace(key,bug_fix[key])
                try:
                    outputs.append(pythonparser.parse(src))
                except:
                    print(src)
                    print(line)
                    input()
        return outputs

    def parse_in(self,path):
        inputs=[]
        with open(path) as f:
            for line in f:
                inputs.append(split(line.strip()))
        return inputs
    
if __name__=="__main__":
    hs=hearthstone()
    X_train,Y_train=hs.dataset('train')
    print(X_train)
    X_train,Y_train=hs.dataset('dev')
