import ast
import astunparse
import dataset
import collections

ignored=['ctx']
def visit(node, args):
    assert isinstance(node,ast.AST)
    fields=list(ast.iter_fields(node))
    for pair in reversed(fields):
        if pair[0] in ignored:
            continue
        args[0].insert(0,(pair[0],pair[1],node))

def node_to_string(node):
    print_field_name=False
    if isinstance(node,ast.AST):
        class_name=node.__class__.__name__
        if not print_field_name:
            return '__'+class_name+'__'
        try:
            field_names=','.join(list(zip(*ast.iter_fields(node)))[0])
        except:
            print(node)
            print(list(ast.iter_fields(node)))
        return '%s(%s)'%(class_name,field_names)
    elif isinstance(node,list):
        return str(list(map(node_to_string,node)))
    else:
        return str(node)

grammar={}
def parse(source, gen_grammar_rule=False):
            global grammar
            tokens=[]
            root=ast.parse(source)
            q=[]
            q.append(('root',root, 'root'))
            while len(q)!=0:
                next_node=q[0]
                q=q[1:]
                name=next_node[0]
                val=next_node[1]
                parent=next_node[2]
                if name in ignored:
                    continue
                parent_string=node_to_string(parent)
                if gen_grammar_rule:
                    if isinstance(parent,ast.AST):
                        if not parent_string in grammar:
                            grammar[parent_string]=collections.OrderedDict()
                            fields=list(ast.iter_fields(parent))
                            for fieldname, value in fields:
                                if fieldname in ignored:
                                    continue
                                #TODO: all valid types
                                grammar[parent_string][fieldname]=type(value)
                
                if isinstance(val,list):
                    for node in val:
                        tokens.append((parent_string,name,node_to_string(node)))
                    tokens.append((parent_string,name,'EOL'))
                else:
                    tokens.append((parent_string,name,node_to_string(val)))
                if isinstance(val,ast.AST):
                    visit(val,[q])
                elif isinstance(val,list):
                    for node in reversed(val):
                        visit(node,[q])
            return tokens

if __name__=="__main__":
    with open('card2code/third_party/hearthstone/train_hs.out') as f:
        for line in f:
            line=line.strip()
            line=line.replace('''§''','\n')
            print(line)
            a=(parse(line))
            for pair in a:
                print(pair)
            input()
