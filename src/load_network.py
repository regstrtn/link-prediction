from __future__ import division
import os
import sys
import networkx as nx
#import matplotlib.pyplot as plt
import numpy as np
import random
#from sklearn.metrics import *
import math

inputfile = "../data/soc-sign-Slashdot090221.txt"
epinion_nodes = 82144


random.seed(100)

'''
G = nx.gnp_random_graph(100,0.02)

degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
#print "Degree sequence", degree_sequence
dmax=max(degree_sequence)

plt.loglog(degree_sequence,'b-',marker='o')
plt.title("Degree rank plot")
plt.ylabel("degree")
plt.xlabel("rank")

# draw graph in inset
plt.axes([0.45,0.45,0.45,0.45])
Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)[0]
pos=nx.spring_layout(Gcc)
plt.axis('off')
nx.draw_networkx_nodes(Gcc,pos,node_size=20)
nx.draw_networkx_edges(Gcc,pos,alpha=0.4)

plt.savefig("degree_histogram.png")
plt.show()
Next  Previous


   seq = list(seq.values())
   plt.rcParams['font.size'] = 10
   plt.hist(seq, normed = 1, bins = 10)
   plt.show()
   #plt.axes([0, 40000, 0, 0.5])
   plt.title("Degree Centrality Distribution")
   plt.xlabel("Degree Centrality")
   plt.ylabel("Fraction of nodes")
   #plt.savefig("../plots/"+filename)
'''
def dd(G, plotname):

    plt.style.use('ggplot')
    degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
    #print "Degree sequence", degree_sequence
    dmax=max(degree_sequence)
    x = range(0, dmax+1)
    #plt.loglog(degree_sequence)
    #plt.loglog(degree_sequence)
    #plt.plot(degree_sequence[0:200])   
    '''plt.title(plotname.split("plots/")[1], fontsize = 20)
    plt.ylabel("degree", fontsize = 20)
    plt.xlabel("Degree", fontsize = 20)
    plt.axes([0,100,0,100])
    #plt.axis([0, 100, 0, 1])
    plt.style.use('ggplot')'''
    # draw graph in inset
    plt.rcParams['font.size'] = 10
    x = np.log(range(1,len(degree_sequence)))
    plt.hist(degree_sequence, x, normed = 1, bins = 100, log = True)
    #plt.axes([0, 40000, 0, 0.5])
    plt.title("Degree Centrality Distribution")
    plt.xlabel("Degree Centrality")
    plt.ylabel("Fraction of nodes")
    #Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)[0]
    #pos=nx.spring_layout(Gcc)
    #plt.axis('off')
    #nx.draw_networkx_nodes(Gcc,pos,node_size=20)
    #nx.draw_networkx_edges(Gcc,pos,alpha=0.4)
    #plt.savefig(plotname+".png")
    plt.show()





# In[34]:

def bfsnodes(graph, source, hops = 1):
    bfs_tree = nx.bfs_tree(graph,0)
    bfs_nodes = list(bfs_tree.nodes())
    return bfs_nodes[0:5000]


# In[35]:

def gettraindata(pos, numnodes):
    #randnodes = random.sample(range(0, numnodes-1), 20000)
    #randnodes = bfsnodes(pos, 0, 1)
    randnodes = list(pos.nodes())
    #print("posedges length : ",len(posedges))
    pos = pos.subgraph(randnodes)
    posedges = list(pos.edges())
    
    random.shuffle(posedges)
    split = int(len(posedges)*0.8)
    postrain = posedges[0:split]
    postest = posedges[split:len(posedges)]
    pos.remove_edges_from(postest)
    
    print(len(posedges))
    return pos, postrain, postest


# In[36]:

def common_neighbors(g, i, j):
    out_nbrs =  set(g.successors(i)).intersection(g.successors(j))
    in_nbrs = set(g.predecessors(i)).intersection(g.predecessors(j))
    return out_nbrs.union(in_nbrs)

def resource_allocation_index(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)

    def predict(u, v):
        return sum(1 / G.degree(w) for w in common_neighbors(G, u, v))

    return ((u, v, predict(u, v)) for u, v in ebunch)
    


# In[37]:

def predictedges(train, edgelist):
    #algorithm = resource allocation
    prededges = nx.resource_allocation_index(train)
    predset = set()
    testset = set(edgelist)
    i = 0
    prededges = list(prededges)
    prededges = sorted(prededges, key=lambda x: -x[2])
    for u, v, e in prededges[0:5000000]:
        if(i%500000==0): print("I: ", i)
        i += 1
        predset.add((u,v,e))
    return predset


# In[38]:

def makegraph(filename, numnodes=0):
    f = open(filename, 'r')
    total = nx.Graph()
    pos = nx.Graph()
    neg = nx.Graph()
    for i in range(4):
        f.readline()
    i = 0
    for line in f:
        #if(i>100000): break
        i += 1
        line = line.rstrip()
        l = line.split()
        total.add_nodes_from([int(l[0]), int(l[1])])
        total.add_edge(int(l[0]), int(l[1]), weight = int(l[2]))
        if(int(l[2])<0):
            neg.add_node(int(l[0]))
            neg.add_edge(int(l[0]), int(l[1]), weight = int(l[2]))
        else:
            pos.add_node(int(l[0]))
            pos.add_edge(int(l[0]), int(l[1]), weight = int(l[2]))
            
    return pos, neg, total




# In[8]:


def visualise(G):
    nx.draw(G)
    plt.draw(G)
    


# In[ ]:
pos, neg, total = makegraph(inputfile, epinion_nodes)
postrain, edgetrain, edgetest = gettraindata(total, epinion_nodes)
# In[22]:

predset = predictedges(postrain, edgetest)
predset = list(predset)
predset = sorted(predset, key=lambda x: -x[2])


# In[30]:

predfile = open("predicted_node_pairs", 'w')
for u,v,e in predset:
    predfile.write(str(u)+' '+str(v)+' '+str(e)+'\n')


# In[23]:

prededges = set([(u,v) for u,v,e in predset])


# In[24]:

predscore = []
actuals = []
i=0
for u,v,e in predset[0:1000000]:
    if(i%200000==0): print("I: ",i)
    i+=1
    predscore.append(e)
    if((u,v) in edgetest):
        actuals.append(1)
    else:
        actuals.append(0)
from sklearn.metrics import *
#sklearn.metrics.roc_auc_score(y_true, y_score, average='macro', sample_weight=None)
auc = roc_auc_score(actuals, predscore)
print("auc: ",auc)


# In[29]:

#precision_recall_curve(y_true, probas_pred, pos_label=None, sample_weight=None)
precision, recall, thresholds = precision_recall_curve(actuals, predscore)


# In[26]:

print(len(edgetest), len(set(edgetest) & set(prededges)))


plt.plot(recall, precision)
plt.savefig("../plots/precision_recall_ra.png")
plt.show()

