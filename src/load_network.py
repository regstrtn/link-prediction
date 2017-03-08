import os
import sys
import networkx as nx
import matplotlib.pyplot as plt


def dd(G, plotname):

	degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
	#print "Degree sequence", degree_sequence
	dmax=max(degree_sequence)
	x = range(0, dmax+1)
	#plt.loglog(degree_sequence,'b-',marker='o')
	#plt.loglog(degree_sequence)
	plt.plot(degree_sequence[0:100])	
	plt.title(plotname)
	plt.ylabel("degree")
	plt.xlabel("rank")
	plt.axes([0, 100, 0, 1])
	plt.style.use('ggplot')
	# draw graph in inset
	
	#Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)[0]
	#pos=nx.spring_layout(Gcc)
	#plt.axis('off')
	#nx.draw_networkx_nodes(Gcc,pos,node_size=20)
	#nx.draw_networkx_edges(Gcc,pos,alpha=0.4)

	plt.savefig(plotname+".png")
	plt.show()

def visualise(G):
	nx.draw(G)
	plt.draw(G)

def makegraph(filename):
	f = open(filename, 'r')
	total = nx.DiGraph()
	pos = nx.DiGraph()
	neg = nx.DiGraph()
	for i in range(4):
		f.readline()
	i = 0
	for line in f:
		if(i>1000): break
		i += 1
		line = line.rstrip()
		l = line.split()
		total.add_nodes_from([int(l[0]), int(l[1])])
		total.add_edge(int(l[0]), int(l[1]))
		if(int(l[2])<0):
			neg.add_node(int(l[0]))
			neg.add_edge(int(l[0]), int(l[1]))
		else:
			pos.add_node(int(l[0]))
			pos.add_edge(int(l[0]), int(l[1]))
	
	return pos, neg, total

inputfile = "soc-sign-epinions.txt"
pos, neg, total = makegraph(inputfile)
print(pos.number_of_nodes()+ neg.number_of_nodes(), pos.number_of_edges()+neg.number_of_edges())
print(total.number_of_nodes(), total.number_of_edges())

'''
#Degree distribution plots
dd(pos, "posdd")
dd(neg, "negdd")
dd(total, "totaldd")
'''
print(inputfile)
print(nx.info(pos), nx.info(neg))

'''
print("Indegree: ")

indegree = pos.in_degree().values()
print(sum(list(indegree)))
#print(sum(pos.in_degree().values())/float(len(pos)), sum(neg.in_degree().values())/float(len(neg)))

print("Nodes")
print(pos.number_of_nodes(), neg.number_of_nodes())
print("Edges")
print(pos.number_of_edges(), neg.number_of_edges())
'''

print("Degree Centrality: ")
print(sum(nx.algorithms.degree_centrality(pos).values()), sum(nx.algorithms.degree_centrality(neg).values()))

print("Reciprocity: ")
print("Positive: ",nx.algorithms.overall_reciprocity(pos), " Negative: ",nx.algorithms.overall_reciprocity(neg))

