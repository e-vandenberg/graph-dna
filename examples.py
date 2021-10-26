from graph_dna import Graph

# Figure 1 (page 5)
figure_1 = Graph(5, [[0,1], [1,2], [2,3], [2,4]])

# we see that vertices 3,4 and 1,2 respectively get the same embeddings with d=2
# just as depicted in Figure 1
dna_encoding = figure_1.generate_dna_encoding(depth=2, filter_capacity=3)
print(dna_encoding)

# construct our meta bipartite graph and display the set of edges
meta_graph = figure_1.generate_meta_graph(dna_encoding)
print(meta_graph.edges)
