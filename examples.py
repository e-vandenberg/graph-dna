from dna_graph import Graph

# Figure 1 (page 5)
figure_1 = Graph(5, [[0,1], [1,2], [2,3], [2,4]])

# we see that vertices 3,4 and 1,2 respectively get the same embeddings with d=2
# just as depicted in Figure 1
print(figure_1.generate_dna_encoding(depth=2, filter_capacity=3))
