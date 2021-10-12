from pybloom_live import BloomFilter
from math import log

class Graph:
    def __init__(self, num_vertices, edges):
        self.edges = edges
        self.num_vertices = num_vertices
        self.neighbours = self.generate_neighbours(edges) # key vertex: value [neighobur1, neighbour2]

    def generate_adjacency_matrix(self, edges):
        A = [[0 for x in range(self.num_vertices)] for y in range(self.num_vertices)]
        for edge in edges:
            A[edge[0]][edge[1]] = 1
            # assume undirected graph, so adjacency matrix is symmetric
            A[edge[1]][edge[0]] = 1
        return A

    def generate_neighbours(self, edges):
        n = {}
        for edge in edges:
            if edge[0] in n:
                n[edge[0]].append(edge[1])
            else:
                n[edge[0]] = [edge[1]]

            # assume undirected graph, so adjacency matrix is symmetric
            if edge[1] in n:
                n[edge[1]].append(edge[0])
            else:
                n[edge[1]] = [edge[0]]
        return n

    def one_hop_neighbours(self, vertex):
        return self.neighbours[vertex]

    def generate_dna_encoding(self, depth, filter_capacity):
        # initialize empty bloom filters for each depth
        # set our error rate to 0.1, as done for many demos
        bloom_filters = [[BloomFilter(capacity=filter_capacity, error_rate=0.1) for vertex in range(self.num_vertices)] for y in range(depth + 1)]
        code_length = bloom_filters[0][0].num_bits

        for i in range(self.num_vertices):
            bloom_filters[0][i].add(i)

        # merge bloom filters of d_hop neighbours
        # essentially our message passing step
        # perform the step 'depth' times
        for d in range(1, depth + 1):
            # for each vertex, aggregated its neighbours' bloom filters
            for v in range(self.num_vertices):
                # we also want to aggregate data from the vertex itself,
                # so we add v to its neighbour set to include it in the union.
                # This behaviour is desired as seen in Figure 1
                vertices_to_aggregate = self.one_hop_neighbours(v) + [v]
                for vertex in vertices_to_aggregate:
                    # our aggregation is a union of the filters
                    bloom_filters[d][v] = bloom_filters[d][v].union(bloom_filters[d - 1][vertex])


        embedding = [[0 for y in range(code_length)] for vertex in range(self.num_vertices)]
        # fill our embedding array
        for vertex in range(self.num_vertices):
            for y in range(code_length):
                if bloom_filters[depth][vertex].bitarray[y]:
                    embedding[vertex][y] = 1

        return embedding
