## Graph DNA

An implementation of an alternative embedding algorithm for graph data.
Based on the paper [Graph DNA: Deep Neighborhood Aware Graph Encoding for Collaborative Filtering](https://arxiv.org/pdf/1905.12217.pdf) (Liwei Wu et. al)

### The Goal


The goal of Graph DNA is to create a memory and time efficient way to embed graph data into a format that is more easily consumed by Graph Neural Networks, while still preserving as much structural insight as possible. 

<p align="center">
  <img src="/dna_demo.png" alt="drawing1" width="600"/>
  <br>
  <sub><sup>(https://arxiv.org/pdf/1905.12217.pdf, Liwei Wu et. al)</sup></sub>
</p>

Existing methods localize vertices with message passing using adjacency matrix multiplication, but have trouble gathering structural insight
from graphs beyond 1-2 hops without pushing exponential time and space limits. 

<p align="center">
  <img src="https://www.outsystems.com/blog/-/media/images/blog/posts/graph-neural-networks/nn-gif-5.gif?h=393&w=750&updated=20190809155049" alt="drawing2" width="500"/>
</p>

Graph DNA uses bloom filters and gossip algorithms to gain deeper neighbourhood insight at a much lower cost.

### The Bigger Picture

Graph Convolutional Networks are extremely powerful tools in biology, social networks & eCommerce with applications including 
drug interaction prediction, and advanced reccomendation systems. 

Demonstrating more time and memory efficient methods for creating graph embeddings is essential in pushing forward these advancments,
and making graph data more easily consumed by neural networks. 


### Examples

As seen in `examples.py` we can recreate the graph on page 5, and generate its DNA encoding as follows:
  
  ```
  figure_1 = Graph(5, [[0,1], [1,2], [2,3], [2,4]])
  print(figure_1.generate_dna_encoding(depth=2, filter_capacity=3))
  
  >>> [[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], 
       [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
       [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], 
       [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1], 
       [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1]]
  ```
  
The output is the generated embedding for each vertex. We can see that vertices 3,4 and 1,2 respectively get the same embeddings with d=2 just as depicted in Figure 1 in [the paper](https://arxiv.org/pdf/1905.12217.pdf).

