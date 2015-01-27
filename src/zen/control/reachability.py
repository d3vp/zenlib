"""
This modules provides routines that concern reachability characteristics of 
a system and a control set.
"""

from zen import DiGraph, maximum_matching_
from zen.exceptions import type_check
import zen

def num_min_controls(G):
	"""
	Return the smallest number of controls that are required to control the graph ``G``
	assuming structural controllability conditions.
	"""
	type_check(G,DiGraph)
	
	matched_edges = None
	
	matched_edges = maximum_matching_(G)
	
	return max([len(G) - len(matched_edges),1])


def generic_rank_(G,controls):
	"""
	Find the maximum matching for the directed graph such that all 
	matching paths (vis a vis the Hopcroft Karp algorithm) start with one of the seeds.

	Return the generic rank.
	"""

	#####
	# apply the transformation to produce a bipartite graph which combines the controls
	GT = zen.BipartiteGraph()
	tnode2node = {}
	node2unode = {}
	node2vnode = {}
	vnode2unode = {}

	#####
	# first transform the graph itself
	for i in G.nodes_iter_():
		unode = GT.add_u_node()
		vnode = GT.add_v_node()
		tnode2node[unode] = i
		tnode2node[vnode] = i
		node2unode[i] = unode
		node2vnode[i] = vnode
		vnode2unode[vnode] = unode

	# add the edges
	for i in G.edges_iter_():
		u,v = G.endpoints_(i)
		GT.add_edge_(node2unode[u],node2vnode[v],i)

	# now add the controls in
	for control in controls:
		cnode = GT.add_u_node()
		for v in control:
			GT.add_edge_(cnode,node2vnode[v])

	#####
	# run the bipartite matching
	max_matching = zen.matching.hopcroft_karp_(GT)

	return len(max_matching)
	

def kalman_generic_rank(G,controls,repeats=100):
	"""
	Finds the reachability corresponding to a graph (adjacency matrix A) and its
	controls (a matrix B) by brute force computing the Kalman rank condition:
		rank [B AB A^2B A^3B ... A^(n-1)B].
	In order to compute the rank generically, we generate random entries for A and B,
	subject to their zero/non-zero sparcity patterns and compute the true rank. We
	repeat this "repeats" times (default is 100) and return the largest value.
	"""
	from numpy import array, zeros, nonzero, eye, dot
	from numpy.linalg import matrix_rank
	from numpy.random import rand as nprand
	
	rank = 0
	N = G.max_node_idx+1 # there could be some missing indexes in the graph (N > n)
	n = G.num_nodes
	A = G.matrix().T
	
	# build the B matrix
	m = len(controls)
	B = zeros((N,m))
	for i in range(m):
		for d_idx in controls[i]:
			B[d_idx,i] = 1
	
	nonzero_idxs = nonzero(A>0)
	num_nonzero_idxs = len(nonzero_idxs[0])
	for r in range(repeats):
		# create a randomized instance of A
		A1 = zeros((N,N))
		A1[nonzero_idxs] = nprand(num_nonzero_idxs)
		
		# compute the controllability matrix
		An = eye(N)
		C = zeros((N,m*N))
		C[:,0:m] = B
		for i in range(1,N):
			An = dot(An,A1)
			C[:,i*m:i*m+m] = dot(An,B)
			
		# generic rank is the max of all instance ranks
		new_rank = matrix_rank(C)
		if new_rank > rank:
			rank = new_rank
	
	return rank

