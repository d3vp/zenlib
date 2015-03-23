from zen.graph cimport Graph

from numpy cimport ndarray
from cpython cimport bool

cdef class Community:
	cdef object _graph
	cdef set _nodes
	cdef readonly int community_idx
	cdef dict _probabilities

	cpdef bool has_node_(Community self, int nidx)

cdef class CommunitySet:
	cdef object _graph
	cdef int _num_communities
	cdef ndarray _communities

	cdef void __raise_if_invalid_nidx(CommunitySet self, int nidx) except *
	cdef Community __build_community(CommunitySet self, int cidx)
	cdef list __build_all_communities(CommunitySet self)
	cpdef node_community_(CommunitySet self, int nidx)
	cpdef int node_community_index_(CommunitySet self, int nidx)
	cpdef bool check_sharing_(CommunitySet self, int u_idx, int v_idx)

