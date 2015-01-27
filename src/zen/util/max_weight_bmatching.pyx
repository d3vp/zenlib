#cython: boundscheck=False, wraparound=False, nonecheck=False, infer_types=True

import zen, numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from time import time
from os import getpid
from numpy.random import randint

from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from cython.operator cimport dereference as dref
from cython.operator cimport preincrement as incr


cdef extern from "max_weight_matching.hpp":
    cdef struct edge_t:
        int src, tgt
        double w
        edge_t(int, int, double)
    cdef void mwb_matching(vector[vector[int]] &G, vector[edge_t] &E, \
		vector[int] &U, vector[int] &V, vector[int] &M)



def control_rooted_max_weight_matching(Gi,controls, doshuffle=False, indcyc=False):
    cdef:
        vector[int] *U
        vector[int] *V
        vector[edge_t] *E
        vector[vector[int]] *G
        vector[int].iterator it
        vector[edge_t].iterator ite
        unsigned int u,v,N,t,x,y,r,s,e
        double w,MAXW
        edge_t temp_edge
        vector[int] *M

    U = new vector[int]()
    V = new vector[int]()
    M = new vector[int]()
    E = new vector[edge_t]()

    if controls is None:
        controls = []
    node2uv, uv2node = {}, {}

    np.random.seed(int(time()+getpid()*np.random.random()*1000))

    #remove non reachable nodes
    if controls:
        vis = set()

        def dfs(r):
            vis.add(r)
            for v in Gi.out_neighbors_(r):
                if v not in vis:
                    dfs(v)

        for ctl in controls:
            for driver in ctl:
                if driver not in  vis:
                    dfs(driver)
    else:
        vis = set(Gi.nodes_())


    # first transform the graph itself
    N = 0
    #for t in vis:
    for t in Gi.nodes_iter_():
        if indcyc or t in vis:
            u, v = 2*N, 2*N+1
            node2uv[t] = N
            uv2node[N] = t
            N += 1
            U.push_back(u)
            V.push_back(v)
            if doshuffle:
                x, y = randint(U.size()), U.size()-1
                s = dref(U)[x]
                dref(U)[x] = dref(U)[y]
                dref(U)[y] = s
                x, y = randint(V.size()), V.size()-1
                s = dref(V)[x]
                dref(V)[x] = dref(V)[y]
                dref(V)[y] = s


    MAXW = sum(len(ctl) for ctl in controls) + Gi.size() + 100.0

    # add the edges
    for e in Gi.edges_iter_():
        x, y = Gi.endpoints_(e)
        #if x in vis and y in vis:
        if indcyc or (x in vis and y in vis):
            u,v = 2*node2uv[x], 2*node2uv[y]+1
            temp_edge.src = u
            temp_edge.tgt = v
            temp_edge.w = MAXW + 1.0
            E.push_back(temp_edge)


    # add control nodes and forward edges with weight 1
    START_CNODE = N
    for ctl in controls:
        cnode = N
        N += 1
        #if len(ctl) > 0:
            #d = ctl[0]
        for d in ctl:
            u, v = 2*cnode, 2*node2uv[d]+1
            temp_edge.src = u
            temp_edge.tgt = v
            temp_edge.w = MAXW + 1.0
            E.push_back(temp_edge)

        for x in vis:
            u, v = 2*node2uv[x], 2*cnode+1
            temp_edge.src = u
            temp_edge.tgt = v
            temp_edge.w = MAXW
            E.push_back(temp_edge)

        u, v = 2*cnode, 2*cnode+1
        U.push_back(u)
        V.push_back(v)
        if doshuffle:
            x, y = randint(U.size()), U.size()-1
            s = dref(U)[x]
            dref(U)[x] = dref(U)[y]
            dref(U)[y] = s
            x, y = randint(V.size()), V.size()-1
            s = dref(V)[x]
            dref(V)[x] = dref(V)[y]
            dref(V)[y] = s


    # add self loops with weight 0
    for s in xrange(N):
        if s >= START_CNODE or not Gi.has_edge_(uv2node[s],uv2node[s]):
            u, v = 2*s, 2*s+1
            temp_edge.src = u
            temp_edge.tgt = v
            temp_edge.w = MAXW
            E.push_back(temp_edge)

    G = new vector[vector[int]](2*N, vector[int]())

    for e in xrange(E.size()):
        u,v,w = dref(E)[e].src, dref(E)[e].tgt, dref(E)[e].w
        assert u%2==0 and v%2==1
        dref(G)[u].push_back(e)
        dref(G)[v].push_back(e)
        if doshuffle:
            x, y = randint(dref(G)[u].size()), dref(G)[u].size()-1
            s = dref(G)[u][x]
            dref(G)[u][x] = dref(G)[u][y]
            dref(G)[u][y] = s
            x, y = randint(dref(G)[v].size()), dref(G)[v].size()-1
            s = dref(G)[v][x]
            dref(G)[v][x] = dref(G)[v][y]
            dref(G)[v][y] = s

    #####
    # run the weighted bipartite matching

    mwb_matching(dref(G),dref(E),dref(U),dref(V),dref(M))

    result, roots = [], []
    num_matched = 0
    for x in xrange(M.size()):
        e = dref(M)[x]
        u,v,w = dref(E)[e].src, dref(E)[e].tgt, dref(E)[e].w
        assert u%2==0 and v%2==1
        if w > MAXW:
            num_matched += 1
            if (u/2 < START_CNODE):
                result.append(Gi.edge_idx_(uv2node[u/2],uv2node[v/2]))
            else:
                roots.append(uv2node[v/2])



    #  free memory
    del G, U, V, E, M

    return num_matched, result, roots


if __name__ == '__main__':

    print 'Hello World'

    '''
    print 'MAXW = ', MAXW
    print 'num_matched = ', num_matched
    print 'U nodes--------'
    it = U.begin()
    while it != U.end():
        print dref(it)
        incr(it)

    print 'V nodes--------'
    it = V.begin()
    while it != V.end():
        print dref(it)
        incr(it)


    ite = E.begin()
    while ite != E.end():
        u,v,w = dref(ite).src, dref(ite).tgt, dref(ite).w
        incr(ite)
        print u/2,v/2,w

    for i in xrange(G.size()):
        print i/2 ,' ----> ',
        it = dref(G)[i].begin()
        while it != dref(G)[i].end():
            u,v,w = dref(E)[dref(it)].src,dref(E)[dref(it)].tgt,dref(E)[dref(it)].w
            incr(it)
            if u == i:
                print (u/2,v/2,w),' ;; ',

        print ''

    print 'result = ', [Gi.endpoints_(e) for e in result]
    '''


