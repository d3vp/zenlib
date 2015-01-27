"""
This module provides functionality for building, querying, and manipulating the cacti structure of a directed network.

For information cacti structures, see the following reference:
Commault, Dion, Van der Woude. Characterization of generic properties of linear structured systems for efficient computations. Kybernetika, 38(5):503-520, 2002.
"""
import sys, zen
from zen.control.reachability import generic_rank_
from zen.util.max_weight_bmatching import control_rooted_max_weight_matching
from collections import OrderedDict
from numpy import *
import itertools as it
sys.setrecursionlimit(2**27)

class Stem:
    """
    An instance of this class represents a single stem (and any associated buds) in a cacti structure.
    """

    def __init__(self,node_seq):
        self._node_dict = OrderedDict.fromkeys(node_seq)

    def __contains__(self,x):
        """
        Return True if x is a node in the stem (containment doesn't extend to buds).
        """
        return x in self._node_dict

    def __len__(self):
        """
        Number of nodes in stem
        """
        return len(self._node_dict)

    def __iter__(self):
        """
        Iterate over the nodes in the stem itself.
        """
        return self._node_dict.iterkeys()

    def _clear(self):
        self._node_dict.clear()

    def origin(self):
        """
        The origin or starting node of the stem
        """
        return next(self._node_dict.iterkeys(), None)

    def terminus(self):
        """
        The terminus or end node of the stem
        """
        x = None
        for x in self._node_dict.iterkeys():
            pass
        return x

    def _extend(self, nodes):
        for u in nodes:
            self._node_dict[u] = None

    def length_with_buds(self):
        """
         Returns length of stem including the buds attached to it
        """
        l = len(self._node_dict)
        for bud in self._node_dict.itervalues():
            if bud is not None:
                l += len(bud)
        return l

    def _add_bud(self,dnode,cycle,dnode_target):
        cycle._reorder_origin(dnode_target)
        if dnode == self.terminus():
            self._extend(cycle)
            return False
        else:
            self._node_dict[dnode] = cycle
            cycle._set_stem(self, dnode)
            return True

    def buds(self):
        """
        Returns list of buds (Cycle objects) that are attached to this stem
        """
        return list(self.buds_iter())

    def buds_iter(self):
        """
        Returns iterator over buds (Cycle objects) that are attached to this stem
        """
        return (x for x in self._node_dict.itervalues() if x is not None)

class Cycle:
    """
    Cycles can either be buds that are connected to a stem by a distinguished node or independent cycles (no associated stem).
    """
    def __init__(self,node_seq):
        self._node_dict = OrderedDict.fromkeys(node_seq)
        self._stem = None
        self._dnode = None

    def __contains__(self,x):
        """
        Return True if x is a node in the cycle 
        """
        return x in self._node_dict

    def __len__(self):
        """
        Number of nodes in cycle
        """
        return len(self._node_dict)

    def __iter__(self):
        """
        Iterate over nodes in the cycle
        """
        return self._node_dict.iterkeys()

    def _clear(self):
        self._node_dict.clear()

    def origin(self):
        """
        A starting node of this cycle  (in case of bud this is the node where
        distinguished edge is pointing to)
        """
        return next(self._node_dict.iterkeys(), None)

    def _reorder_origin(self, x):
        if x not in self._node_dict.iterkeys():
            return
        toremove = list(it.takewhile(lambda a: a!=x, self._node_dict.iterkeys()))
        for k in toremove:
            t = self._node_dict[k]
            del self._node_dict[k]
            self._node_dict[k] = t

    def _set_stem(self,stem,dnode):
        self._stem = stem
        self._dnode = dnode

    def stem(self):
        """
        Return stem to which this cycle is attached if it is a bud, otherwise
        return None
        """
        return self._stem

    def dist_node(self):
        """
        Return distinguished node of the stem to which this cycle is attached
        if it is a bud, otherwise returh None
        """
        return self._dnode

    def is_bud(self):
        """
        Returns True if this cycle is a bud
        """
        return self._stem != None

# helper function to construct stems and cycles from a given matching
def _stems_cycs_from_matching(matching, roots=None, origins=[]):
    outmap = {a:b for a,b in matching}
    vis = set()
    indegmap = {}
    for a,b in matching:
        indegmap.setdefault(a,0)
        indegmap[b] = indegmap.get(b, 0) + 1

    if roots is None:
        roots = [u for u in indegmap if indegmap[u] == 0]

    def recur(z,cur,stems,cycs):
        if z in vis:
            cycs.append(Cycle(cur))
            return

        vis.add(z)
        cur.append(z)

        if z not in outmap:
            stems.append(Stem(cur))
            return

        recur(outmap[z],cur,stems,cycs)

    stems, cycs = [],[]

    for r in roots:
        if r not in vis:
            recur(r,[],stems,cycs)

    origins = set(origins)
    roots = [x for x in outmap.iterkeys() if x not in vis]
    roots.sort(key=lambda x: 0 if x in origins else 1)
    for r in roots:
        recur(r,[],stems,cycs)

    return stems, cycs


def build_cacti(G, forced_ctls=None, num_ctls=None, shuffle=False, with_cycles=False):
    """
    This method constructs cacti for a given directed graph G.
    (The nodes in forced_ctls  should be indices of DiGraph nodes and not node objects)
    Inputs: 1) A directed Graph G
            2) if forced_ctls is not None, it has to be a list of tuples
            representing controls. Each tuple represents a control node such
            that the control is to be attached to each node in the tuple. For
            example, [(1,), (4,)] would mean that two controls are attached to
            nodes 1 and 4 respectively. Use this option for when controls are
            fixed. Weigthed maximum matching is performed.
            3) num_ctls : This option is not implemented yet and should be kept
            None
            4) if shuffle is True, matching algorithm is randomized so that
            each call to build cacti would result in a different cacti
            structure. This option is implemented yet for case of unweighted
            maximum matching
            5) if forced_ctls is given, with_cycles option dictates whether any
            independent cycles that are not reachable from the forced_ctls
            controls should be included in the cacti
            6) if none of the force_ctls or num_ctls is given, unweighted
            matching is performed such that minimum number of controls required
            is calculated and the cacti represents the corresponding maximum matching

    Returns Cacti object
    """
    cact = Cacti(G)

    if forced_ctls is not None:
        cact._forced_control_case(forced_ctls, shuffle, with_cycles)
    elif num_ctls is not None:
        cact._free_control_case(num_ctls, shuffle)
    else:
        cact._min_controls_case()

    cact._build_cacti_from_stemscycles()
    return cact


class Cacti:
    """
    The Cacti class represents the cacti  control structure of a given network. It can be used to find the location (and number) of inputs required to control the network as
    well as to access the underlying cacti structure.  It can also be used to find the extent to which a network can be controlled
    by a specified set of controls and the related constrained cacti.
    """

    def __init__(self,G):
        self._G = G

        self._stems = []
        self._cycles = []
        self._matching = []
        self._controls = []
        self._controllable_nodes=set()

    def num_controls(self):
        """
        Return number of controls inputs
        """
        return len(self._controls)

    def stems(self):
        return list(self._stems)

    def cycles(self):
        return list(self._cycles)

    def non_bud_cycles(self):
        """
        Return the non-bud (independent cycles)
        """
        return filter(lambda x: not x.is_bud(), self._cycles)


    # helper function to build cacti from stems and cycles
    def _build_cacti_from_stemscycles(self):
        self._stems.sort(key=lambda x:len(x), reverse= True)
        self._cycles.sort(key=lambda x:len(x), reverse= True)

        for stem in self._stems:
            for u in stem:
                for v in self._G.out_neighbors_(u):
                    cyc = [c for c in self._cycles if not c.is_bud() and v in c]
                    if cyc:
                        if not stem._add_bud(u,cyc[0],v):
                            cyc[0]._clear()

        self._cycles = [c for c in self._cycles if len(c) > 0]

        controls = [ [stem.origin()] for stem in self._stems ]

        if controls:
            idx = 0
            for cyc in self._cycles:
                if not cyc.is_bud():
                    controls[idx].append(cyc.origin())
                    idx = (idx+1) % len(controls)
            controls = [tuple(a) for a in controls]
        else:
            controls = [ tuple(cyc.origin() for cyc in self._cycles) ]

        self._controls = controls

        ctlable_nodes = set()
        ctlable_nodes.update([a for s in self._stems for a in s])
        ctlable_nodes.update([a for c in self._cycles for a in c])
        self._controllable_nodes = ctlable_nodes


    # To find minimum controls required to fully control a network using
    # Unweighted maximum
    # Matching
    def _min_controls_case(self):
        #TODO Shuffle
        G = self._G
        matching = set(zen.matching.maximum_matching_(G))
        matching = [G.endpoints_(eidx) for eidx in matching]
        stems, cycles = _stems_cycs_from_matching(matching)
        for u in set(G.nodes_iter_()).difference(set(x for y in matching for x in y)) :
            stems.append(Stem([u]))
        self._matching, self._stems, self._cycles = matching, stems, cycles

    # To find cacti and hence controllable nodes given a fixed set of controls
    # (forced_ctls) using maximum weighted matching
    def _forced_control_case(self, forced_ctls, doshuffle=False, indcyc=False):
        G = self._G
        origins = [a for b in forced_ctls for a in b]
        matching,roots = control_rooted_max_weight_matching(G, forced_ctls,
                doshuffle=doshuffle, indcyc=indcyc)[1:3]
        matching = [G.endpoints_(eidx) for eidx in matching]
        self._stems, self._cycles = _stems_cycs_from_matching(matching, roots, origins)
        self._matching = matching


    def _free_control_case(self, num_ctls, doshuffle=False):
        raise NotImplementedError("Yet to be implemented.")


    def controls(self):
        """
        Returns list of the controls that are required for maximum control of
        the network. In case of unweigthed matching, it is the minimum number
        of controls required for full control of the network. In case of
        weighted matching (when fixed set of controls are given), this method
        returns the controls that are sufficent for controlling the maximum
        possible nodes of the network.
        """
        return list(self._controls)

    def controllable_nodes(self):
        """
        Returns set of nodes that are controllable
        """
        return set(self._controllable_nodes)

    def num_controllable_nodes(self):
        """
        Return number of nodes that are controllable
        """
        return len(self._controllable_nodes)

    def matching(self):
        """
        Return  matching as calculated by the maximum matching algorithm (unweighted or weighted as the case maybe)
        """
        return list(self._matching)

