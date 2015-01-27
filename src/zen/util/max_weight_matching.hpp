
using namespace std;
#include<cstddef>
#include<vector>
#include<fiboheap.hpp>
#include<assert.h>
#include<iostream>

struct edge_t {
    int src, tgt;
    double w;
    edge_t(int s, int t, double w): src(s), tgt(t), w(w){}
    edge_t() {}
};

void _augment_path_to(vector<edge_t> &E, int pred[], int v)     {
    int eidx = pred[v];
    while(eidx != -1)      {
        edge_t &edge = E[eidx];
        eidx = pred[edge.src];
        int t = edge.src;
        edge.src = edge.tgt;
        edge.tgt = t;
    }
}


void _relax(vector< vector<int> > &G, vector<edge_t> &E, int pred[], double dist[], 
                fibnode_t *nodes, vector<int>  &RB, double pot[], FiboHeap &pq, int a)      
{
    vector<int> &edges = G[a];
    for(vector<int>::iterator iv = edges.begin(); iv != edges.end(); iv++)  {
        int eidx = *iv;
        edge_t &edge = E[eidx];
        if(edge.src != a) 
            continue;

        int b = edge.tgt;
        double db = dist[a] + (pot[a] + pot[b] - edge.w);
        if(pred[b] == -1)       {
            dist[b] = db;  pred[b] = eidx ;
            RB.push_back(b);
	    nodes[b].init();
            nodes[b].val = b; nodes[b].key = db; nodes[b].inheap=true;
            pq.insert(&nodes[b]);
        } else if(db < dist[b])         {
            dist[b] = db;  pred[b] = eidx;
            pq.decrease_key(&nodes[b],db);
        }
    }

}


void mwb_matching(vector< vector<int> > &G, vector<edge_t> &E, 
		vector<int> &U, vector<int> &V, vector<int> &M) 	{
    FiboHeap pq;
    int N = G.size();
    double *pot = new double[N]; 
    bool *isfree = new bool[N]; 
    int *pred = new int[N]; 
    double *dist = new double[N]; 
    fibnode_t *nodes = new fibnode_t[N];
    vector<int> RA, RB;
    RA.reserve(N); RB.reserve(N);

    for(int i = 0; i < N; i++)    {
        pot[i] = dist[i] = 0.0;
        isfree[i] = true;
        pred[i] = -1;
	nodes[i].init();
    }

    for(vector<int>::iterator it = U.begin(); it != U.end(); it++)         {
        int a = *it; 
        double maxw = -INF;
        edge_t *maxe = NULL;

        vector<int> &edges = G[a];
        for(vector<int>::iterator iv = edges.begin(); iv != edges.end(); iv++)  {
            edge_t &edge = E[*iv];
            if(edge.src != a) 
                continue;
                
            if(edge.w > maxw) {
                maxw = edge.w; 
                maxe = &edge;
            }
            
        }

        pot[a] = maxw;
        if(maxe != NULL)  {
            assert( a != maxe->tgt);
            if(isfree[maxe->tgt])    {
                maxe->src = maxe->tgt; maxe->tgt = a;    
                isfree[maxe->src] = isfree[maxe->tgt] = false;
            }
        }
    }

    for(vector<int>::iterator it = U.begin(); it != U.end(); it++)         {
        int a = *it; 

        if(!isfree[a]) continue;

        int bestA = a;
        double minA = pot[a];
        dist[a] = 0;
        RA.clear(); RB.clear();
        RA.push_back(a);

        _relax(G, E, pred, dist, nodes, RB, pot, pq, a);

        double delta=0.0;
        while(true)     {
            int b = -1;
            double db = -INF;
            if(!pq.is_empty())  {
                fibnode_t *bnode = pq.delete_min();
                b = bnode->val;
                bnode->inheap = false;
                db = dist[b];
            }

            //distinguish 3 cases
            if(b == -1 || db >= minA)   {
                delta = minA;
                // aug to best in A
                _augment_path_to(E, pred, bestA);
                isfree[a] = false;
                isfree[bestA] = true;
                break;
            } else {
                if(isfree[b])   {
                    delta = db;
                    //aug to b
                    _augment_path_to(E, pred, b);
                    isfree[a] = isfree[b] = false;
                    break;
                } else {
                    //conti SP comp
                    int eidx = -1, a1 = -1;
                    vector<int> &edges = G[b];
                    for(vector<int>::iterator iv = edges.begin(); iv != edges.end(); iv++)  {
                        edge_t &edge = E[*iv];
                        if(edge.src != b) 
                            continue;
                        eidx = *iv; a1 = edge.tgt;
                        break;
                    }
                    assert(eidx != -1);
                    //a1,w1 = G[b].iteritems().next()
                    pred[a1] = eidx; dist[a1] = db;
                    RA.push_back(a1);

                    if(db + pot[a1] < minA)     {
                        bestA = a1;
                        minA = db + pot[a1];
                    }

                    _relax(G, E, pred, dist, nodes, RB, pot, pq, a1);
                }
            }
        }

        // augment: pot update and reinitialize
        while(RA.size() > 0)    {
            int ra = RA.back();  RA.pop_back();
            pred[ra] = -1;
            double dpot = delta - dist[ra];
            if(dpot <= 0) continue;
            pot[ra] = pot[ra] - dpot;
        }

        while(RB.size() > 0)    {
            int rb = RB.back(); RB.pop_back();
            pred[rb] = -1; double dpot =  delta - dist[rb];
            if(nodes[rb].inheap)        {
                pq.delete_node(&nodes[rb]);
                nodes[rb].inheap = false;
            }
            if(dpot <= 0) continue;
            pot[rb] = pot[rb] + dpot;
        }

    }


    for(vector<int>::iterator it = V.begin(); it != V.end(); it++)         {
        int b = *it;
        vector<int> &edges = G[b];
        for(vector<int>::iterator iv = edges.begin(); iv != edges.end(); iv++)  {
            int e = *iv;
            edge_t &edge = E[e];
            if(edge.src != b) 
                continue;

            M.push_back(e);

            int t = edge.src;
            edge.src = edge.tgt;
            edge.tgt = t;
        }
    }

    delete[] pot, delete[] isfree, delete[] pred;
    delete[] dist, delete[] nodes;

}



