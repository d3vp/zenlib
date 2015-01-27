using namespace std;
#include<cstddef>
#include<vector>

const double INF = 1e20;

struct fibnode_t {
    unsigned int degree;
    int val;
    double key;
    bool is_marked;
    bool inheap;
    fibnode_t *next, *prev, *parent,*child;
    fibnode_t() {init(); }
    void init() {
        degree = val = 0; key = 0.0;
        is_marked = inheap = false;
        next = prev = this;
        parent = child = NULL;
    }
};

class FiboHeap {
    fibnode_t *min_node;
    int hsize;

    public:

    FiboHeap() {
        hsize=0; min_node = NULL;
    }
    ~FiboHeap() {
        // TODO free memory if required
    }

    void insert(fibnode_t *node) {
        min_node = merge_heaps(min_node, node);
        hsize++;
    }

    fibnode_t* get_min()    {
        return min_node;
    }

    bool is_empty() {
        return min_node == NULL;
    }

    int size()  {
        return hsize;
    }

    fibnode_t* delete_min()     {
        if(is_empty())
            return NULL;

        hsize--;
        fibnode_t *min_elem = min_node;

        if(min_node->next == min_node)
            min_node = NULL;
        else    {
            min_node->prev->next = min_node->next;
            min_node->next->prev = min_node->prev;
            min_node = min_node->next;
        }

        fibnode_t *curr;
        if(min_elem->child != NULL)     {
            curr = min_elem->child;

            do  {
                curr->parent = NULL;
                curr = curr->next;
            } while(curr != min_elem->child);
        }

        min_node = merge_heaps(min_node, min_elem->child);

        if(min_node == NULL)
            return min_elem;

        curr = min_node;
        vector<fibnode_t*> treetable, tovisit;

        while(tovisit.empty() || curr != min_node)  {
            tovisit.push_back(curr);
            curr = curr->next;
        }

        for(vector<fibnode_t*>::iterator it = tovisit.begin(); it != tovisit.end(); it++)   {
            curr = *it;

            while (true)   {
                while(curr->degree >= treetable.size())
                    treetable.push_back(NULL);

                if(treetable[curr->degree]  == NULL)   {
                    treetable[curr->degree] = curr;
                    break;
                }

                fibnode_t *other = treetable[curr->degree];
                treetable[curr->degree] = NULL;

                fibnode_t *tmin = other->key < curr->key ? other : curr;
                fibnode_t *tmax = other->key < curr->key ? curr : other;

                tmax->next->prev = tmax->prev;
                tmax->prev->next = tmax->next;

                tmax->next = tmax->prev = tmax;
                tmin->child = merge_heaps(tmin->child, tmax);

                tmax->parent = tmin;
                tmax->is_marked = false;
                tmin->degree += 1;

                curr = tmin;
            }

            if(curr->key <= min_node->key)
                min_node = curr;
        }

        return min_elem;
    }


    void decrease_key(fibnode_t *node, double new_key)   {
        if(new_key > node->key)
            return;

        node->key = new_key;

        if(node->parent != NULL && node->key <= node->parent->key)
            cut_node(node);

        if(node->key <= min_node->key)
            min_node = node;

    }
    void delete_node(fibnode_t *node)   {
        decrease_key(node, -INF);
        delete_min();
    }

    void cut_node(fibnode_t *node)  {
        node->is_marked = false;
        if(node->parent == NULL)
            return;

        if(node->next != node)     {
            node->next->prev = node->prev;
            node->prev->next = node->next;
        }

        if(node->parent->child == node)     {
            if(node->next !=  node)
                node->parent->child = node->next;
            else
                node->parent->child = NULL;
        }

        (node->parent->degree)--;

        node->prev = node->next = node;
        min_node = merge_heaps(min_node, node);

        if(node->parent->is_marked)
            cut_node(node->parent);
        else
            node->parent->is_marked = true;

        node->parent = NULL;

    }


    FiboHeap* meld_heaps(FiboHeap *one,FiboHeap *two)    {
        FiboHeap *result = new FiboHeap();

        result->min_node = merge_heaps(one->min_node, two->min_node);
        result->hsize = one->hsize + two->hsize;

        one->hsize = two->hsize = 0;
        one->min_node = two->min_node = NULL;

        return result;
    }


    fibnode_t* merge_heaps(fibnode_t* one, fibnode_t* two)  {
        if(one == NULL  && two == NULL)
            return NULL;
        if(one == NULL && two != NULL)
            return two;
        if(one != NULL && two == NULL)
            return one;

        fibnode_t *one_next = one->next;
        one->next = two->next;
        one->next->prev = one;
        two->next = one_next;
        two->next->prev = two;

        return one->key < two->key ? one : two;
    }


};


