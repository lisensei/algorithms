#ifndef _TREE_H
#define _TREE_H

#define NODESIZE sizeof(node)
enum nodeType{NNUM,NADD,NSUB,NMUL,NDIV,NSAND,NSOR,NNOT,NGT,NLT,NGE,NLE,NASN,NWHILE,NBOO,NIDF,ATERM,BTERM,AFACT,NEQU,AEXP,BEXP,LEAF,STMT,STMTS};

typedef enum nodeType nodeType;
struct node
{
	int  nodeType;
	char name;
	int value;
	struct node* left;
	struct node* right;
};

struct leaf{
	int nodeType;
	int value;
};

typedef struct node node;
typedef struct leaf leaf;
void prog(node*root);
int eval(node*root);
int getIndex(char c);
node* createNode(int nt,node* l,node* r);
node* createID(int nt,char c);
node* createLeaf(int nt,int val);
void preorder(node * root);
void inorder(node * root);
void postorder(node * root);
int eval(node* root);
void prog(node* root);
int valueTable[52];
#endif