#ifndef _TREE_H
#define _TREE_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define NODESIZE sizeof(node)
#define SYMSIZE 100

// Stack
struct stack
{
	char* symbol[SYMSIZE]; 
	int top;
};

typedef struct stack stack;

stack* init();

void push(stack* s,char* c);

char* pop(stack* s);

int stackSize(stack * s);

void printStack(stack* s);
int getIndex(stack* s, char* c);
int exist(stack* s,char * c);


//Tree:
enum nodeType{NNUM,NARR,NADD,NSUB,NMUL,NDIV,NSAND,NSOR,NNOT,NUSUB,
			  NPOW,NGT,NLT,NGE,NLE,NASN,IFES,NWHILE,NBOO,NIDF,
			  ATERM,BTERM,AFACT,ASING,NEQU,AEXP,BEXP,LEAF,STMT,STMTS};

typedef enum nodeType nodeType;
struct node
{
	int  nodeType;
	char *name;
	int value;
	struct node* left;
	struct node* right;
	struct node* sibOne;
	struct node* sibTwo;
};

struct leaf{
	int nodeType;
	int value;
};

typedef struct node node;
typedef struct leaf leaf;
void prog(node*root);
int eval(node*root);
int getIndex(stack* s,char* c);
node* createNode(int nt,node* l,node* r);
node* createID(int nt,char* c);
node* createLeaf(int nt,int val);
node* createIFES(int nt,node* l,node*r,node*s1,node*s2);
void preorder(node * root);
void inorder(node * root);
void postorder(node * root);
int eval(node* root);
void prog(node* root);
extern stack* symbolTable;
extern int valueTable[SYMSIZE];
stack* symbolTable;
int valueTable[SYMSIZE];
#endif