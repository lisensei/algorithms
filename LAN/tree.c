#include <stdio.h>
#include <stdlib.h>
#define NODESIZE sizeof(node)
enum nodeType{NNUM,NADD,NSUB,NMUL,NDIV,NWHILE,NBOO,NEQU,AEX,LEAF,STMT,STMTS};

typedef enum nodeType nodeType;
struct node
{
	int  nodeType;
	double value;
	struct node* left;
	struct node* mid;
	struct node* right;
};

struct leaf{
	int nodeType;
	double value;
};

typedef struct node node;
typedef struct leaf leaf;

node* createNode(int nt,node* l,node*m,node* r){
	node* temp=malloc(NODESIZE);
	temp->nodeType=nt;
	temp->left=l;
	temp->mid=m;
	temp->right=r;
	return temp;
}

node* createLeaf(int nt,double val){
	node* leaf=malloc(sizeof(node));
	leaf->nodeType=LEAF;
	leaf->left=NULL;
	leaf->mid=NULL;
	leaf->right=NULL;
	leaf->value=val;
	return  leaf;
}


void preorder(node * root){
	if(root==NULL)
		return;
	
	switch(root->nodeType){
		case STMTS:printf("Statements\n");break;
		case STMT:printf("Statement\n");break;
		case NADD: printf("ADD\n");break;
		case NSUB: printf("SUB\n");break;
		case NMUL: printf("MUL\n");break;
		case NDIV: printf("DIV\n");break;
		case NWHILE:printf("WHILE\n");break;
		case LEAF:printf("Terminal reached,value:%f\n",root->value);
	}

	preorder(root->left);
	preorder(root->mid);
	preorder(root->right);
}
void inorder(node *root){
	if(root==NULL)
		return;
	inorder(root->left);
	printf("%d\n",root->nodeType);
	inorder(root->mid);
	inorder(root->right);
}

void postorder(node *root){
	if(root==NULL)
		return;
	postorder(root->left);
	postorder(root->mid);
	postorder(root->right);
	printf("%d\n",root->nodeType);
}


double eval(node*root){
	double v=0;
		switch(root->nodeType){
		case LEAF: v=root->value;break;
		case NADD: printf("ADD\n");v=eval(root->left)+eval(root->mid);break;
		case NSUB: printf("SUB\n");v=eval(root->left)-eval(root->mid);break;
		case NMUL: printf("MUL\n");v=eval(root->left)*eval(root->mid);break;
		case NDIV: printf("DIV\n");v=eval(root->left)/eval(root->mid);break;
		case NWHILE:printf("WHILE\n");break;
		case STMT:v=eval(root->left);break;
		default:
			printf("Invalid node type\n");
			break;
	}
	return v;
}

void prog(node* root){
	if(root==NULL){
		printf("Return run value:%f\n",eval(root) );
		return;
	}

	if (root->nodeType!=STMTS){
		printf("%f\n",eval(root) );
		return;
	 }	
		prog(root->left);
		prog(root->mid);

}