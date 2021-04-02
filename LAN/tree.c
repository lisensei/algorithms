#include <stdio.h>
#include <stdlib.h>
#include "tree.h"
#define NODESIZE sizeof(node)

int getIndex(char c){
	if(c>=97){
		return c-97;
	}else{
		return c-65+26;
	}

}

node* createNode(int nt,node* l,node* r){
	node* temp=malloc(NODESIZE);
	temp->nodeType=nt;
	temp->left=l;
	temp->right=r;
	return temp;
}

node* createID(int nt,char c){
	node* temp=malloc(NODESIZE);
	temp->name=c;
	temp->nodeType=nt;
	temp->left=NULL;
	temp->right=NULL;
	return temp;
}
node* createLeaf(int nt,int val){
	node* leaf=malloc(sizeof(node));
	leaf->nodeType=nt;
	leaf->left=NULL;
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
		case NSAND:printf("NSAND\n");break;
		case NSOR:printf("NSOR\n");break;
		case NNOT:printf("NNOT\n");break;
		case NGT:printf("NGT\n");break;
		case NLT:printf("NLT\n");break;
		case NGE:printf("NGE\n");break;
		case NLE:printf("NLE\n");break;
		case AEXP:printf("AEXP\n");break;
		case BEXP:printf("BEXP\n");break;
		case NWHILE:printf("WHILE\n");break;
		case NNUM:printf("Terminal reached,value:%d\n",root->value);break;
		case NBOO:printf("Bool Terminal Reached,value:%d\n",root->value);break;
		case NIDF:printf("NIDF\n");break;
		case NASN:printf("NASN\n");break;
		case BTERM:printf("BTERM\n");break;
		default:
			printf("Unkown type\n");
	}

	preorder(root->left);
	preorder(root->right);
}
void inorder(node *root){
	if(root==NULL)
		return;
	inorder(root->left);
	printf("%d\n",root->nodeType);
	inorder(root->right);
}

void postorder(node *root){
	if(root==NULL)
		return;
	postorder(root->left);
	postorder(root->right);
	printf("%d\n",root->nodeType);
}


int eval(node*root){
	int v=0;
		switch(root->nodeType){
		case NNUM: v=root->value;break;
		case NBOO: v=root->value;break;
		case NADD: v=eval(root->left)+eval(root->right);break;
		case NSUB: v=eval(root->left)-eval(root->right);break;
		case NMUL: printf("OP MUL\n");v=eval(root->left)*eval(root->right);break;
		case NDIV: printf("OP DIV\n");v=eval(root->left)/eval(root->right);break;
		case NNOT:printf("OP NNOT\n");v=!eval(root->left);break;
		case NSAND:printf("NSAND\n");v=eval(root->left)&&eval(root->right);break;
		case NSOR:printf("NSOR\n");v=eval(root->left)||eval(root->right);break;
		case NGT:v=eval(root->left)>eval(root->right);break;
		case NLT:printf("NLT\n");v=eval(root->left)<eval(root->right);break;
		case NGE:printf("NGE\n");v=eval(root->left)>=eval(root->right);break;
		case NLE:printf("NLE\n");v=eval(root->left)<=eval(root->right);break;
		case AEXP:v=eval(root->left);break;
		case BEXP:v=eval(root->left);break;
		case NASN:v=eval(root->right);valueTable[getIndex(root->left->name)]=v;break;
		case NIDF:v=valueTable[getIndex(root->name)];break;
		case NWHILE:while(eval(root->left)){prog(root->right);};break;
		case STMT:v=eval(root->left);break;
	
		default:

			printf("type:%d,Invalid node type\n",root->nodeType);
			break;
	}
	return v;
}

void prog(node* root){
	if(root==NULL){
		return;
	}

	if (root->nodeType!=STMTS){
		printf("%d:%d\n",root->nodeType,eval(root) );
		return;
	 }	
		prog(root->left);
		prog(root->right);

}