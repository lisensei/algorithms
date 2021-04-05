#include "tree.h"

node* createNode(int nt,node* l,node* r){
	node* temp=malloc(NODESIZE);
	temp->nodeType=nt;
	temp->left=l;
	temp->right=r;
	temp->sibOne=NULL;
	temp->sibTwo=NULL;
	return temp;
}

node* createID(int nt,char* c){
	node* temp=malloc(NODESIZE);
	temp->name=c;
	temp->nodeType=nt;
	temp->left=NULL;
	temp->right=NULL;
	temp->sibOne=NULL;
	temp->sibTwo=NULL;
	return temp;
}
node* createLeaf(int nt,int val){
	node* leaf=malloc(sizeof(node));
	leaf->nodeType=nt;
	leaf->left=NULL;
	leaf->right=NULL;
	leaf->sibOne=NULL;
	leaf->sibTwo=NULL;
	leaf->value=val;
	return  leaf;
}

node* createIDX(int nt,node* l,node* r,node* s1){
	node* temp=malloc(NODESIZE);
	temp->nodeType=nt;
	temp->left=l;
	temp->right=r;
	temp->sibOne=s1;
	temp->sibTwo=NULL;
}
node* createIFES(int nt,node* l,node*r,node*s1,node*s2){
	node* temp=malloc(NODESIZE);
	temp->nodeType=nt;
	temp->left=l;
	temp->right=r;
	temp->sibOne=s1;
	temp->sibTwo=s2;
	return temp;
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
		case NPOW:printf("POWER\n");break;
		case NUSUB:printf("USUB\n");break;
		case NSAND:printf("NSAND\n");break;
		case NSOR:printf("NSOR\n");break;
		case NNOT:printf("NNOT\n");break;
		case NEQU:printf("NEQU");break;
		case NGT:printf("NGT\n");break;
		case NLT:printf("NLT\n");break;
		case NGE:printf("NGE\n");break;
		case NLE:printf("NLE\n");break;
		case AEXP:printf("AEXP\n");break;
		case BEXP:printf("BEXP\n");break;
		case IFES:printf("IFES\n");break;
		case NWHILE:printf("WHILE\n");break;
		case NNUM:printf("Terminal reached,value:%d\n",root->value);break;
		case NARR:printf("Array\n");break;
		case NIDX:printf("Array Indexing");break;
		case NELE:printf("NELE\n");break;
		case NBOO:printf("Bool Terminal Reached,value:%d\n",root->value);break;
		case NIDF:printf("NIDF:name:%s\n",root->name);break;
		case NASN:printf("NASN\n");break;
		case ATERM:printf("ATERM\n");break;
		case AFACT:printf("AFACT\n");break;
		case ASING:printf("ASING\n");break;
		case BTERM:printf("BTERM\n");break;
		case NNLIST:printf("NNLIST\n");break;
		case NPRT:printf("NPRT\n");break;

		default:
			printf("Unkown type,%d\n",root->nodeType);
	}

	preorder(root->left);
	preorder(root->right);
	preorder(root->sibOne);
	preorder(root->sibTwo);
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
		case NPOW: printf("OP POWER\n");v=pow(eval(root->left),eval(root->right));break;
		case NUSUB:printf("OP USUB\n");v=-eval(root->left);break;
		case NNOT:printf("OP NNOT\n");v=!eval(root->left);break;
		case NSAND:printf("NSAND\n");v=eval(root->left)&&eval(root->right);break;
		case NSOR:printf("NSOR\n");v=eval(root->left)||eval(root->right);break;
		case NGT:v=eval(root->left)>eval(root->right);break;
		case NEQU:v=eval(root->left)==eval(root->right);break;
		case NLT:printf("NLT\n");v=eval(root->left)<eval(root->right);break;
		case NGE:printf("NGE\n");v=eval(root->left)>=eval(root->right);break;
		case NLE:printf("NLE\n");v=eval(root->left)<=eval(root->right);break;
		case AEXP:v=eval(root->left);break;
		case BEXP:v=eval(root->left);break;
		case NASN:v=eval(root->right);node* t=createLeaf(NNUM,v);valueTree[getIndex(symbolTable,root->left->name)]=t;break;
		//root->dataType is NULL;
		case NIDF:v=valueTree[getIndex(symbolTable,root->name)]->value;;break;
		case NPRT:printf("%s:",root->left->name);printArray(valueTree[getIndex(symbolTable,root->left->name)]);break;
		case NARR:valueTree[getIndex(symbolTable,root->left->name)]=root->right;break;
		case NIDX:updateNodeValue(valueTree[getIndex(symbolTable,root->left->name)],eval(root->right),eval(root->sibOne));break;
		case NELE:printf("AELE\n");
		v=getNodeValue(valueTree[getIndex(symbolTable,root->left->name)],eval(root->right));break;
		case NIF:if(eval(root->left)){prog(root->right);}break;
		case IFES:printf("IF ELSE:\n");if(eval(root->left)){prog(root->right);}else{prog(root->sibTwo);}break;
		case NWHILE:while(eval(root->left)){prog(root->right);};break;
		case STMT:v=eval(root->left);break;
		case ATERM:v=eval(root->left);break;
		case AFACT:v=eval(root->left);break;
		case ASING:v=eval(root->left);break;
		case BTERM:v=eval(root->left);break;
		case STMTS:prog(root->left);break;
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
		eval(root);
		return;
	 }	
		prog(root->left);
		prog(root->right);
		prog(root->sibOne);
		prog(root->sibTwo);

}

//stack:
stack* init() {
	stack* temp = malloc(sizeof(stack));
	temp->top = 0;
	return temp;
}

void push(stack* s, char* c) {
	if(exist(s,c))
		return;
	s->symbol[s->top]=c;
	s->top++;

	return;
}

char* pop(stack* s) {
	char* temp = s->symbol[s->top];
	s->top--;
	return temp;
}

int stackSize(stack* s) {
	return s->top;
}

void printStack(stack* s) {
	int size = stackSize(s);
	for (int i = 0; i < size; i++) {
		printf("%d item:%s\n", i,s->symbol[i]);
	}
};

int getIndex(stack* s, char* c) {
	if(s==NULL){
		printf("%s\n","TRIED TO INDEX WITH NULL POINTER" );
		return 0;
	}
	for (int i = 0; i < stackSize(s); i++) {
		if (strcmp(s->symbol[i], c)==0) {
			return i;
		}
	}
	return -1;

}

int exist(stack* s,char * c){
	if(getIndex(s,c)==-1)
		return 0;
	return 1;

}


int getNodeValue(node* root,int index){
		if(root==NULL){
			printf("TRIED TO GET NODE VALUE WITH NULL POINTER");
			return 0;
		}
		node* temp=root;
	for(int i=0;i<index;i++){
		temp=temp->left;
	}
	return temp->value;
}

void updateNodeValue(node* root,int index,int val){
		if(root==NULL){
			printf("NULL POINTER");
			return;
		}
		node* temp=root;
	for(int i=0;i<index;i++){
		temp=temp->left;
	}
	temp->value=val;

}
void printArray(node* root){
		if(root==NULL){
			printf("%s\n","NULL POINTER" );
			return;
		}
		if(root->left==NULL){
			printf("%d\n",root->value );
			return;
		}

		node* temp=root;
		printf("%s", "{");
		while(temp->left!=NULL){
			printf("%d,",temp->value);
			temp=temp->left;
		}
		printf("%d",temp->value);
		printf("%s\n","}" );
}
