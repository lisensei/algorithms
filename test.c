#include <stdio.h>
#include <stdlib.h>

struct node {
	int data;
    struct node* left;
    struct node* mid;
    struct node* right;
};

typedef struct node node;

node* createNode(int i){
	node* root=malloc(sizeof(node));
	root->data=i;
	root->left=NULL;
	root->mid=NULL;
	root->right=NULL;
	return root;
}

void preorder(node* root){
	if(root==NULL)
	     return;
	printf("%d",root->data);
	preorder(root->left);
	preorder(root->mid);
	preorder(root->right);
	
}
void inorder(node* root){
        if(root==NULL)
             return;
        inorder(root->left);
	printf("%d",root->data);
        inorder(root->mid);
        inorder(root->right);

}

void postorder(node* root){
        if(root==NULL)
             return;
        postorder(root->left);  
	postorder(root->mid);
        postorder(root->right);
	printf("%d",root->data);

}





int main(){ return 0;}
