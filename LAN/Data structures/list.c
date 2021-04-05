#include <stdio.h>
#include <stdlib.h>

struct list
{
	int data;
	struct list* next;
}; 

typedef struct list list;

int isEmpty(list *p){
	return (p==NULL);
}
void printList(list *p){
	printf("%d\n",p->data);
	if(p->next==NULL)
		return;
	printList(p->next);
}

int peekFirst(list *p){
	if(p!=NULL)
		return p->data;
}

void addLast(list *p,int data){
	
	list* temp;
	temp=p;
	while(temp->next!=NULL){
		temp=temp->next;
	}
	temp->next=malloc(sizeof(list));
	temp->next->data=data;
	temp->next->next=NULL;
	return;
}

void addFirst(list *p,int data){
	list* temp=malloc(sizeof(list));
	temp->data=data;
	if(p->next==NULL){
	temp->next=NULL;
	p->next=temp;
	return;
	}
	temp->next=p->next;
	p->next=temp;
	return;
}

int peakLast(list *p){
	list *temp =p;
	while(temp->next!=NULL){
		temp=temp->next;
	}
	return temp->data;

}

list* init(int data){
	list *head=malloc(sizeof(list));
	head->data=data;
	head->next=NULL;
	return head;
}


list* arrayToList(int a[],int size){
	list* p=init(a[0]);
	for(int i=1;i<size;i++){
		addFirst(p,a[i]);
	}
	return p;

}

void cat(list* p1,list*p2){
	if(p1==NULL)
		return;
	if(p2==NULL)
		return;
	if(p1->next!=NULL){		
		cat(p1->next,p2);
	}else{
		p1->next=p2;
		return;
	}
}

int size(list* p){
	if(p->next==NULL)
		return 1;
	return 1+size(p->next);
}

int getItem(list*head,int index){
	for(int i=0;i<index;i++){
		head=head->next;
	}
	return head->data;

}

int main(int argc, char const *argv[])
{
	list * head=init(-1);
	addLast(head,1);
	addLast(head,2);
	addLast(head,3);
	addLast(head,4);
	addLast(head,42);
	printf("Data at index 3:%d\n", getItem(head,0));	


	return 0;

}