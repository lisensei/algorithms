#include <stdio.h>
#include <stdlib.h>

struct stack
{
	char* id[100]; 
	int top;
};

typedef struct stack stack;

stack* init(){
	stack *temp=malloc(sizeof(stack));
	temp->top=-1;
	return temp;
}

void push(stack* s,char* data){
	s->top++;
	s->id[s->top]=data;
}

char* pop(stack* s){
	char* temp=s->id[s->top];
	s->top--;
	return temp;
}

int stackSize(stack * s){
	return s->top+1; 
}

void printStack(stack* s){
	int size=stackSize(s);
	for(int i=0;i<size;i++){
		printf("%s\n",s->id[i]);
	}
};



