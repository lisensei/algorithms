#ifndef _STACK_H_
#define _STACK_H_
struct stack
{
	char* id[100]; 
	int top;
};

typedef struct stack stack;

stack* init();

void push(stack* s,char* data);

char* pop(stack* s);

int stackSize(stack * s);

void printStack(stack* s);
#endif