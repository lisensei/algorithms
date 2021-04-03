#ifndef _STACK_H_
#define _STACK_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define SYMSIZE 100

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
#endif