#include "stack.h"

stack* init() {
	stack* temp = malloc(sizeof(stack));
	temp->top = 0;
	return temp;
}

void push(stack* s, char* c) {
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
	for (int i = stackSize(s) - 1; i > 0; i--) {
		if (!strcmp(s->symbol[i], c)) {
			return i;
		}
	}
}


