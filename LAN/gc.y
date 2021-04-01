%{
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "tree.c"
int yylex(void);
void yyerror(char *);
node* ast;
%}
%union{
	struct node* ast;
	int ITG;
	double DBL;
	char  ID[30];
}

%token ADD SUB MUL DIV NL IF THEN  FI CMA CLN MOD POW AND OR SAND SOR LT GT LE GE ASN LB RB LC RC EQU DO OD PRINT END
%token LP RP
%token NOT 
%token <ID> IDF 
%token <DBL> NUM TRUE FALSE 
%type <ast>	 FACT   
%type <ast>   STMTS STMT AEXP TERM BEXP BTERM 
%start LAN
%%
LAN:STMTS END{printf("Syntax tree:\n");preorder($1);printf("\nEvaluate:");prog($1);}
;

STMTS: 
|STMT STMTS {$$=createNode(STMTS,$1,$2);}
|DO BEXP THEN STMTS OD CLN {$$=createNode(NWHILE,$2,$4);}


STMT:AEXP CLN	{$$=createNode(AEXP,$1,NULL);}
|BEXP CLN
;


//Boolean expression grammar
BEXP:BTERM			
|BEXP SAND BTERM	{$$=createNode(NSAND,$1,$3);}
|BEXP SOR BTERM		{$$=createNode(NSOR,$1,$3);}
|AEXP GT AEXP		{$$=createNode(NGT,$1,$3);}
|AEXP LT AEXP	 	{$$=createNode(NLT,$1,$3);}
|AEXP GE AEXP		{$$=createNode(NGE,$1,$3);}
|AEXP LE AEXP	 	{$$=createNode(NLE,$1,$3);}
;
BTERM:TRUE 				{$$=createLeaf(NBOO,$1);}
|FALSE 				{$$=createLeaf(NBOO,$1);}
|NOT BTERM			{$$=createNode(NNOT,$2,NULL);}
|LP BEXP RP			{$$=createNode(BEXP,$2,NULL);}
;

//Arithmetic expression grammar
AEXP:TERM		
|AEXP ADD TERM	{$$=createNode(NADD,$1,$3);}
|AEXP SUB TERM	{$$=createNode(NSUB,$1,$3);}
;
TERM:FACT    	
|TERM MUL FACT {$$=createNode(NMUL,$1,$3);}
|TERM DIV FACT {$$=createNode(NDIV,$1,$3);}
;
FACT:NUM	{$$=createLeaf(NNUM,$1);}
|LP AEXP RP	{$$=createNode(AEXP,$2,NULL);}	
;

%%
void yyerror(char *str){
	fprintf(stderr,"Error:%s\n",str);
}

int yywrap(){
	return 1;
}

int main(){

yyparse();

}




