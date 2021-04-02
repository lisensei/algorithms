%{
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "tree.h"
#include "tree.c"
int yylex(void);
void yyerror(char *);
%}
%union{
	struct node* ast;
	int INT;
	double DBL;
	char  ID;
}

%token NL IF THEN  FI CMA CLN MOD POW AND OR SAND SOR LT GT LE GE ASN LB RB LC RC EQU DO OD PRINT END LP RP
%left ADD SUB
%left MUL DIV 
%token NOT 
%token <INT> NUM TRUE FALSE 
%token <ast> IDF
%type <ast>   STMTS STMT AEXP  BEXP BTERM 
%start LAN
%%
LAN:STMTS END{printf("\nEvaluate:");prog($1);}
;

STMTS: STMT
|STMT STMTS {$$=createNode(STMTS,$1,$2);}
|DO BEXP THEN STMTS OD CLN {$$=createNode(NWHILE,$2,$4);}


STMT:AEXP CLN	{$$=createNode(AEXP,$1,NULL);}
|BEXP CLN
|IDF ASN AEXP CLN	{$$=createNode(NASN,$1,$3);}
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
AEXP:		
AEXP ADD AEXP	{$$=createNode(NADD,$1,$3);}
|AEXP SUB AEXP	{$$=createNode(NSUB,$1,$3);}    	
|AEXP MUL AEXP {$$=createNode(NMUL,$1,$3);}
|AEXP DIV AEXP {$$=createNode(NDIV,$1,$3);}
|NUM	{$$=createLeaf(NNUM,$1);}
|LP AEXP RP	{$$=createNode(AEXP,$2,NULL);}	
|IDF		
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
printf("exited\n");
}




