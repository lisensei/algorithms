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
LAN:STMTS END{preorder($1);prog($1);}
;

STMTS: 
|STMT STMTS {$$=createNode(STMTS,$1,$2,NULL);}

STMT:AEXP CLN	{$$=createNode(STMT,$1,NULL,NULL);}
|BEXP CLN {$$=createNode(STMT,$1,NULL,NULL);}
;


//Boolean expression grammar
BEXP:BTERM
|BEXP SAND BTERM	{$$=createNode(NSAND,$1,$3,NULL);}
|BEXP SOR BTERM		{$$=createNode(NSOR,$1,$3,NULL);}
|AEXP GT AEXP		{$$=createNode(NGT,$1,$3,NULL);}
|AEXP LT AEXP	 	{$$=createNode(NLT,$1,$3,NULL);}
|AEXP GE AEXP		{$$=createNode(NGE,$1,$3,NULL);}
|AEXP LE AEXP	 	{$$=createNode(NLE,$1,$3,NULL);}
;
BTERM:
TRUE 				{$$=createLeaf(NNUM,$1);}
|FALSE 				{$$=createLeaf(NNUM,$1);}
|NOT BTERM			{$$=createNode(NNOT,$2,NULL,NULL);}
|LP BEXP RP			{$$=createNode(BEXP,$2,NULL,NULL);}
;

//Arithmetic expression grammar
AEXP:TERM			
|AEXP ADD TERM	{$$=createNode(NADD,$1,$3,NULL);}
|AEXP SUB TERM	{$$=createNode(NSUB,$1,$3,NULL);}
;
TERM:FACT    
|TERM MUL FACT {$$=createNode(NMUL,$1,$3,NULL);}
|TERM DIV FACT {$$=createNode(NADD,$1,$3,NULL);}
;
FACT:NUM	{$$=createLeaf(NNUM,$1);}
|LP AEXP RP	{$$=createNode(AEXP,$2,NULL,NULL);}	
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




