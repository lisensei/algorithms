%{
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "tree.h"

int yylex(void);
void yyerror(char *);
%}
%union{
	struct node* ast;
	int INT;
	double DBL;
	char  ID;
}

%token ADD SUB MUL DIV POW NL IF THEN  FI CMA CLN MOD  AND OR SAND SOR LT GT LE GE ASN LB RB LC RC EQU DO OD PRINT END LP RP
%left ADD SUB
%left MUL DIV 
%token NOT 
%token <INT> NUM TRUE FALSE 
%token <ast> IDF
%type <ast>   STMTS STMT AEXP TERM BEXP BTERM FACT SING NUMLIST
%start LAN
%%
LAN:STMTS END{printf("=====\n");prog($1);}
;

STMTS: STMT
|STMT STMTS {$$=createNode(STMTS,$1,$2);}
|DO BEXP THEN STMTS OD CLN {$$=createNode(NWHILE,$2,$4);}
|IF BEXP THEN STMTS LB RB BEXP THEN STMTS FI CLN {$$=createIFES(IFES,$2,$4,$7,$9);}

STMT:AEXP CLN	{$$=createNode(AEXP,$1,NULL);}
|BEXP CLN
|IDF ASN AEXP CLN	{$$=createNode(NASN,$1,$3);}
|IDF ASN LB NUMLIST RB {$$=createNode(NARR,$1,$4);}
|IDF LB AEXP RB ASN AEXP CLN {$$=createIDX(NIDX,$1,$3,$6);}
;


//Boolean expression grammar
BEXP:BTERM			
|BEXP SAND BTERM	{$$=createNode(NSAND,$1,$3);}
|BEXP SOR BTERM		{$$=createNode(NSOR,$1,$3);}
|AEXP EQU AEXP		{$$=createNode(NEQU,$1,$3);}
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
AEXP:TERM		{$$=createNode(ATERM,$1,NULL);}
|AEXP ADD TERM	{$$=createNode(NADD,$1,$3);}
|AEXP SUB TERM	{$$=createNode(NSUB,$1,$3);}   	
;

TERM:FACT		{$$=createNode(AFACT,$1,NULL);}
|TERM MUL FACT {$$=createNode(NMUL,$1,$3);}
|TERM DIV FACT {$$=createNode(NDIV,$1,$3);}
;

FACT:SING       {$$=createNode(ASING,$1,NULL);}	
|FACT POW SING {$$=createNode(NPOW,$1,$3);}	
;

SING:NUM	{$$=createLeaf(NNUM,$1);}
|SUB SING {$$=createNode(NUSUB,$2,NULL);}
|LP AEXP RP	{$$=createNode(AEXP,$2,NULL);}
|IDF LB AEXP RB {$$=createNode(NELE,$1,$3);}
|IDF
;

NUMLIST:NUM   	  {$$=createLeaf(NNUM,$1);}	
|NUM CMA NUMLIST {node* n=createLeaf(NNUM,$1);n->left=$3;$$=n;}
;


%%
void yyerror(char *str){
	fprintf(stderr,"Error:%s\n",str);
}

int yywrap(){
	return 1;
}

int main(){
symbolTable=init();
for(int i=0;i<SYMSIZE;i++){
	valueTree[i]=malloc(NODESIZE);
}
yyparse();
printf("exited\n");
}




