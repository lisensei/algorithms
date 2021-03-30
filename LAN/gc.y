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
%token LPR RPR
%token NOT 
%token <ID> IDF 
%token <DBL> NUM TRUE FALSE 
%type <ast>	 FACT   
%type <ast>   STMTS STMT  AEXP TERM  
%start LAN
%%
LAN:STMTS END{preorder($1);}
;

STMTS: STMT {$$=createNode(STMTS,$1,NULL,NULL);}
|STMTS STMT {$$=createNode(STMTS,$1,$2,NULL);}

STMT:AEXP CLN	{$$=createNode(STMT,$1,NULL,NULL);}

AEXP:TERM			
|AEXP ADD TERM	{$$=createNode(NADD,$1,$3,NULL);}
|AEXP SUB TERM	{$$=createNode(NSUB,$1,$3,NULL);}
;

TERM:FACT    
|TERM MUL FACT 
|TERM DIV FACT 
;

FACT:NUM	{$$=createLeaf(NNUM,$1);}
|LPR AEXP RPR	{$$=createNode(AEX,$2,NULL,NULL);}	
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




