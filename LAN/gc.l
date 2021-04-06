%{
#include <stdio.h>
#include "y.tab.h"
#include "tree.h"
void yyerror(char *);
%}
%%
[1-9][0-9]*|0			  {sscanf(yytext,"%d",&yylval.INT);return NUM;}
"true"	{yylval.INT=1;return TRUE;}
"false" {yylval.INT=0;return FALSE;}
"+"     return ADD;
"-"     return SUB;
"*"	    return MUL;
"/"		return DIV;
"%"		return MOD;
"!"		return NOT;	
"^"     return POW;
"&"		return AND;
"|"		return OR;
"&&"	return SAND;
"||"	return SOR;
"="		return EQU;
"<"		return LT;
">"		return GT;
"<="	return LE;
">="	return GE;
","		return CMA;
";"	    return CLN;	
"("		return LP;
")"		return RP;
":="	return ASN;
"->"	return THEN;
"["		return LB;
"]"		return RB;
"[]"    return ELSE;
"{"	    return LC;
"}"		return RC;
"if"/[\u00A0 \t \r \n]	return IF;
"do"/[\u00A0 \t \r \n] 	return DO;
[\u00A0 \t \r \n]+"fi"	return FI;
[\u00A0 \t \r \n]+"od"	return OD;
"END"	return END;
"print"	return PRINT;
[a-zA-Z_]+[a-zA-Z0-9_]* {push(symbolTable,strdup(yytext));node* root=createID(NIDF,strdup(yytext));yylval.ast=root;return IDF;}
[\u00A0]+
[\t]+
[\r]+
[\n]+
.
%%

