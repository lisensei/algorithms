%{
#include <stdio.h>
#include "y.tab.h"
#include "tree.h"
void yyerror(char *);
%}
%%
[1-9][0-9]*				  {sscanf(yytext,"%d",&yylval.INT);return NUM;}
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
"if"	return IF;
"->"	return THEN;
"["		return LB;
"]"		return RB;
"{"	    return LC;
"}"		return RC;
"fi" 	return FI;
"do" 	return DO;
"od"	return OD;
"END"	return END;
"print"	return PRINT;
[a-zA-Z]+ {char c;node* root=createID(NIDF,yytext[0]);yylval.ast=root;return IDF;}
\n      
[ \t]+ 
.
%%

