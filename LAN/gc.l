%{
#include <stdio.h>
#include "y.tab.h"
double getValue(char idf[]);
void yyerror(char *);
%}
%%
[1-9][0-9]*				  {int d;sscanf(yytext,"%d",&d);yylval.DBL=(double)d;return NUM;}
([1-9][0-9]*|0)\.[0-9]+   {sscanf(yytext,"%lf",&yylval.DBL);return NUM;}
"+"     return ADD;
"-"     return SUB;
"*"	    return MUL;
"/"		return DIV;
"%"		return MOD;
"!"		return NOT;	
"^"     return POW;
"&"		return SAND;
"|"		return SOR;
"&&"	return AND;
"||"	return OR;
"="		return EQU;
"<"		return LT;
">"		return GT;
"<="	return LE;
">="	return GE;
","		return CMA;
";"	    return CLN;	
"("		return LPR;
")"		return RPR;
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
"true"	{yylval.DBL=(double)1.0;return TRUE;}
"false" {yylval.DBL=(double)0.0;return FALSE;}
"print"	return PRINT;
[a-zA-Z_]+[a-zA-Z_]* {sscanf(yytext,"%s",yylval.ID); return IDF;}
\n      
[ \t]+ 
.
%%

