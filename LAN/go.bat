bison --yacc -dv gc.y
flex gc.l
gcc y.tab.c lex.yy.c tree.c 