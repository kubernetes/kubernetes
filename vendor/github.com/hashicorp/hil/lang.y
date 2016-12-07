// This is the yacc input for creating the parser for interpolation
// expressions in Go. To build it, just run `go generate` on this
// package, as the lexer has the go generate pragma within it.

%{
package hil

import (
    "github.com/hashicorp/hil/ast"
)

%}

%union {
    node     ast.Node
    nodeList []ast.Node
    str      string
    token    *parserToken
}

%token  <str> PROGRAM_BRACKET_LEFT PROGRAM_BRACKET_RIGHT
%token  <str> PROGRAM_STRING_START PROGRAM_STRING_END
%token  <str> PAREN_LEFT PAREN_RIGHT COMMA
%token  <str> SQUARE_BRACKET_LEFT SQUARE_BRACKET_RIGHT

%token <token> ARITH_OP IDENTIFIER INTEGER FLOAT STRING

%type <node> expr interpolation literal literalModeTop literalModeValue
%type <nodeList> args

%left ARITH_OP

%%

top:
    {
        parserResult = &ast.LiteralNode{
            Value: "",
            Typex:  ast.TypeString,
            Posx:  ast.Pos{Column: 1, Line: 1},
        }
    }
|   literalModeTop
	{
        parserResult = $1

        // We want to make sure that the top value is always an Output
        // so that the return value is always a string, list of map from an
        // interpolation.
        //
        // The logic for checking for a LiteralNode is a little annoying
        // because functionally the AST is the same, but we do that because
        // it makes for an easy literal check later (to check if a string
        // has any interpolations).
        if _, ok := $1.(*ast.Output); !ok {
            if n, ok := $1.(*ast.LiteralNode); !ok || n.Typex != ast.TypeString {
                parserResult = &ast.Output{
                    Exprs: []ast.Node{$1},
                    Posx:  $1.Pos(),
                }
            }
        }
	}

literalModeTop:
    literalModeValue
    {
        $$ = $1
    }
|   literalModeTop literalModeValue
    {
        var result []ast.Node
        if c, ok := $1.(*ast.Output); ok {
            result = append(c.Exprs, $2)
        } else {
            result = []ast.Node{$1, $2}
        }

        $$ = &ast.Output{
            Exprs: result,
            Posx:  result[0].Pos(),
        }
    }

literalModeValue:
	literal
	{
        $$ = $1
	}
|   interpolation
    {
        $$ = $1
    }

interpolation:
    PROGRAM_BRACKET_LEFT expr PROGRAM_BRACKET_RIGHT
    {
        $$ = $2
    }

expr:
    PAREN_LEFT expr PAREN_RIGHT
    {
        $$ = $2
    }
|   literalModeTop
    {
        $$ = $1
    }
|   INTEGER
    {
        $$ = &ast.LiteralNode{
            Value: $1.Value.(int),
            Typex:  ast.TypeInt,
            Posx:  $1.Pos,
        }
    }
|   FLOAT
    {
        $$ = &ast.LiteralNode{
            Value: $1.Value.(float64),
            Typex:  ast.TypeFloat,
            Posx:  $1.Pos,
        }
    }
|   ARITH_OP expr
    {
        // This is REALLY jank. We assume that a singular ARITH_OP
        // means 0 ARITH_OP expr, which... is weird. We don't want to
        // support *, /, etc., only -. We should fix this later with a pure
        // Go scanner/parser.
        if $1.Value.(ast.ArithmeticOp) != ast.ArithmeticOpSub {
            panic("Unary - is only allowed")
        }

        $$ = &ast.Arithmetic{
            Op:    $1.Value.(ast.ArithmeticOp),
            Exprs: []ast.Node{
                &ast.LiteralNode{Value: 0, Typex: ast.TypeInt},
                $2,
            },
            Posx:  $2.Pos(),
        }
    }
|   expr ARITH_OP expr
    {
        $$ = &ast.Arithmetic{
            Op:    $2.Value.(ast.ArithmeticOp),
            Exprs: []ast.Node{$1, $3},
            Posx:  $1.Pos(),
        }
    }
|   IDENTIFIER
    {
        $$ = &ast.VariableAccess{Name: $1.Value.(string), Posx: $1.Pos}
    }
|   IDENTIFIER PAREN_LEFT args PAREN_RIGHT
    {
        $$ = &ast.Call{Func: $1.Value.(string), Args: $3, Posx: $1.Pos}
    }
|   IDENTIFIER SQUARE_BRACKET_LEFT expr SQUARE_BRACKET_RIGHT
    {
        $$ = &ast.Index{
                Target: &ast.VariableAccess{
                    Name: $1.Value.(string),
                    Posx: $1.Pos,
                },
                Key: $3,
                Posx: $1.Pos,
            }
    }

args:
	{
		$$ = nil
	}
|	args COMMA expr
	{
		$$ = append($1, $3)
	}
|	expr
	{
		$$ = append($$, $1)
	}

literal:
    STRING
    {
        $$ = &ast.LiteralNode{
            Value: $1.Value.(string),
            Typex:  ast.TypeString,
            Posx:  $1.Pos,
        }
    }

%%
