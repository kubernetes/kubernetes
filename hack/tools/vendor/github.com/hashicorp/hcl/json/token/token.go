package token

import (
	"fmt"
	"strconv"

	hcltoken "github.com/hashicorp/hcl/hcl/token"
)

// Token defines a single HCL token which can be obtained via the Scanner
type Token struct {
	Type Type
	Pos  Pos
	Text string
}

// Type is the set of lexical tokens of the HCL (HashiCorp Configuration Language)
type Type int

const (
	// Special tokens
	ILLEGAL Type = iota
	EOF

	identifier_beg
	literal_beg
	NUMBER // 12345
	FLOAT  // 123.45
	BOOL   // true,false
	STRING // "abc"
	NULL   // null
	literal_end
	identifier_end

	operator_beg
	LBRACK // [
	LBRACE // {
	COMMA  // ,
	PERIOD // .
	COLON  // :

	RBRACK // ]
	RBRACE // }

	operator_end
)

var tokens = [...]string{
	ILLEGAL: "ILLEGAL",

	EOF: "EOF",

	NUMBER: "NUMBER",
	FLOAT:  "FLOAT",
	BOOL:   "BOOL",
	STRING: "STRING",
	NULL:   "NULL",

	LBRACK: "LBRACK",
	LBRACE: "LBRACE",
	COMMA:  "COMMA",
	PERIOD: "PERIOD",
	COLON:  "COLON",

	RBRACK: "RBRACK",
	RBRACE: "RBRACE",
}

// String returns the string corresponding to the token tok.
func (t Type) String() string {
	s := ""
	if 0 <= t && t < Type(len(tokens)) {
		s = tokens[t]
	}
	if s == "" {
		s = "token(" + strconv.Itoa(int(t)) + ")"
	}
	return s
}

// IsIdentifier returns true for tokens corresponding to identifiers and basic
// type literals; it returns false otherwise.
func (t Type) IsIdentifier() bool { return identifier_beg < t && t < identifier_end }

// IsLiteral returns true for tokens corresponding to basic type literals; it
// returns false otherwise.
func (t Type) IsLiteral() bool { return literal_beg < t && t < literal_end }

// IsOperator returns true for tokens corresponding to operators and
// delimiters; it returns false otherwise.
func (t Type) IsOperator() bool { return operator_beg < t && t < operator_end }

// String returns the token's literal text. Note that this is only
// applicable for certain token types, such as token.IDENT,
// token.STRING, etc..
func (t Token) String() string {
	return fmt.Sprintf("%s %s %s", t.Pos.String(), t.Type.String(), t.Text)
}

// HCLToken converts this token to an HCL token.
//
// The token type must be a literal type or this will panic.
func (t Token) HCLToken() hcltoken.Token {
	switch t.Type {
	case BOOL:
		return hcltoken.Token{Type: hcltoken.BOOL, Text: t.Text}
	case FLOAT:
		return hcltoken.Token{Type: hcltoken.FLOAT, Text: t.Text}
	case NULL:
		return hcltoken.Token{Type: hcltoken.STRING, Text: ""}
	case NUMBER:
		return hcltoken.Token{Type: hcltoken.NUMBER, Text: t.Text}
	case STRING:
		return hcltoken.Token{Type: hcltoken.STRING, Text: t.Text, JSON: true}
	default:
		panic(fmt.Sprintf("unimplemented HCLToken for type: %s", t.Type))
	}
}
