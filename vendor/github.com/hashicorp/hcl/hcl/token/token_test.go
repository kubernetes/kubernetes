package token

import (
	"reflect"
	"testing"
)

func TestTypeString(t *testing.T) {
	var tokens = []struct {
		tt  Type
		str string
	}{
		{ILLEGAL, "ILLEGAL"},
		{EOF, "EOF"},
		{COMMENT, "COMMENT"},
		{IDENT, "IDENT"},
		{NUMBER, "NUMBER"},
		{FLOAT, "FLOAT"},
		{BOOL, "BOOL"},
		{STRING, "STRING"},
		{HEREDOC, "HEREDOC"},
		{LBRACK, "LBRACK"},
		{LBRACE, "LBRACE"},
		{COMMA, "COMMA"},
		{PERIOD, "PERIOD"},
		{RBRACK, "RBRACK"},
		{RBRACE, "RBRACE"},
		{ASSIGN, "ASSIGN"},
		{ADD, "ADD"},
		{SUB, "SUB"},
	}

	for _, token := range tokens {
		if token.tt.String() != token.str {
			t.Errorf("want: %q got:%q\n", token.str, token.tt)
		}
	}

}

func TestTokenValue(t *testing.T) {
	var tokens = []struct {
		tt Token
		v  interface{}
	}{
		{Token{Type: BOOL, Text: `true`}, true},
		{Token{Type: BOOL, Text: `false`}, false},
		{Token{Type: FLOAT, Text: `3.14`}, float64(3.14)},
		{Token{Type: NUMBER, Text: `42`}, int64(42)},
		{Token{Type: IDENT, Text: `foo`}, "foo"},
		{Token{Type: STRING, Text: `"foo"`}, "foo"},
		{Token{Type: STRING, Text: `"foo\nbar"`}, "foo\nbar"},
		{Token{Type: STRING, Text: `"${file("foo")}"`}, `${file("foo")}`},
		{Token{Type: HEREDOC, Text: "<<EOF\nfoo\nbar\nEOF"}, "foo\nbar"},
	}

	for _, token := range tokens {
		if val := token.tt.Value(); !reflect.DeepEqual(val, token.v) {
			t.Errorf("want: %v got:%v\n", token.v, val)
		}
	}

}
