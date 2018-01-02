package token

import (
	"testing"
)

func TestTypeString(t *testing.T) {
	var tokens = []struct {
		tt  Type
		str string
	}{
		{ILLEGAL, "ILLEGAL"},
		{EOF, "EOF"},
		{NUMBER, "NUMBER"},
		{FLOAT, "FLOAT"},
		{BOOL, "BOOL"},
		{STRING, "STRING"},
		{NULL, "NULL"},
		{LBRACK, "LBRACK"},
		{LBRACE, "LBRACE"},
		{COMMA, "COMMA"},
		{PERIOD, "PERIOD"},
		{RBRACK, "RBRACK"},
		{RBRACE, "RBRACE"},
	}

	for _, token := range tokens {
		if token.tt.String() != token.str {
			t.Errorf("want: %q got:%q\n", token.str, token.tt)

		}
	}

}
