// +build go1.7

package ini

import (
	"bytes"
	"io"
	"reflect"
	"testing"
)

func TestTokenize(t *testing.T) {
	numberToken := newToken(TokenLit, []rune("123"), IntegerType)
	numberToken.base = 10
	cases := []struct {
		r              io.Reader
		expectedTokens []Token
		expectedError  bool
	}{
		{
			r: bytes.NewBuffer([]byte(`x = 123`)),
			expectedTokens: []Token{
				newToken(TokenLit, []rune("x"), StringType),
				newToken(TokenWS, []rune(" "), NoneType),
				newToken(TokenOp, []rune("="), NoneType),
				newToken(TokenWS, []rune(" "), NoneType),
				numberToken,
			},
		},
		{
			r: bytes.NewBuffer([]byte(`[ foo ]`)),
			expectedTokens: []Token{
				newToken(TokenSep, []rune("["), NoneType),
				newToken(TokenWS, []rune(" "), NoneType),
				newToken(TokenLit, []rune("foo"), StringType),
				newToken(TokenWS, []rune(" "), NoneType),
				newToken(TokenSep, []rune("]"), NoneType),
			},
		},
	}

	for _, c := range cases {
		lex := iniLexer{}
		tokens, err := lex.Tokenize(c.r)

		if e, a := c.expectedError, err != nil; e != a {
			t.Errorf("expected %t, but received %t", e, a)
		}

		if e, a := c.expectedTokens, tokens; !reflect.DeepEqual(e, a) {
			t.Errorf("expected %v, but received %v", e, a)
		}
	}
}
