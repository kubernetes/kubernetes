package toml

import "testing"

func TestTokenStringer(t *testing.T) {
	var tests = []struct {
		tt     tokenType
		expect string
	}{
		{tokenError, "Error"},
		{tokenEOF, "EOF"},
		{tokenComment, "Comment"},
		{tokenKey, "Key"},
		{tokenString, "String"},
		{tokenInteger, "Integer"},
		{tokenTrue, "True"},
		{tokenFalse, "False"},
		{tokenFloat, "Float"},
		{tokenEqual, "="},
		{tokenLeftBracket, "["},
		{tokenRightBracket, "]"},
		{tokenLeftCurlyBrace, "{"},
		{tokenRightCurlyBrace, "}"},
		{tokenLeftParen, "("},
		{tokenRightParen, ")"},
		{tokenDoubleLeftBracket, "]]"},
		{tokenDoubleRightBracket, "[["},
		{tokenDate, "Date"},
		{tokenKeyGroup, "KeyGroup"},
		{tokenKeyGroupArray, "KeyGroupArray"},
		{tokenComma, ","},
		{tokenColon, ":"},
		{tokenDollar, "$"},
		{tokenStar, "*"},
		{tokenQuestion, "?"},
		{tokenDot, "."},
		{tokenDotDot, ".."},
		{tokenEOL, "EOL"},
		{tokenEOL + 1, "Unknown"},
	}

	for i, test := range tests {
		got := test.tt.String()
		if got != test.expect {
			t.Errorf("[%d] invalid string of token type; got %q, expected %q", i, got, test.expect)
		}
	}
}

func TestTokenString(t *testing.T) {
	var tests = []struct {
		tok    token
		expect string
	}{
		{token{Position{1, 1}, tokenEOF, ""}, "EOF"},
		{token{Position{1, 1}, tokenError, "Δt"}, "Δt"},
		{token{Position{1, 1}, tokenString, "bar"}, `"bar"`},
		{token{Position{1, 1}, tokenString, "123456789012345"}, `"123456789012345"`},
	}

	for i, test := range tests {
		got := test.tok.String()
		if got != test.expect {
			t.Errorf("[%d] invalid of string token; got %q, expected %q", i, got, test.expect)
		}
	}
}
