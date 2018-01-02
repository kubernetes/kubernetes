package toml

import (
	"testing"
)

func testQLFlow(t *testing.T, input string, expectedFlow []token) {
	ch := lexQuery(input)
	for idx, expected := range expectedFlow {
		token := <-ch
		if token != expected {
			t.Log("While testing #", idx, ":", input)
			t.Log("compared (got)", token, "to (expected)", expected)
			t.Log("\tvalue:", token.val, "<->", expected.val)
			t.Log("\tvalue as bytes:", []byte(token.val), "<->", []byte(expected.val))
			t.Log("\ttype:", token.typ.String(), "<->", expected.typ.String())
			t.Log("\tline:", token.Line, "<->", expected.Line)
			t.Log("\tcolumn:", token.Col, "<->", expected.Col)
			t.Log("compared", token, "to", expected)
			t.FailNow()
		}
	}

	tok, ok := <-ch
	if ok {
		t.Log("channel is not closed!")
		t.Log(len(ch)+1, "tokens remaining:")

		t.Log("token ->", tok)
		for token := range ch {
			t.Log("token ->", token)
		}
		t.FailNow()
	}
}

func TestLexSpecialChars(t *testing.T) {
	testQLFlow(t, " .$[]..()?*", []token{
		token{Position{1, 2}, tokenDot, "."},
		token{Position{1, 3}, tokenDollar, "$"},
		token{Position{1, 4}, tokenLeftBracket, "["},
		token{Position{1, 5}, tokenRightBracket, "]"},
		token{Position{1, 6}, tokenDotDot, ".."},
		token{Position{1, 8}, tokenLeftParen, "("},
		token{Position{1, 9}, tokenRightParen, ")"},
		token{Position{1, 10}, tokenQuestion, "?"},
		token{Position{1, 11}, tokenStar, "*"},
		token{Position{1, 12}, tokenEOF, ""},
	})
}

func TestLexString(t *testing.T) {
	testQLFlow(t, "'foo\n'", []token{
		token{Position{1, 2}, tokenString, "foo\n"},
		token{Position{2, 2}, tokenEOF, ""},
	})
}

func TestLexDoubleString(t *testing.T) {
	testQLFlow(t, `"bar"`, []token{
		token{Position{1, 2}, tokenString, "bar"},
		token{Position{1, 6}, tokenEOF, ""},
	})
}

func TestLexStringEscapes(t *testing.T) {
	testQLFlow(t, `"foo \" \' \b \f \/ \t \r \\ \u03A9 \U00012345 \n bar"`, []token{
		token{Position{1, 2}, tokenString, "foo \" ' \b \f / \t \r \\ \u03A9 \U00012345 \n bar"},
		token{Position{1, 55}, tokenEOF, ""},
	})
}

func TestLexStringUnfinishedUnicode4(t *testing.T) {
	testQLFlow(t, `"\u000"`, []token{
		token{Position{1, 2}, tokenError, "unfinished unicode escape"},
	})
}

func TestLexStringUnfinishedUnicode8(t *testing.T) {
	testQLFlow(t, `"\U0000"`, []token{
		token{Position{1, 2}, tokenError, "unfinished unicode escape"},
	})
}

func TestLexStringInvalidEscape(t *testing.T) {
	testQLFlow(t, `"\x"`, []token{
		token{Position{1, 2}, tokenError, "invalid escape sequence: \\x"},
	})
}

func TestLexStringUnfinished(t *testing.T) {
	testQLFlow(t, `"bar`, []token{
		token{Position{1, 2}, tokenError, "unclosed string"},
	})
}

func TestLexKey(t *testing.T) {
	testQLFlow(t, "foo", []token{
		token{Position{1, 1}, tokenKey, "foo"},
		token{Position{1, 4}, tokenEOF, ""},
	})
}

func TestLexRecurse(t *testing.T) {
	testQLFlow(t, "$..*", []token{
		token{Position{1, 1}, tokenDollar, "$"},
		token{Position{1, 2}, tokenDotDot, ".."},
		token{Position{1, 4}, tokenStar, "*"},
		token{Position{1, 5}, tokenEOF, ""},
	})
}

func TestLexBracketKey(t *testing.T) {
	testQLFlow(t, "$[foo]", []token{
		token{Position{1, 1}, tokenDollar, "$"},
		token{Position{1, 2}, tokenLeftBracket, "["},
		token{Position{1, 3}, tokenKey, "foo"},
		token{Position{1, 6}, tokenRightBracket, "]"},
		token{Position{1, 7}, tokenEOF, ""},
	})
}

func TestLexSpace(t *testing.T) {
	testQLFlow(t, "foo bar baz", []token{
		token{Position{1, 1}, tokenKey, "foo"},
		token{Position{1, 5}, tokenKey, "bar"},
		token{Position{1, 9}, tokenKey, "baz"},
		token{Position{1, 12}, tokenEOF, ""},
	})
}

func TestLexInteger(t *testing.T) {
	testQLFlow(t, "100 +200 -300", []token{
		token{Position{1, 1}, tokenInteger, "100"},
		token{Position{1, 5}, tokenInteger, "+200"},
		token{Position{1, 10}, tokenInteger, "-300"},
		token{Position{1, 14}, tokenEOF, ""},
	})
}

func TestLexFloat(t *testing.T) {
	testQLFlow(t, "100.0 +200.0 -300.0", []token{
		token{Position{1, 1}, tokenFloat, "100.0"},
		token{Position{1, 7}, tokenFloat, "+200.0"},
		token{Position{1, 14}, tokenFloat, "-300.0"},
		token{Position{1, 20}, tokenEOF, ""},
	})
}

func TestLexFloatWithMultipleDots(t *testing.T) {
	testQLFlow(t, "4.2.", []token{
		token{Position{1, 1}, tokenError, "cannot have two dots in one float"},
	})
}

func TestLexFloatLeadingDot(t *testing.T) {
	testQLFlow(t, "+.1", []token{
		token{Position{1, 1}, tokenError, "cannot start float with a dot"},
	})
}

func TestLexFloatWithTrailingDot(t *testing.T) {
	testQLFlow(t, "42.", []token{
		token{Position{1, 1}, tokenError, "float cannot end with a dot"},
	})
}

func TestLexNumberWithoutDigit(t *testing.T) {
	testQLFlow(t, "+", []token{
		token{Position{1, 1}, tokenError, "no digit in that number"},
	})
}

func TestLexUnknown(t *testing.T) {
	testQLFlow(t, "^", []token{
		token{Position{1, 1}, tokenError, "unexpected char: '94'"},
	})
}
