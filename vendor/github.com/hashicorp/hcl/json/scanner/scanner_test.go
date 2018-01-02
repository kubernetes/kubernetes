package scanner

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/hashicorp/hcl/json/token"
)

var f100 = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

type tokenPair struct {
	tok  token.Type
	text string
}

var tokenLists = map[string][]tokenPair{
	"operator": []tokenPair{
		{token.LBRACK, "["},
		{token.LBRACE, "{"},
		{token.COMMA, ","},
		{token.PERIOD, "."},
		{token.RBRACK, "]"},
		{token.RBRACE, "}"},
	},
	"bool": []tokenPair{
		{token.BOOL, "true"},
		{token.BOOL, "false"},
	},
	"string": []tokenPair{
		{token.STRING, `" "`},
		{token.STRING, `"a"`},
		{token.STRING, `"本"`},
		{token.STRING, `"${file("foo")}"`},
		{token.STRING, `"${file(\"foo\")}"`},
		{token.STRING, `"\a"`},
		{token.STRING, `"\b"`},
		{token.STRING, `"\f"`},
		{token.STRING, `"\n"`},
		{token.STRING, `"\r"`},
		{token.STRING, `"\t"`},
		{token.STRING, `"\v"`},
		{token.STRING, `"\""`},
		{token.STRING, `"\000"`},
		{token.STRING, `"\777"`},
		{token.STRING, `"\x00"`},
		{token.STRING, `"\xff"`},
		{token.STRING, `"\u0000"`},
		{token.STRING, `"\ufA16"`},
		{token.STRING, `"\U00000000"`},
		{token.STRING, `"\U0000ffAB"`},
		{token.STRING, `"` + f100 + `"`},
	},
	"number": []tokenPair{
		{token.NUMBER, "0"},
		{token.NUMBER, "1"},
		{token.NUMBER, "9"},
		{token.NUMBER, "42"},
		{token.NUMBER, "1234567890"},
		{token.NUMBER, "-0"},
		{token.NUMBER, "-1"},
		{token.NUMBER, "-9"},
		{token.NUMBER, "-42"},
		{token.NUMBER, "-1234567890"},
	},
	"float": []tokenPair{
		{token.FLOAT, "0."},
		{token.FLOAT, "1."},
		{token.FLOAT, "42."},
		{token.FLOAT, "01234567890."},
		{token.FLOAT, ".0"},
		{token.FLOAT, ".1"},
		{token.FLOAT, ".42"},
		{token.FLOAT, ".0123456789"},
		{token.FLOAT, "0.0"},
		{token.FLOAT, "1.0"},
		{token.FLOAT, "42.0"},
		{token.FLOAT, "01234567890.0"},
		{token.FLOAT, "0e0"},
		{token.FLOAT, "1e0"},
		{token.FLOAT, "42e0"},
		{token.FLOAT, "01234567890e0"},
		{token.FLOAT, "0E0"},
		{token.FLOAT, "1E0"},
		{token.FLOAT, "42E0"},
		{token.FLOAT, "01234567890E0"},
		{token.FLOAT, "0e+10"},
		{token.FLOAT, "1e-10"},
		{token.FLOAT, "42e+10"},
		{token.FLOAT, "01234567890e-10"},
		{token.FLOAT, "0E+10"},
		{token.FLOAT, "1E-10"},
		{token.FLOAT, "42E+10"},
		{token.FLOAT, "01234567890E-10"},
		{token.FLOAT, "01.8e0"},
		{token.FLOAT, "1.4e0"},
		{token.FLOAT, "42.2e0"},
		{token.FLOAT, "01234567890.12e0"},
		{token.FLOAT, "0.E0"},
		{token.FLOAT, "1.12E0"},
		{token.FLOAT, "42.123E0"},
		{token.FLOAT, "01234567890.213E0"},
		{token.FLOAT, "0.2e+10"},
		{token.FLOAT, "1.2e-10"},
		{token.FLOAT, "42.54e+10"},
		{token.FLOAT, "01234567890.98e-10"},
		{token.FLOAT, "0.1E+10"},
		{token.FLOAT, "1.1E-10"},
		{token.FLOAT, "42.1E+10"},
		{token.FLOAT, "01234567890.1E-10"},
		{token.FLOAT, "-0.0"},
		{token.FLOAT, "-1.0"},
		{token.FLOAT, "-42.0"},
		{token.FLOAT, "-01234567890.0"},
		{token.FLOAT, "-0e0"},
		{token.FLOAT, "-1e0"},
		{token.FLOAT, "-42e0"},
		{token.FLOAT, "-01234567890e0"},
		{token.FLOAT, "-0E0"},
		{token.FLOAT, "-1E0"},
		{token.FLOAT, "-42E0"},
		{token.FLOAT, "-01234567890E0"},
		{token.FLOAT, "-0e+10"},
		{token.FLOAT, "-1e-10"},
		{token.FLOAT, "-42e+10"},
		{token.FLOAT, "-01234567890e-10"},
		{token.FLOAT, "-0E+10"},
		{token.FLOAT, "-1E-10"},
		{token.FLOAT, "-42E+10"},
		{token.FLOAT, "-01234567890E-10"},
		{token.FLOAT, "-01.8e0"},
		{token.FLOAT, "-1.4e0"},
		{token.FLOAT, "-42.2e0"},
		{token.FLOAT, "-01234567890.12e0"},
		{token.FLOAT, "-0.E0"},
		{token.FLOAT, "-1.12E0"},
		{token.FLOAT, "-42.123E0"},
		{token.FLOAT, "-01234567890.213E0"},
		{token.FLOAT, "-0.2e+10"},
		{token.FLOAT, "-1.2e-10"},
		{token.FLOAT, "-42.54e+10"},
		{token.FLOAT, "-01234567890.98e-10"},
		{token.FLOAT, "-0.1E+10"},
		{token.FLOAT, "-1.1E-10"},
		{token.FLOAT, "-42.1E+10"},
		{token.FLOAT, "-01234567890.1E-10"},
	},
}

var orderedTokenLists = []string{
	"comment",
	"operator",
	"bool",
	"string",
	"number",
	"float",
}

func TestPosition(t *testing.T) {
	// create artifical source code
	buf := new(bytes.Buffer)

	for _, listName := range orderedTokenLists {
		for _, ident := range tokenLists[listName] {
			fmt.Fprintf(buf, "\t\t\t\t%s\n", ident.text)
		}
	}

	s := New(buf.Bytes())

	pos := token.Pos{"", 4, 1, 5}
	s.Scan()
	for _, listName := range orderedTokenLists {

		for _, k := range tokenLists[listName] {
			curPos := s.tokPos
			// fmt.Printf("[%q] s = %+v:%+v\n", k.text, curPos.Offset, curPos.Column)

			if curPos.Offset != pos.Offset {
				t.Fatalf("offset = %d, want %d for %q", curPos.Offset, pos.Offset, k.text)
			}
			if curPos.Line != pos.Line {
				t.Fatalf("line = %d, want %d for %q", curPos.Line, pos.Line, k.text)
			}
			if curPos.Column != pos.Column {
				t.Fatalf("column = %d, want %d for %q", curPos.Column, pos.Column, k.text)
			}
			pos.Offset += 4 + len(k.text) + 1     // 4 tabs + token bytes + newline
			pos.Line += countNewlines(k.text) + 1 // each token is on a new line

			s.Error = func(pos token.Pos, msg string) {
				t.Errorf("error %q for %q", msg, k.text)
			}

			s.Scan()
		}
	}
	// make sure there were no token-internal errors reported by scanner
	if s.ErrorCount != 0 {
		t.Errorf("%d errors", s.ErrorCount)
	}
}

func TestComment(t *testing.T) {
	testTokenList(t, tokenLists["comment"])
}

func TestOperator(t *testing.T) {
	testTokenList(t, tokenLists["operator"])
}

func TestBool(t *testing.T) {
	testTokenList(t, tokenLists["bool"])
}

func TestIdent(t *testing.T) {
	testTokenList(t, tokenLists["ident"])
}

func TestString(t *testing.T) {
	testTokenList(t, tokenLists["string"])
}

func TestNumber(t *testing.T) {
	testTokenList(t, tokenLists["number"])
}

func TestFloat(t *testing.T) {
	testTokenList(t, tokenLists["float"])
}

func TestRealExample(t *testing.T) {
	complexReal := `
{
    "variable": {
        "foo": {
            "default": "bar",
            "description": "bar",
            "depends_on": ["something"]
        }
    }
}`

	literals := []struct {
		tokenType token.Type
		literal   string
	}{
		{token.LBRACE, `{`},
		{token.STRING, `"variable"`},
		{token.COLON, `:`},
		{token.LBRACE, `{`},
		{token.STRING, `"foo"`},
		{token.COLON, `:`},
		{token.LBRACE, `{`},
		{token.STRING, `"default"`},
		{token.COLON, `:`},
		{token.STRING, `"bar"`},
		{token.COMMA, `,`},
		{token.STRING, `"description"`},
		{token.COLON, `:`},
		{token.STRING, `"bar"`},
		{token.COMMA, `,`},
		{token.STRING, `"depends_on"`},
		{token.COLON, `:`},
		{token.LBRACK, `[`},
		{token.STRING, `"something"`},
		{token.RBRACK, `]`},
		{token.RBRACE, `}`},
		{token.RBRACE, `}`},
		{token.RBRACE, `}`},
		{token.EOF, ``},
	}

	s := New([]byte(complexReal))
	for _, l := range literals {
		tok := s.Scan()
		if l.tokenType != tok.Type {
			t.Errorf("got: %s want %s for %s\n", tok, l.tokenType, tok.String())
		}

		if l.literal != tok.Text {
			t.Errorf("got: %s want %s\n", tok, l.literal)
		}
	}

}

func TestError(t *testing.T) {
	testError(t, "\x80", "1:1", "illegal UTF-8 encoding", token.ILLEGAL)
	testError(t, "\xff", "1:1", "illegal UTF-8 encoding", token.ILLEGAL)

	testError(t, `"ab`+"\x80", "1:4", "illegal UTF-8 encoding", token.STRING)
	testError(t, `"abc`+"\xff", "1:5", "illegal UTF-8 encoding", token.STRING)

	testError(t, `01238`, "1:7", "numbers cannot start with 0", token.NUMBER)
	testError(t, `01238123`, "1:10", "numbers cannot start with 0", token.NUMBER)
	testError(t, `'aa'`, "1:1", "illegal char: '", token.ILLEGAL)

	testError(t, `"`, "1:2", "literal not terminated", token.STRING)
	testError(t, `"abc`, "1:5", "literal not terminated", token.STRING)
	testError(t, `"abc`+"\n", "1:5", "literal not terminated", token.STRING)
}

func testError(t *testing.T, src, pos, msg string, tok token.Type) {
	s := New([]byte(src))

	errorCalled := false
	s.Error = func(p token.Pos, m string) {
		if !errorCalled {
			if pos != p.String() {
				t.Errorf("pos = %q, want %q for %q", p, pos, src)
			}

			if m != msg {
				t.Errorf("msg = %q, want %q for %q", m, msg, src)
			}
			errorCalled = true
		}
	}

	tk := s.Scan()
	if tk.Type != tok {
		t.Errorf("tok = %s, want %s for %q", tk, tok, src)
	}
	if !errorCalled {
		t.Errorf("error handler not called for %q", src)
	}
	if s.ErrorCount == 0 {
		t.Errorf("count = %d, want > 0 for %q", s.ErrorCount, src)
	}
}

func testTokenList(t *testing.T, tokenList []tokenPair) {
	// create artifical source code
	buf := new(bytes.Buffer)
	for _, ident := range tokenList {
		fmt.Fprintf(buf, "%s\n", ident.text)
	}

	s := New(buf.Bytes())
	for _, ident := range tokenList {
		tok := s.Scan()
		if tok.Type != ident.tok {
			t.Errorf("tok = %q want %q for %q\n", tok, ident.tok, ident.text)
		}

		if tok.Text != ident.text {
			t.Errorf("text = %q want %q", tok.String(), ident.text)
		}

	}
}

func countNewlines(s string) int {
	n := 0
	for _, ch := range s {
		if ch == '\n' {
			n++
		}
	}
	return n
}
