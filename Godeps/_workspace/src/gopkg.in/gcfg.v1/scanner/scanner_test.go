// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scanner

import (
	"os"
	"strings"
	"testing"
)

import (
	"gopkg.in/gcfg.v1/token"
)

var fset = token.NewFileSet()

const /* class */ (
	special = iota
	literal
	operator
)

func tokenclass(tok token.Token) int {
	switch {
	case tok.IsLiteral():
		return literal
	case tok.IsOperator():
		return operator
	}
	return special
}

type elt struct {
	tok   token.Token
	lit   string
	class int
	pre   string
	suf   string
}

var tokens = [...]elt{
	// Special tokens
	{token.COMMENT, "; a comment", special, "", "\n"},
	{token.COMMENT, "# a comment", special, "", "\n"},

	// Operators and delimiters
	{token.ASSIGN, "=", operator, "", "value"},
	{token.LBRACK, "[", operator, "", ""},
	{token.RBRACK, "]", operator, "", ""},
	{token.EOL, "\n", operator, "", ""},

	// Identifiers
	{token.IDENT, "foobar", literal, "", ""},
	{token.IDENT, "a۰۱۸", literal, "", ""},
	{token.IDENT, "foo६४", literal, "", ""},
	{token.IDENT, "bar９８７６", literal, "", ""},
	{token.IDENT, "foo-bar", literal, "", ""},
	{token.IDENT, "foo", literal, ";\n", ""},
	// String literals (subsection names)
	{token.STRING, `"foobar"`, literal, "", ""},
	{token.STRING, `"\""`, literal, "", ""},
	// String literals (values)
	{token.STRING, `"\n"`, literal, "=", ""},
	{token.STRING, `"foobar"`, literal, "=", ""},
	{token.STRING, `"foo\nbar"`, literal, "=", ""},
	{token.STRING, `"foo\"bar"`, literal, "=", ""},
	{token.STRING, `"foo\\bar"`, literal, "=", ""},
	{token.STRING, `"foobar"`, literal, "=", ""},
	{token.STRING, `"foobar"`, literal, "= ", ""},
	{token.STRING, `"foobar"`, literal, "=", "\n"},
	{token.STRING, `"foobar"`, literal, "=", ";"},
	{token.STRING, `"foobar"`, literal, "=", " ;"},
	{token.STRING, `"foobar"`, literal, "=", "#"},
	{token.STRING, `"foobar"`, literal, "=", " #"},
	{token.STRING, "foobar", literal, "=", ""},
	{token.STRING, "foobar", literal, "= ", ""},
	{token.STRING, "foobar", literal, "=", " "},
	{token.STRING, `"foo" "bar"`, literal, "=", " "},
	{token.STRING, "foo\\\nbar", literal, "=", ""},
	{token.STRING, "foo\\\r\nbar", literal, "=", ""},
}

const whitespace = "  \t  \n\n\n" // to separate tokens

var source = func() []byte {
	var src []byte
	for _, t := range tokens {
		src = append(src, t.pre...)
		src = append(src, t.lit...)
		src = append(src, t.suf...)
		src = append(src, whitespace...)
	}
	return src
}()

func newlineCount(s string) int {
	n := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			n++
		}
	}
	return n
}

func checkPos(t *testing.T, lit string, p token.Pos, expected token.Position) {
	pos := fset.Position(p)
	if pos.Filename != expected.Filename {
		t.Errorf("bad filename for %q: got %s, expected %s", lit, pos.Filename, expected.Filename)
	}
	if pos.Offset != expected.Offset {
		t.Errorf("bad position for %q: got %d, expected %d", lit, pos.Offset, expected.Offset)
	}
	if pos.Line != expected.Line {
		t.Errorf("bad line for %q: got %d, expected %d", lit, pos.Line, expected.Line)
	}
	if pos.Column != expected.Column {
		t.Errorf("bad column for %q: got %d, expected %d", lit, pos.Column, expected.Column)
	}
}

// Verify that calling Scan() provides the correct results.
func TestScan(t *testing.T) {
	// make source
	src_linecount := newlineCount(string(source))
	whitespace_linecount := newlineCount(whitespace)

	index := 0

	// error handler
	eh := func(_ token.Position, msg string) {
		t.Errorf("%d: error handler called (msg = %s)", index, msg)
	}

	// verify scan
	var s Scanner
	s.Init(fset.AddFile("", fset.Base(), len(source)), source, eh, ScanComments)
	// epos is the expected position
	epos := token.Position{
		Filename: "",
		Offset:   0,
		Line:     1,
		Column:   1,
	}
	for {
		pos, tok, lit := s.Scan()
		if lit == "" {
			// no literal value for non-literal tokens
			lit = tok.String()
		}
		e := elt{token.EOF, "", special, "", ""}
		if index < len(tokens) {
			e = tokens[index]
		}
		if tok == token.EOF {
			lit = "<EOF>"
			epos.Line = src_linecount
			epos.Column = 2
		}
		if e.pre != "" && strings.ContainsRune("=;#", rune(e.pre[0])) {
			epos.Column = 1
			checkPos(t, lit, pos, epos)
			var etok token.Token
			if e.pre[0] == '=' {
				etok = token.ASSIGN
			} else {
				etok = token.COMMENT
			}
			if tok != etok {
				t.Errorf("bad token for %q: got %q, expected %q", lit, tok, etok)
			}
			pos, tok, lit = s.Scan()
		}
		epos.Offset += len(e.pre)
		if tok != token.EOF {
			epos.Column = 1 + len(e.pre)
		}
		if e.pre != "" && e.pre[len(e.pre)-1] == '\n' {
			epos.Offset--
			epos.Column--
			checkPos(t, lit, pos, epos)
			if tok != token.EOL {
				t.Errorf("bad token for %q: got %q, expected %q", lit, tok, token.EOL)
			}
			epos.Line++
			epos.Offset++
			epos.Column = 1
			pos, tok, lit = s.Scan()
		}
		checkPos(t, lit, pos, epos)
		if tok != e.tok {
			t.Errorf("bad token for %q: got %q, expected %q", lit, tok, e.tok)
		}
		if e.tok.IsLiteral() {
			// no CRs in value string literals
			elit := e.lit
			if strings.ContainsRune(e.pre, '=') {
				elit = string(stripCR([]byte(elit)))
				epos.Offset += len(e.lit) - len(lit) // correct position
			}
			if lit != elit {
				t.Errorf("bad literal for %q: got %q, expected %q", lit, lit, elit)
			}
		}
		if tokenclass(tok) != e.class {
			t.Errorf("bad class for %q: got %d, expected %d", lit, tokenclass(tok), e.class)
		}
		epos.Offset += len(lit) + len(e.suf) + len(whitespace)
		epos.Line += newlineCount(lit) + newlineCount(e.suf) + whitespace_linecount
		index++
		if tok == token.EOF {
			break
		}
		if e.suf == "value" {
			pos, tok, lit = s.Scan()
			if tok != token.STRING {
				t.Errorf("bad token for %q: got %q, expected %q", lit, tok, token.STRING)
			}
		} else if strings.ContainsRune(e.suf, ';') || strings.ContainsRune(e.suf, '#') {
			pos, tok, lit = s.Scan()
			if tok != token.COMMENT {
				t.Errorf("bad token for %q: got %q, expected %q", lit, tok, token.COMMENT)
			}
		}
		// skip EOLs
		for i := 0; i < whitespace_linecount+newlineCount(e.suf); i++ {
			pos, tok, lit = s.Scan()
			if tok != token.EOL {
				t.Errorf("bad token for %q: got %q, expected %q", lit, tok, token.EOL)
			}
		}
	}
	if s.ErrorCount != 0 {
		t.Errorf("found %d errors", s.ErrorCount)
	}
}

func TestScanValStringEOF(t *testing.T) {
	var s Scanner
	src := "= value"
	f := fset.AddFile("src", fset.Base(), len(src))
	s.Init(f, []byte(src), nil, 0)
	s.Scan()              // =
	s.Scan()              // value
	_, tok, _ := s.Scan() // EOF
	if tok != token.EOF {
		t.Errorf("bad token: got %s, expected %s", tok, token.EOF)
	}
	if s.ErrorCount > 0 {
		t.Error("scanning error")
	}
}

// Verify that initializing the same scanner more then once works correctly.
func TestInit(t *testing.T) {
	var s Scanner

	// 1st init
	src1 := "\nname = value"
	f1 := fset.AddFile("src1", fset.Base(), len(src1))
	s.Init(f1, []byte(src1), nil, 0)
	if f1.Size() != len(src1) {
		t.Errorf("bad file size: got %d, expected %d", f1.Size(), len(src1))
	}
	s.Scan()              // \n
	s.Scan()              // name
	_, tok, _ := s.Scan() // =
	if tok != token.ASSIGN {
		t.Errorf("bad token: got %s, expected %s", tok, token.ASSIGN)
	}

	// 2nd init
	src2 := "[section]"
	f2 := fset.AddFile("src2", fset.Base(), len(src2))
	s.Init(f2, []byte(src2), nil, 0)
	if f2.Size() != len(src2) {
		t.Errorf("bad file size: got %d, expected %d", f2.Size(), len(src2))
	}
	_, tok, _ = s.Scan() // [
	if tok != token.LBRACK {
		t.Errorf("bad token: got %s, expected %s", tok, token.LBRACK)
	}

	if s.ErrorCount != 0 {
		t.Errorf("found %d errors", s.ErrorCount)
	}
}

func TestStdErrorHandler(t *testing.T) {
	const src = "@\n" + // illegal character, cause an error
		"@ @\n" // two errors on the same line

	var list ErrorList
	eh := func(pos token.Position, msg string) { list.Add(pos, msg) }

	var s Scanner
	s.Init(fset.AddFile("File1", fset.Base(), len(src)), []byte(src), eh, 0)
	for {
		if _, tok, _ := s.Scan(); tok == token.EOF {
			break
		}
	}

	if len(list) != s.ErrorCount {
		t.Errorf("found %d errors, expected %d", len(list), s.ErrorCount)
	}

	if len(list) != 3 {
		t.Errorf("found %d raw errors, expected 3", len(list))
		PrintError(os.Stderr, list)
	}

	list.Sort()
	if len(list) != 3 {
		t.Errorf("found %d sorted errors, expected 3", len(list))
		PrintError(os.Stderr, list)
	}

	list.RemoveMultiples()
	if len(list) != 2 {
		t.Errorf("found %d one-per-line errors, expected 2", len(list))
		PrintError(os.Stderr, list)
	}
}

type errorCollector struct {
	cnt int            // number of errors encountered
	msg string         // last error message encountered
	pos token.Position // last error position encountered
}

func checkError(t *testing.T, src string, tok token.Token, pos int, err string) {
	var s Scanner
	var h errorCollector
	eh := func(pos token.Position, msg string) {
		h.cnt++
		h.msg = msg
		h.pos = pos
	}
	s.Init(fset.AddFile("", fset.Base(), len(src)), []byte(src), eh, ScanComments)
	if src[0] == '=' {
		_, _, _ = s.Scan()
	}
	_, tok0, _ := s.Scan()
	_, tok1, _ := s.Scan()
	if tok0 != tok {
		t.Errorf("%q: got %s, expected %s", src, tok0, tok)
	}
	if tok1 != token.EOF {
		t.Errorf("%q: got %s, expected EOF", src, tok1)
	}
	cnt := 0
	if err != "" {
		cnt = 1
	}
	if h.cnt != cnt {
		t.Errorf("%q: got cnt %d, expected %d", src, h.cnt, cnt)
	}
	if h.msg != err {
		t.Errorf("%q: got msg %q, expected %q", src, h.msg, err)
	}
	if h.pos.Offset != pos {
		t.Errorf("%q: got offset %d, expected %d", src, h.pos.Offset, pos)
	}
}

var errors = []struct {
	src string
	tok token.Token
	pos int
	err string
}{
	{"\a", token.ILLEGAL, 0, "illegal character U+0007"},
	{"/", token.ILLEGAL, 0, "illegal character U+002F '/'"},
	{"_", token.ILLEGAL, 0, "illegal character U+005F '_'"},
	{`…`, token.ILLEGAL, 0, "illegal character U+2026 '…'"},
	{`""`, token.STRING, 0, ""},
	{`"`, token.STRING, 0, "string not terminated"},
	{"\"\n", token.STRING, 0, "string not terminated"},
	{`="`, token.STRING, 1, "string not terminated"},
	{"=\"\n", token.STRING, 1, "string not terminated"},
	{"=\\", token.STRING, 1, "unquoted '\\' must be followed by new line"},
	{"=\\\r", token.STRING, 1, "unquoted '\\' must be followed by new line"},
	{`"\z"`, token.STRING, 2, "unknown escape sequence"},
	{`"\a"`, token.STRING, 2, "unknown escape sequence"},
	{`"\b"`, token.STRING, 2, "unknown escape sequence"},
	{`"\f"`, token.STRING, 2, "unknown escape sequence"},
	{`"\r"`, token.STRING, 2, "unknown escape sequence"},
	{`"\t"`, token.STRING, 2, "unknown escape sequence"},
	{`"\v"`, token.STRING, 2, "unknown escape sequence"},
	{`"\0"`, token.STRING, 2, "unknown escape sequence"},
}

func TestScanErrors(t *testing.T) {
	for _, e := range errors {
		checkError(t, e.src, e.tok, e.pos, e.err)
	}
}

func BenchmarkScan(b *testing.B) {
	b.StopTimer()
	fset := token.NewFileSet()
	file := fset.AddFile("", fset.Base(), len(source))
	var s Scanner
	b.StartTimer()
	for i := b.N - 1; i >= 0; i-- {
		s.Init(file, source, nil, ScanComments)
		for {
			_, tok, _ := s.Scan()
			if tok == token.EOF {
				break
			}
		}
	}
}
