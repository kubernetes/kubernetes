// Copyright (c) 2014 ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ql

import (
	"fmt"
	"testing"
	"unicode"
)

var bad = int(unicode.ReplacementChar)

func tok2name(i int) string {
	if i == unicode.ReplacementChar {
		return "<?>"
	}

	if i < 128 {
		return fmt.Sprintf("tok-'%c'", i)
	}

	return fmt.Sprintf("tok-%d", i)
}

func TestScaner0(t *testing.T) {
	table := []struct {
		src                         string
		tok, line, col, nline, ncol int
		val                         string
	}{
		{"a", identifier, 1, 1, 1, 2, "a"},
		{" a", identifier, 1, 2, 1, 3, "a"},
		{"a ", identifier, 1, 1, 1, 2, "a"},
		{" a ", identifier, 1, 2, 1, 3, "a"},
		{"\na", identifier, 2, 1, 2, 2, "a"},

		{"a\n", identifier, 1, 1, 1, 2, "a"},
		{"\na\n", identifier, 2, 1, 2, 2, "a"},
		{"\n a", identifier, 2, 2, 2, 3, "a"},
		{"a \n", identifier, 1, 1, 1, 2, "a"},
		{"\n a \n", identifier, 2, 2, 2, 3, "a"},

		{"ab", identifier, 1, 1, 1, 3, "ab"},
		{" ab", identifier, 1, 2, 1, 4, "ab"},
		{"ab ", identifier, 1, 1, 1, 3, "ab"},
		{" ab ", identifier, 1, 2, 1, 4, "ab"},
		{"\nab", identifier, 2, 1, 2, 3, "ab"},

		{"ab\n", identifier, 1, 1, 1, 3, "ab"},
		{"\nab\n", identifier, 2, 1, 2, 3, "ab"},
		{"\n ab", identifier, 2, 2, 2, 4, "ab"},
		{"ab \n", identifier, 1, 1, 1, 3, "ab"},
		{"\n ab \n", identifier, 2, 2, 2, 4, "ab"},

		{"c", identifier, 1, 1, 1, 2, "c"},
		{"cR", identifier, 1, 1, 1, 3, "cR"},
		{"cRe", identifier, 1, 1, 1, 4, "cRe"},
		{"cReA", identifier, 1, 1, 1, 5, "cReA"},
		{"cReAt", identifier, 1, 1, 1, 6, "cReAt"},

		{"cReATe", create, 1, 1, 1, 7, "cReATe"},
		{"cReATeD", identifier, 1, 1, 1, 8, "cReATeD"},
		{"2", intLit, 1, 1, 1, 2, "2"},
		{"2.", floatLit, 1, 1, 1, 3, "2."},
		{"2.3", floatLit, 1, 1, 1, 4, "2.3"},
	}

	lval := &yySymType{}
	for i, test := range table {
		l := newLexer(test.src)
		tok := l.Lex(lval)
		nline, ncol := l.npos()
		val := string(l.val)
		if tok != test.tok || l.line != test.line || l.col != test.col ||
			nline != test.nline || ncol != test.ncol ||
			val != test.val {
			t.Fatalf(
				"%d g: %s %d:%d-%d:%d %q, e: %s %d:%d-%d:%d %q",
				i, tok2name(tok), l.line, l.col, nline, ncol, val,
				tok2name(test.tok), test.line, test.col, test.nline, test.ncol, test.val,
			)
		}
	}
}
