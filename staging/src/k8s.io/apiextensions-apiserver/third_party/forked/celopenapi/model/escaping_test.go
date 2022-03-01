/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package model

import (
	"fmt"
	"regexp"
	"testing"

	fuzz "github.com/google/gofuzz"
)

// TestEscaping tests that property names are escaped as expected.
func TestEscaping(t *testing.T) {
	cases := []struct {
		unescaped   string
		escaped     string
		unescapable bool
	}{
		// '.', '-', '/' and '__' are escaped since
		// CEL only allows identifiers of the form: [a-zA-Z_][a-zA-Z0-9_]*
		{unescaped: "a.a", escaped: "a__dot__a"},
		{unescaped: "a-a", escaped: "a__dash__a"},
		{unescaped: "a__a", escaped: "a__underscores__a"},
		{unescaped: "a.-/__a", escaped: "a__dot____dash____slash____underscores__a"},
		{unescaped: "a._a", escaped: "a__dot___a"},
		{unescaped: "a__.__a", escaped: "a__underscores____dot____underscores__a"},
		{unescaped: "a___a", escaped: "a__underscores___a"},
		{unescaped: "a____a", escaped: "a__underscores____underscores__a"},
		{unescaped: "a__dot__a", escaped: "a__underscores__dot__underscores__a"},
		{unescaped: "a__underscores__a", escaped: "a__underscores__underscores__underscores__a"},
		// CEL lexer RESERVED keywords must be escaped
		{unescaped: "true", escaped: "__true__"},
		{unescaped: "false", escaped: "__false__"},
		{unescaped: "null", escaped: "__null__"},
		{unescaped: "in", escaped: "__in__"},
		{unescaped: "as", escaped: "__as__"},
		{unescaped: "break", escaped: "__break__"},
		{unescaped: "const", escaped: "__const__"},
		{unescaped: "continue", escaped: "__continue__"},
		{unescaped: "else", escaped: "__else__"},
		{unescaped: "for", escaped: "__for__"},
		{unescaped: "function", escaped: "__function__"},
		{unescaped: "if", escaped: "__if__"},
		{unescaped: "import", escaped: "__import__"},
		{unescaped: "let", escaped: "__let__"},
		{unescaped: "loop", escaped: "__loop__"},
		{unescaped: "package", escaped: "__package__"},
		{unescaped: "namespace", escaped: "__namespace__"},
		{unescaped: "return", escaped: "__return__"},
		{unescaped: "var", escaped: "__var__"},
		{unescaped: "void", escaped: "__void__"},
		{unescaped: "while", escaped: "__while__"},
		// Not all property names are escapable
		{unescaped: "@", unescapable: true},
		{unescaped: "1up", unescapable: true},
		{unescaped: "👑", unescapable: true},
		// CEL macro and function names do not need to be escaped because the parser keeps identifiers in a
		// different namespace than function and  macro names.
		{unescaped: "has", escaped: "has"},
		{unescaped: "all", escaped: "all"},
		{unescaped: "exists", escaped: "exists"},
		{unescaped: "exists_one", escaped: "exists_one"},
		{unescaped: "filter", escaped: "filter"},
		{unescaped: "size", escaped: "size"},
		{unescaped: "contains", escaped: "contains"},
		{unescaped: "startsWith", escaped: "startsWith"},
		{unescaped: "endsWith", escaped: "endsWith"},
		{unescaped: "matches", escaped: "matches"},
		{unescaped: "duration", escaped: "duration"},
		{unescaped: "timestamp", escaped: "timestamp"},
		{unescaped: "getDate", escaped: "getDate"},
		{unescaped: "getDayOfMonth", escaped: "getDayOfMonth"},
		{unescaped: "getDayOfWeek", escaped: "getDayOfWeek"},
		{unescaped: "getFullYear", escaped: "getFullYear"},
		{unescaped: "getHours", escaped: "getHours"},
		{unescaped: "getMilliseconds", escaped: "getMilliseconds"},
		{unescaped: "getMinutes", escaped: "getMinutes"},
		{unescaped: "getMonth", escaped: "getMonth"},
		{unescaped: "getSeconds", escaped: "getSeconds"},
		// we don't escape a single _
		{unescaped: "_if", escaped: "_if"},
		{unescaped: "_has", escaped: "_has"},
		{unescaped: "_int", escaped: "_int"},
		{unescaped: "_anything", escaped: "_anything"},
	}

	for _, tc := range cases {
		t.Run(tc.unescaped, func(t *testing.T) {
			e, escapable := Escape(tc.unescaped)
			if tc.unescapable {
				if escapable {
					t.Errorf("Expected escapable=false, but got %t", escapable)
				}
				return
			}
			if !escapable {
				t.Fatalf("Expected escapable=true, but got %t", escapable)
			}
			if tc.escaped != e {
				t.Errorf("Expected %s to escape to %s, but got %s", tc.unescaped, tc.escaped, e)
			}

			if !validCelIdent.MatchString(e) {
				t.Errorf("Expected %s to escape to a valid CEL identifier, but got %s", tc.unescaped, e)
			}

			u, ok := Unescape(tc.escaped)
			if !ok {
				t.Fatalf("Expected %s to be escapable, but it was not", tc.escaped)
			}
			if tc.unescaped != u {
				t.Errorf("Expected %s to unescape to %s, but got %s", tc.escaped, tc.unescaped, u)
			}
		})
	}
}

func TestUnescapeMalformed(t *testing.T) {
	for _, s := range []string{"__int__extra", "__illegal__"} {
		t.Run(s, func(t *testing.T) {
			e, ok := Unescape(s)
			if ok {
				t.Fatalf("Expected %s to be unescapable, but it escaped to: %s", s, e)
			}
		})
	}
}

func TestEscapingFuzz(t *testing.T) {
	fuzzer := fuzz.New()
	for i := 0; i < 1000; i++ {
		var unescaped string
		fuzzer.Fuzz(&unescaped)
		t.Run(fmt.Sprintf("%d - '%s'", i, unescaped), func(t *testing.T) {
			if len(unescaped) == 0 {
				return
			}
			escaped, ok := Escape(unescaped)
			if !ok {
				return
			}

			if !validCelIdent.MatchString(escaped) {
				t.Errorf("Expected %s to escape to a valid CEL identifier, but got %s", unescaped, escaped)
			}
			u, ok := Unescape(escaped)
			if !ok {
				t.Fatalf("Expected %s to be unescapable, but it was not", escaped)
			}
			if unescaped != u {
				t.Errorf("Expected %s to unescape to %s, but got %s", escaped, unescaped, u)
			}
		})
	}
}

var validCelIdent = regexp.MustCompile(`^[a-zA-Z_][a-zA-Z0-9_]*$`)
