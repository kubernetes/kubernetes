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
	"regexp"

	"k8s.io/apimachinery/pkg/util/sets"
)

// celReservedSymbols is a list of RESERVED symbols defined in the CEL lexer.
// No identifiers are allowed to collide with these symbols.
// https://github.com/google/cel-spec/blob/master/doc/langdef.md#syntax
var celReservedSymbols = sets.NewString(
	"true", "false", "null", "in",
	"as", "break", "const", "continue", "else",
	"for", "function", "if", "import", "let",
	"loop", "package", "namespace", "return", // !! 'namespace' is used heavily in Kubernetes
	"var", "void", "while",
)

// expandMatcher matches the escape sequence, characters that are escaped, and characters that are unsupported
var expandMatcher = regexp.MustCompile(`(__|[-./]|[^a-zA-Z0-9-./_])`)

// Escape escapes ident and returns a CEL identifier (of the form '[a-zA-Z_][a-zA-Z0-9_]*'), or returns
// false if the ident does not match the supported input format of `[a-zA-Z_.-/][a-zA-Z0-9_.-/]*`.
// Escaping Rules:
// - '__' escapes to '__underscores__'
// - '.' escapes to '__dot__'
// - '-' escapes to '__dash__'
// - '/' escapes to '__slash__'
// - Identifiers that exactly match a CEL RESERVED keyword escape to '__{keyword}__'. The keywords are: "true", "false",
//	  "null", "in", "as", "break", "const", "continue", "else", "for", "function", "if", "import", "let", loop", "package",
//	  "namespace", "return".
func Escape(ident string) (string, bool) {
	if len(ident) == 0 || ('0' <= ident[0] && ident[0] <= '9') {
		return "", false
	}
	if celReservedSymbols.Has(ident) {
		return "__" + ident + "__", true
	}
	ok := true
	ident = expandMatcher.ReplaceAllStringFunc(ident, func(s string) string {
		switch s {
		case "__":
			return "__underscores__"
		case ".":
			return "__dot__"
		case "-":
			return "__dash__"
		case "/":
			return "__slash__"
		default: // matched a unsupported supported
			ok = false
			return ""
		}
	})
	if !ok {
		return "", false
	}
	return ident, true
}

var unexpandMatcher = regexp.MustCompile(`(_{2}[^_]+_{2})`)

// Unescape unescapes an CEL identifier containing the escape sequences described in Escape, or return false if the
// string contains invalid escape sequences. The escaped input is expected to be a valid CEL identifier, but is
// not checked.
func Unescape(escaped string) (string, bool) {
	ok := true
	escaped = unexpandMatcher.ReplaceAllStringFunc(escaped, func(s string) string {
		contents := s[2 : len(s)-2]
		switch contents {
		case "underscores":
			return "__"
		case "dot":
			return "."
		case "dash":
			return "-"
		case "slash":
			return "/"
		}
		if celReservedSymbols.Has(contents) {
			if len(s) != len(escaped) {
				ok = false
			}
			return contents
		}
		ok = false
		return ""
	})
	if !ok {
		return "", false
	}
	return escaped, true
}
