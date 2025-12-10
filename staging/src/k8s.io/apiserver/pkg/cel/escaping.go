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

package cel

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

// newCharacterFilter returns a boolean array to indicate the allowed characters
func newCharacterFilter(characters string) []bool {
	maxChar := 0
	for _, c := range characters {
		if maxChar < int(c) {
			maxChar = int(c)
		}
	}
	filter := make([]bool, maxChar+1)

	for _, c := range characters {
		filter[int(c)] = true
	}

	return filter
}

type escapeCheck struct {
	canSkipRegex     bool
	invalidCharFound bool
}

// skipRegexCheck checks if escape would be skipped.
// if invalidCharFound is true, it must have invalid character; if invalidCharFound is false, not sure if it has invalid character or not
func skipRegexCheck(ident string) escapeCheck {
	escapeCheck := escapeCheck{canSkipRegex: true, invalidCharFound: false}
	// skip escape if possible
	previous_underscore := false
	for _, c := range ident {
		if c == '/' || c == '-' || c == '.' {
			escapeCheck.canSkipRegex = false
			return escapeCheck
		}
		intc := int(c)
		if intc < 0 || intc >= len(validCharacterFilter) || !validCharacterFilter[intc] {
			escapeCheck.invalidCharFound = true
			return escapeCheck
		}
		if c == '_' && previous_underscore {
			escapeCheck.canSkipRegex = false
			return escapeCheck
		}

		previous_underscore = c == '_'
	}
	return escapeCheck
}

// validCharacterFilter indicates the allowed characters.
var validCharacterFilter = newCharacterFilter("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

// Escape escapes ident and returns a CEL identifier (of the form '[a-zA-Z_][a-zA-Z0-9_]*'), or returns
// false if the ident does not match the supported input format of `[a-zA-Z_.-/][a-zA-Z0-9_.-/]*`.
// Escaping Rules:
//   - '__' escapes to '__underscores__'
//   - '.' escapes to '__dot__'
//   - '-' escapes to '__dash__'
//   - '/' escapes to '__slash__'
//   - Identifiers that exactly match a CEL RESERVED keyword escape to '__{keyword}__'. The keywords are: "true", "false",
//     "null", "in", "as", "break", "const", "continue", "else", "for", "function", "if", "import", "let", loop", "package",
//     "namespace", "return".
func Escape(ident string) (string, bool) {
	if len(ident) == 0 || ('0' <= ident[0] && ident[0] <= '9') {
		return "", false
	}
	if celReservedSymbols.Has(ident) {
		return "__" + ident + "__", true
	}

	escapeCheck := skipRegexCheck(ident)
	if escapeCheck.invalidCharFound {
		return "", false
	}
	if escapeCheck.canSkipRegex {
		return ident, true
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
