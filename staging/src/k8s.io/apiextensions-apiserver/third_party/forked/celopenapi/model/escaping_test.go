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
	"testing"
)

// TestEscaping tests that property names are escaped as expected.
func TestEscaping(t *testing.T) {
	cases := []struct{
		unescaped string
		escaped string
		reservedAtRoot bool
	} {
		// CEL lexer RESERVED keywords must be escaped
		{ unescaped: "true", escaped: "_true" },
		{ unescaped: "false", escaped: "_false" },
		{ unescaped: "null", escaped: "_null" },
		{ unescaped: "in", escaped: "_in" },
		{ unescaped: "as", escaped: "_as" },
		{ unescaped: "break", escaped: "_break" },
		{ unescaped: "const", escaped: "_const" },
		{ unescaped: "continue", escaped: "_continue" },
		{ unescaped: "else", escaped: "_else" },
		{ unescaped: "for", escaped: "_for" },
		{ unescaped: "function", escaped: "_function" },
		{ unescaped: "if", escaped: "_if" },
		{ unescaped: "import", escaped: "_import" },
		{ unescaped: "let", escaped: "_let" },
		{ unescaped: "loop", escaped: "_loop" },
		{ unescaped: "package", escaped: "_package" },
		{ unescaped: "namespace", escaped: "_namespace" },
		{ unescaped: "return", escaped: "_return" },
		{ unescaped: "var", escaped: "_var" },
		{ unescaped: "void", escaped: "_void" },
		{ unescaped: "while", escaped: "_while" },
		// CEL language identifiers do not need to be escaped, but collide with builtin language identifier if bound as
		// root variable names.
		// i.e. "self.int == 1" is legal, but "int == 1" is not.
		{ unescaped: "int", escaped: "int", reservedAtRoot: true },
		{ unescaped: "uint", escaped: "uint", reservedAtRoot: true },
		{ unescaped: "double", escaped: "double", reservedAtRoot: true },
		{ unescaped: "bool", escaped: "bool", reservedAtRoot: true },
		{ unescaped: "string", escaped: "string", reservedAtRoot: true },
		{ unescaped: "bytes", escaped: "bytes", reservedAtRoot: true },
		{ unescaped: "list", escaped: "list", reservedAtRoot: true },
		{ unescaped: "map", escaped: "map", reservedAtRoot: true },
		{ unescaped: "null_type", escaped: "null_type", reservedAtRoot: true },
		{ unescaped: "type", escaped: "type", reservedAtRoot: true },
		// To prevent escaping from colliding with other identifiers, all identifiers prefixed by _s are escaped by
		// prefixing them with N+1 _s.
		{ unescaped: "_if", escaped: "__if" },
		{ unescaped: "__if", escaped: "___if" },
		{ unescaped: "___if", escaped: "____if" },
		{ unescaped: "_has", escaped: "__has" },
		{ unescaped: "_int", escaped: "__int" },
		{ unescaped: "_anything", escaped: "__anything" },
		// CEL macro and function names do not need to be escaped because the parser can disambiguate them from the function and
		// macro identifiers.
		{ unescaped: "has", escaped: "has" },
		{ unescaped: "all", escaped: "all" },
		{ unescaped: "exists", escaped: "exists" },
		{ unescaped: "exists_one", escaped: "exists_one" },
		{ unescaped: "filter", escaped: "filter" },
		{ unescaped: "size", escaped: "size" },
		{ unescaped: "contains", escaped: "contains" },
		{ unescaped: "startsWith", escaped: "startsWith" },
		{ unescaped: "endsWith", escaped: "endsWith" },
		{ unescaped: "matches", escaped: "matches" },
		{ unescaped: "duration", escaped: "duration" },
		{ unescaped: "timestamp", escaped: "timestamp" },
		{ unescaped: "getDate", escaped: "getDate" },
		{ unescaped: "getDayOfMonth", escaped: "getDayOfMonth" },
		{ unescaped: "getDayOfWeek", escaped: "getDayOfWeek" },
		{ unescaped: "getFullYear", escaped: "getFullYear" },
		{ unescaped: "getHours", escaped: "getHours" },
		{ unescaped: "getMilliseconds", escaped: "getMilliseconds" },
		{ unescaped: "getMinutes", escaped: "getMinutes" },
		{ unescaped: "getMonth", escaped: "getMonth" },
		{ unescaped: "getSeconds", escaped: "getSeconds" },
	}

	for _, tc := range cases {
		t.Run(tc.unescaped, func(t *testing.T) {
			e := Escape(tc.unescaped)
			if tc.escaped != e {
				t.Errorf("Expected %s to escape to %s, but got %s", tc.unescaped, tc.escaped, e)
			}
			u := Unescape(tc.escaped)
			if tc.unescaped != u {
				t.Errorf("Expected %s to unescape to %s, but got %s", tc.escaped, tc.unescaped, e)
			}

			isRootReserved := IsRootReserved(tc.unescaped)
			if tc.reservedAtRoot != isRootReserved {
				t.Errorf("Expected isRootReserved=%t for %s, but got %t", tc.reservedAtRoot, tc.unescaped, isRootReserved)
			}
		})
	}
}
