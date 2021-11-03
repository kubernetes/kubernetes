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
	"k8s.io/apimachinery/pkg/util/sets"
	"strings"
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

// celLanguageIdentifiers is a list of identifiers that are part of the CEL language.
// This does NOT include builtin macro or function identifiers.
// https://github.com/google/cel-spec/blob/master/doc/langdef.md#values
var celLanguageIdentifiers = sets.NewString(
	"int", "uint", "double", "bool", "string", "bytes", "list", "map", "null_type", "type",
)

var celBuiltinMacroIdentifiers = sets.NewString(
	"has", "all", "exists", "exists_one", "map", "filter",
)

var celBuiltinFunctionIdentifiers = sets.NewString(
	"size", "contains", "dyn", "startsWith", "endsWith", "matches",
	// time related
	"duration", "timestamp",
	"getDate", "getDayOfMonth", "getDayOfWeek", "getDayOfYear", "getFullYear", "getHours", "getMilliseconds", "getMinutes", "getMonth", "getSeconds",
)

// AlwaysReservedIdentifiers is a list of identifiers that may not appear unescaped anywhere in a CEL program.
var AlwaysReservedIdentifiers = celReservedSymbols

// RootReservedIdentifiers is a list of identifiers that may not appear as bound variables in a CEL program because
// because they would be ambiguous with the names of CEL builtin type names, macros or functions.
var RootReservedIdentifiers = celLanguageIdentifiers.Union(celBuiltinMacroIdentifiers).Union(celBuiltinFunctionIdentifiers)

// IsRootReserved returns true if an identifier is in the RootReservedIdentifiers set.
func IsRootReserved(prop string) bool {
	return RootReservedIdentifiers.Has(prop)
}

// Escape escapes identifiers in the AlwaysReservedIdentifiers set by prefixing ident with "_" and by prefixing
// any ident already prefixed with N '_' with N+1 '_'.
// For an identifier that does not require escaping, the identifier is returned as-is.
func Escape(ident string) string {
	if strings.HasPrefix(ident, "_") || AlwaysReservedIdentifiers.Has(ident) {
		return "_" + ident
	}
	return ident
}

// EscapeSlice returns identifiers with Escape applied to each.
func EscapeSlice(idents []string) []string {
	result := make([]string, len(idents))
	for i, prop := range idents {
		result[i] = Escape(prop)
	}
	return result
}

// Unescape unescapes an identifier escaped by Escape.
func Unescape(escaped string) string {
	if strings.HasPrefix(escaped, "_") {
		trimmed := strings.TrimPrefix(escaped, "_")
		if strings.HasPrefix(trimmed, "_") || AlwaysReservedIdentifiers.Has(trimmed) {
			return trimmed
		}
		panic(fmt.Sprintf("failed to unescape improperly escaped string: %v", escaped))
	}
	return escaped
}
