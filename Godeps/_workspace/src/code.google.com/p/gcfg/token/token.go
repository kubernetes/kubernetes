// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package token defines constants representing the lexical tokens of the gcfg
// configuration syntax and basic operations on tokens (printing, predicates).
//
// Note that the API for the token package may change to accommodate new
// features or implementation changes in gcfg.
//
package token

import "strconv"

// Token is the set of lexical tokens of the gcfg configuration syntax.
type Token int

// The list of tokens.
const (
	// Special tokens
	ILLEGAL Token = iota
	EOF
	COMMENT

	literal_beg
	// Identifiers and basic type literals
	// (these tokens stand for classes of literals)
	IDENT  // section-name, variable-name
	STRING // "subsection-name", variable value
	literal_end

	operator_beg
	// Operators and delimiters
	ASSIGN // =
	LBRACK // [
	RBRACK // ]
	EOL    // \n
	operator_end
)

var tokens = [...]string{
	ILLEGAL: "ILLEGAL",

	EOF:     "EOF",
	COMMENT: "COMMENT",

	IDENT:  "IDENT",
	STRING: "STRING",

	ASSIGN: "=",
	LBRACK: "[",
	RBRACK: "]",
	EOL:    "\n",
}

// String returns the string corresponding to the token tok.
// For operators and delimiters, the string is the actual token character
// sequence (e.g., for the token ASSIGN, the string is "="). For all other
// tokens the string corresponds to the token constant name (e.g. for the
// token IDENT, the string is "IDENT").
//
func (tok Token) String() string {
	s := ""
	if 0 <= tok && tok < Token(len(tokens)) {
		s = tokens[tok]
	}
	if s == "" {
		s = "token(" + strconv.Itoa(int(tok)) + ")"
	}
	return s
}

// Predicates

// IsLiteral returns true for tokens corresponding to identifiers
// and basic type literals; it returns false otherwise.
//
func (tok Token) IsLiteral() bool { return literal_beg < tok && tok < literal_end }

// IsOperator returns true for tokens corresponding to operators and
// delimiters; it returns false otherwise.
//
func (tok Token) IsOperator() bool { return operator_beg < tok && tok < operator_end }
