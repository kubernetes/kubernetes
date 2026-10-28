// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package operators defines the internal function names of operators.
//
// All operators in the expression language are modelled as function calls.
package operators

// String "names" for CEL operators.
const (
	// Symbolic operators.
	Conditional   = "_?_:_"
	LogicalAnd    = "_&&_"
	LogicalOr     = "_||_"
	LogicalNot    = "!_"
	Equals        = "_==_"
	NotEquals     = "_!=_"
	Less          = "_<_"
	LessEquals    = "_<=_"
	Greater       = "_>_"
	GreaterEquals = "_>=_"
	Add           = "_+_"
	Subtract      = "_-_"
	Multiply      = "_*_"
	Divide        = "_/_"
	Modulo        = "_%_"
	Negate        = "-_"
	Index         = "_[_]"
	OptIndex      = "_[?_]"
	OptSelect     = "_?._"

	// Macros, must have a valid identifier.
	Has       = "has"
	All       = "all"
	Exists    = "exists"
	ExistsOne = "exists_one"
	Map       = "map"
	Filter    = "filter"

	// Named operators, must not have be valid identifiers.
	NotStrictlyFalse = "@not_strictly_false"
	In               = "@in"

	// Deprecated: named operators with valid identifiers.
	OldNotStrictlyFalse = "__not_strictly_false__"
	OldIn               = "_in_"
)

var (
	operators = map[string]string{
		"+":  Add,
		"/":  Divide,
		"==": Equals,
		">":  Greater,
		">=": GreaterEquals,
		"in": In,
		"<":  Less,
		"<=": LessEquals,
		"%":  Modulo,
		"*":  Multiply,
		"!=": NotEquals,
		"-":  Subtract,
	}
	// operatorMap of the operator symbol which refers to a struct containing the display name,
	// if applicable, the operator precedence, and the arity.
	//
	// If the symbol does not have a display name listed in the map, it is only because it requires
	// special casing to render properly as text.
	operatorMap = map[string]struct {
		displayName string
		precedence  int
		arity       int
	}{
		Conditional:   {displayName: "", precedence: 8, arity: 3},
		LogicalOr:     {displayName: "||", precedence: 7, arity: 2},
		LogicalAnd:    {displayName: "&&", precedence: 6, arity: 2},
		Equals:        {displayName: "==", precedence: 5, arity: 2},
		Greater:       {displayName: ">", precedence: 5, arity: 2},
		GreaterEquals: {displayName: ">=", precedence: 5, arity: 2},
		In:            {displayName: "in", precedence: 5, arity: 2},
		Less:          {displayName: "<", precedence: 5, arity: 2},
		LessEquals:    {displayName: "<=", precedence: 5, arity: 2},
		NotEquals:     {displayName: "!=", precedence: 5, arity: 2},
		OldIn:         {displayName: "in", precedence: 5, arity: 2},
		Add:           {displayName: "+", precedence: 4, arity: 2},
		Subtract:      {displayName: "-", precedence: 4, arity: 2},
		Divide:        {displayName: "/", precedence: 3, arity: 2},
		Modulo:        {displayName: "%", precedence: 3, arity: 2},
		Multiply:      {displayName: "*", precedence: 3, arity: 2},
		LogicalNot:    {displayName: "!", precedence: 2, arity: 1},
		Negate:        {displayName: "-", precedence: 2, arity: 1},
		Index:         {displayName: "", precedence: 1, arity: 2},
		OptIndex:      {displayName: "", precedence: 1, arity: 2},
		OptSelect:     {displayName: "", precedence: 1, arity: 2},
	}
)

// Find the internal function name for an operator, if the input text is one.
func Find(text string) (string, bool) {
	op, found := operators[text]
	return op, found
}

// FindReverse returns the unmangled, text representation of the operator.
func FindReverse(symbol string) (string, bool) {
	op, found := operatorMap[symbol]
	if !found {
		return "", false
	}
	return op.displayName, true
}

// FindReverseBinaryOperator returns the unmangled, text representation of a binary operator.
//
// If the symbol does refer to an operator, but the operator does not have a display name the
// result is false.
func FindReverseBinaryOperator(symbol string) (string, bool) {
	op, found := operatorMap[symbol]
	if !found || op.arity != 2 {
		return "", false
	}
	if op.displayName == "" {
		return "", false
	}
	return op.displayName, true
}

// Precedence returns the operator precedence, where the higher the number indicates
// higher precedence operations.
func Precedence(symbol string) int {
	op, found := operatorMap[symbol]
	if !found {
		return 0
	}
	return op.precedence
}

// Arity returns the number of argument the operator takes
// -1 is returned if an undefined symbol is provided
func Arity(symbol string) int {
	op, found := operatorMap[symbol]
	if !found {
		return -1
	}
	return op.arity
}
