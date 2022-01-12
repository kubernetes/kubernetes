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
// ALl operators in the expression language are modelled as function calls.
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
	reverseOperators = map[string]string{
		Add:           "+",
		Divide:        "/",
		Equals:        "==",
		Greater:       ">",
		GreaterEquals: ">=",
		In:            "in",
		Less:          "<",
		LessEquals:    "<=",
		LogicalAnd:    "&&",
		LogicalNot:    "!",
		LogicalOr:     "||",
		Modulo:        "%",
		Multiply:      "*",
		Negate:        "-",
		NotEquals:     "!=",
		OldIn:         "in",
		Subtract:      "-",
	}
	// precedence of the operator, where the higher value means higher.
	precedence = map[string]int{
		Conditional:   8,
		LogicalOr:     7,
		LogicalAnd:    6,
		Equals:        5,
		Greater:       5,
		GreaterEquals: 5,
		In:            5,
		Less:          5,
		LessEquals:    5,
		NotEquals:     5,
		OldIn:         5,
		Add:           4,
		Subtract:      4,
		Divide:        3,
		Modulo:        3,
		Multiply:      3,
		LogicalNot:    2,
		Negate:        2,
		Index:         1,
	}
)

// Find the internal function name for an operator, if the input text is one.
func Find(text string) (string, bool) {
	op, found := operators[text]
	return op, found
}

// FindReverse returns the unmangled, text representation of the operator.
func FindReverse(op string) (string, bool) {
	txt, found := reverseOperators[op]
	return txt, found
}

// FindReverseBinaryOperator returns the unmangled, text representation of a binary operator.
func FindReverseBinaryOperator(op string) (string, bool) {
	if op == LogicalNot || op == Negate {
		return "", false
	}
	txt, found := reverseOperators[op]
	return txt, found
}

// Precedence returns the operator precedence, where the higher the number indicates
// higher precedence operations.
func Precedence(op string) int {
	p, found := precedence[op]
	if found {
		return p
	}
	return 0
}
