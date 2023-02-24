// Copyright 2022 Google LLC
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

package cel

import (
	"github.com/google/cel-go/common"
	"github.com/google/cel-go/parser"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// Macro describes a function signature to match and the MacroExpander to apply.
//
// Note: when a Macro should apply to multiple overloads (based on arg count) of a given function,
// a Macro should be created per arg-count or as a var arg macro.
type Macro = parser.Macro

// MacroExpander converts a call and its associated arguments into a new CEL abstract syntax tree.
//
// If the MacroExpander determines within the implementation that an expansion is not needed it may return
// a nil Expr value to indicate a non-match. However, if an expansion is to be performed, but the arguments
// are not well-formed, the result of the expansion will be an error.
//
// The MacroExpander accepts as arguments a MacroExprHelper as well as the arguments used in the function call
// and produces as output an Expr ast node.
//
// Note: when the Macro.IsReceiverStyle() method returns true, the target argument will be nil.
type MacroExpander = parser.MacroExpander

// MacroExprHelper exposes helper methods for creating new expressions within a CEL abstract syntax tree.
type MacroExprHelper = parser.ExprHelper

// NewGlobalMacro creates a Macro for a global function with the specified arg count.
func NewGlobalMacro(function string, argCount int, expander MacroExpander) Macro {
	return parser.NewGlobalMacro(function, argCount, expander)
}

// NewReceiverMacro creates a Macro for a receiver function matching the specified arg count.
func NewReceiverMacro(function string, argCount int, expander MacroExpander) Macro {
	return parser.NewReceiverMacro(function, argCount, expander)
}

// NewGlobalVarArgMacro creates a Macro for a global function with a variable arg count.
func NewGlobalVarArgMacro(function string, expander MacroExpander) Macro {
	return parser.NewGlobalVarArgMacro(function, expander)
}

// NewReceiverVarArgMacro creates a Macro for a receiver function matching a variable arg count.
func NewReceiverVarArgMacro(function string, expander MacroExpander) Macro {
	return parser.NewReceiverVarArgMacro(function, expander)
}

// HasMacroExpander expands the input call arguments into a presence test, e.g. has(<operand>.field)
func HasMacroExpander(meh MacroExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *common.Error) {
	return parser.MakeHas(meh, target, args)
}

// ExistsMacroExpander expands the input call arguments into a comprehension that returns true if any of the
// elements in the range match the predicate expressions:
// <iterRange>.exists(<iterVar>, <predicate>)
func ExistsMacroExpander(meh MacroExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *common.Error) {
	return parser.MakeExists(meh, target, args)
}

// ExistsOneMacroExpander expands the input call arguments into a comprehension that returns true if exactly
// one of the elements in the range match the predicate expressions:
// <iterRange>.exists_one(<iterVar>, <predicate>)
func ExistsOneMacroExpander(meh MacroExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *common.Error) {
	return parser.MakeExistsOne(meh, target, args)
}

// MapMacroExpander expands the input call arguments into a comprehension that transforms each element in the
// input to produce an output list.
//
// There are two call patterns supported by map:
//
//	<iterRange>.map(<iterVar>, <transform>)
//	<iterRange>.map(<iterVar>, <predicate>, <transform>)
//
// In the second form only iterVar values which return true when provided to the predicate expression
// are transformed.
func MapMacroExpander(meh MacroExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *common.Error) {
	return parser.MakeMap(meh, target, args)
}

// FilterMacroExpander expands the input call arguments into a comprehension which produces a list which contains
// only elements which match the provided predicate expression:
// <iterRange>.filter(<iterVar>, <predicate>)
func FilterMacroExpander(meh MacroExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *common.Error) {
	return parser.MakeFilter(meh, target, args)
}

var (
	// Aliases to each macro in the CEL standard environment.
	// Note: reassigning these macro variables may result in undefined behavior.

	// HasMacro expands "has(m.f)" which tests the presence of a field, avoiding the need to
	// specify the field as a string.
	HasMacro = parser.HasMacro

	// AllMacro expands "range.all(var, predicate)" into a comprehension which ensures that all
	// elements in the range satisfy the predicate.
	AllMacro = parser.AllMacro

	// ExistsMacro expands "range.exists(var, predicate)" into a comprehension which ensures that
	// some element in the range satisfies the predicate.
	ExistsMacro = parser.ExistsMacro

	// ExistsOneMacro expands "range.exists_one(var, predicate)", which is true if for exactly one
	// element in range the predicate holds.
	ExistsOneMacro = parser.ExistsOneMacro

	// MapMacro expands "range.map(var, function)" into a comprehension which applies the function
	// to each element in the range to produce a new list.
	MapMacro = parser.MapMacro

	// MapFilterMacro expands "range.map(var, predicate, function)" into a comprehension which
	// first filters the elements in the range by the predicate, then applies the transform function
	// to produce a new list.
	MapFilterMacro = parser.MapFilterMacro

	// FilterMacro expands "range.filter(var, predicate)" into a comprehension which filters
	// elements in the range, producing a new list from the elements that satisfy the predicate.
	FilterMacro = parser.FilterMacro

	// StandardMacros provides an alias to all the CEL macros defined in the standard environment.
	StandardMacros = []Macro{
		HasMacro, AllMacro, ExistsMacro, ExistsOneMacro, MapMacro, MapFilterMacro, FilterMacro,
	}

	// NoMacros provides an alias to an empty list of macros
	NoMacros = []Macro{}
)
