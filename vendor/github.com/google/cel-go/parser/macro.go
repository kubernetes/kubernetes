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

package parser

import (
	"fmt"

	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/operators"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// TODO: Consider moving macros to common.

// NewGlobalMacro creates a Macro for a global function with the specified arg count.
func NewGlobalMacro(function string, argCount int, expander MacroExpander) Macro {
	return &macro{
		function: function,
		argCount: argCount,
		expander: expander}
}

// NewReceiverMacro creates a Macro for a receiver function matching the specified arg count.
func NewReceiverMacro(function string, argCount int, expander MacroExpander) Macro {
	return &macro{
		function:      function,
		argCount:      argCount,
		expander:      expander,
		receiverStyle: true}
}

// NewGlobalVarArgMacro creates a Macro for a global function with a variable arg count.
func NewGlobalVarArgMacro(function string, expander MacroExpander) Macro {
	return &macro{
		function:    function,
		expander:    expander,
		varArgStyle: true}
}

// NewReceiverVarArgMacro creates a Macro for a receiver function matching a variable arg
// count.
func NewReceiverVarArgMacro(function string, expander MacroExpander) Macro {
	return &macro{
		function:      function,
		expander:      expander,
		receiverStyle: true,
		varArgStyle:   true}
}

// Macro interface for describing the function signature to match and the MacroExpander to apply.
//
// Note: when a Macro should apply to multiple overloads (based on arg count) of a given function,
// a Macro should be created per arg-count.
type Macro interface {
	// Function name to match.
	Function() string

	// ArgCount for the function call.
	//
	// When the macro is a var-arg style macro, the return value will be zero, but the MacroKey
	// will contain a `*` where the arg count would have been.
	ArgCount() int

	// IsReceiverStyle returns true if the macro matches a receiver style call.
	IsReceiverStyle() bool

	// MacroKey returns the macro signatures accepted by this macro.
	//
	// Format: `<function>:<arg-count>:<is-receiver>`.
	//
	// When the macros is a var-arg style macro, the `arg-count` value is represented as a `*`.
	MacroKey() string

	// Expander returns the MacroExpander to apply when the macro key matches the parsed call
	// signature.
	Expander() MacroExpander
}

// Macro type which declares the function name and arg count expected for the
// macro, as well as a macro expansion function.
type macro struct {
	function      string
	receiverStyle bool
	varArgStyle   bool
	argCount      int
	expander      MacroExpander
}

// Function returns the macro's function name (i.e. the function whose syntax it mimics).
func (m *macro) Function() string {
	return m.function
}

// ArgCount returns the number of arguments the macro expects.
func (m *macro) ArgCount() int {
	return m.argCount
}

// IsReceiverStyle returns whether the macro is receiver style.
func (m *macro) IsReceiverStyle() bool {
	return m.receiverStyle
}

// Expander implements the Macro interface method.
func (m *macro) Expander() MacroExpander {
	return m.expander
}

// MacroKey implements the Macro interface method.
func (m *macro) MacroKey() string {
	if m.varArgStyle {
		return makeVarArgMacroKey(m.function, m.receiverStyle)
	}
	return makeMacroKey(m.function, m.argCount, m.receiverStyle)
}

func makeMacroKey(name string, args int, receiverStyle bool) string {
	return fmt.Sprintf("%s:%d:%v", name, args, receiverStyle)
}

func makeVarArgMacroKey(name string, receiverStyle bool) string {
	return fmt.Sprintf("%s:*:%v", name, receiverStyle)
}

// MacroExpander converts the target and args of a function call that matches a Macro.
//
// Note: when the Macros.IsReceiverStyle() is true, the target argument will be nil.
type MacroExpander func(eh ExprHelper,
	target *exprpb.Expr,
	args []*exprpb.Expr) (*exprpb.Expr, *common.Error)

// ExprHelper assists with the manipulation of proto-based Expr values in a manner which is
// consistent with the source position and expression id generation code leveraged by both
// the parser and type-checker.
type ExprHelper interface {
	// LiteralBool creates an Expr value for a bool literal.
	LiteralBool(value bool) *exprpb.Expr

	// LiteralBytes creates an Expr value for a byte literal.
	LiteralBytes(value []byte) *exprpb.Expr

	// LiteralDouble creates an Expr value for double literal.
	LiteralDouble(value float64) *exprpb.Expr

	// LiteralInt creates an Expr value for an int literal.
	LiteralInt(value int64) *exprpb.Expr

	// LiteralString creates am Expr value for a string literal.
	LiteralString(value string) *exprpb.Expr

	// LiteralUint creates an Expr value for a uint literal.
	LiteralUint(value uint64) *exprpb.Expr

	// NewList creates a CreateList instruction where the list is comprised of the optional set
	// of elements provided as arguments.
	NewList(elems ...*exprpb.Expr) *exprpb.Expr

	// NewMap creates a CreateStruct instruction for a map where the map is comprised of the
	// optional set of key, value entries.
	NewMap(entries ...*exprpb.Expr_CreateStruct_Entry) *exprpb.Expr

	// NewMapEntry creates a Map Entry for the key, value pair.
	NewMapEntry(key *exprpb.Expr, val *exprpb.Expr) *exprpb.Expr_CreateStruct_Entry

	// NewObject creates a CreateStruct instruction for an object with a given type name and
	// optional set of field initializers.
	NewObject(typeName string, fieldInits ...*exprpb.Expr_CreateStruct_Entry) *exprpb.Expr

	// NewObjectFieldInit creates a new Object field initializer from the field name and value.
	NewObjectFieldInit(field string, init *exprpb.Expr) *exprpb.Expr_CreateStruct_Entry

	// Fold creates a fold comprehension instruction.
	//
	// - iterVar is the iteration variable name.
	// - iterRange represents the expression that resolves to a list or map where the elements or
	//   keys (respectively) will be iterated over.
	// - accuVar is the accumulation variable name, typically parser.AccumulatorName.
	// - accuInit is the initial expression whose value will be set for the accuVar prior to
	//   folding.
	// - condition is the expression to test to determine whether to continue folding.
	// - step is the expression to evaluation at the conclusion of a single fold iteration.
	// - result is the computation to evaluate at the conclusion of the fold.
	//
	// The accuVar should not shadow variable names that you would like to reference within the
	// environment in the step and condition expressions. Presently, the name __result__ is commonly
	// used by built-in macros but this may change in the future.
	Fold(iterVar string,
		iterRange *exprpb.Expr,
		accuVar string,
		accuInit *exprpb.Expr,
		condition *exprpb.Expr,
		step *exprpb.Expr,
		result *exprpb.Expr) *exprpb.Expr

	// Ident creates an identifier Expr value.
	Ident(name string) *exprpb.Expr

	// GlobalCall creates a function call Expr value for a global (free) function.
	GlobalCall(function string, args ...*exprpb.Expr) *exprpb.Expr

	// ReceiverCall creates a function call Expr value for a receiver-style function.
	ReceiverCall(function string, target *exprpb.Expr, args ...*exprpb.Expr) *exprpb.Expr

	// PresenceTest creates a Select TestOnly Expr value for modelling has() semantics.
	PresenceTest(operand *exprpb.Expr, field string) *exprpb.Expr

	// Select create a field traversal Expr value.
	Select(operand *exprpb.Expr, field string) *exprpb.Expr

	// OffsetLocation returns the Location of the expression identifier.
	OffsetLocation(exprID int64) common.Location
}

var (
	// AllMacros includes the list of all spec-supported macros.
	AllMacros = []Macro{
		// The macro "has(m.f)" which tests the presence of a field, avoiding the need to specify
		// the field as a string.
		NewGlobalMacro(operators.Has, 1, makeHas),

		// The macro "range.all(var, predicate)", which is true if for all elements in range the
		// predicate holds.
		NewReceiverMacro(operators.All, 2, makeAll),

		// The macro "range.exists(var, predicate)", which is true if for at least one element in
		// range the predicate holds.
		NewReceiverMacro(operators.Exists, 2, makeExists),

		// The macro "range.exists_one(var, predicate)", which is true if for exactly one element
		// in range the predicate holds.
		NewReceiverMacro(operators.ExistsOne, 2, makeExistsOne),

		// The macro "range.map(var, function)", applies the function to the vars in the range.
		NewReceiverMacro(operators.Map, 2, makeMap),

		// The macro "range.map(var, predicate, function)", applies the function to the vars in
		// the range for which the predicate holds true. The other variables are filtered out.
		NewReceiverMacro(operators.Map, 3, makeMap),

		// The macro "range.filter(var, predicate)", filters out the variables for which the
		// predicate is false.
		NewReceiverMacro(operators.Filter, 2, makeFilter),
	}

	// NoMacros list.
	NoMacros = []Macro{}
)

// AccumulatorName is the traditional variable name assigned to the fold accumulator variable.
const AccumulatorName = "__result__"

type quantifierKind int

const (
	quantifierAll quantifierKind = iota
	quantifierExists
	quantifierExistsOne
)

func makeAll(eh ExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *common.Error) {
	return makeQuantifier(quantifierAll, eh, target, args)
}

func makeExists(eh ExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *common.Error) {
	return makeQuantifier(quantifierExists, eh, target, args)
}

func makeExistsOne(eh ExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *common.Error) {
	return makeQuantifier(quantifierExistsOne, eh, target, args)
}

func makeQuantifier(kind quantifierKind, eh ExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *common.Error) {
	v, found := extractIdent(args[0])
	if !found {
		location := eh.OffsetLocation(args[0].GetId())
		return nil, &common.Error{
			Message:  "argument must be a simple name",
			Location: location}
	}
	accuIdent := func() *exprpb.Expr {
		return eh.Ident(AccumulatorName)
	}

	var init *exprpb.Expr
	var condition *exprpb.Expr
	var step *exprpb.Expr
	var result *exprpb.Expr
	switch kind {
	case quantifierAll:
		init = eh.LiteralBool(true)
		condition = eh.GlobalCall(operators.NotStrictlyFalse, accuIdent())
		step = eh.GlobalCall(operators.LogicalAnd, accuIdent(), args[1])
		result = accuIdent()
	case quantifierExists:
		init = eh.LiteralBool(false)
		condition = eh.GlobalCall(
			operators.NotStrictlyFalse,
			eh.GlobalCall(operators.LogicalNot, accuIdent()))
		step = eh.GlobalCall(operators.LogicalOr, accuIdent(), args[1])
		result = accuIdent()
	case quantifierExistsOne:
		zeroExpr := eh.LiteralInt(0)
		oneExpr := eh.LiteralInt(1)
		init = zeroExpr
		condition = eh.LiteralBool(true)
		step = eh.GlobalCall(operators.Conditional, args[1],
			eh.GlobalCall(operators.Add, accuIdent(), oneExpr), accuIdent())
		result = eh.GlobalCall(operators.Equals, accuIdent(), oneExpr)
	default:
		return nil, &common.Error{Message: fmt.Sprintf("unrecognized quantifier '%v'", kind)}
	}
	return eh.Fold(v, target, AccumulatorName, init, condition, step, result), nil
}

func makeMap(eh ExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *common.Error) {
	v, found := extractIdent(args[0])
	if !found {
		return nil, &common.Error{Message: "argument is not an identifier"}
	}

	var fn *exprpb.Expr
	var filter *exprpb.Expr

	if len(args) == 3 {
		filter = args[1]
		fn = args[2]
	} else {
		filter = nil
		fn = args[1]
	}

	accuExpr := eh.Ident(AccumulatorName)
	init := eh.NewList()
	condition := eh.LiteralBool(true)
	// TODO: use compiler internal method for faster, stateful add.
	step := eh.GlobalCall(operators.Add, accuExpr, eh.NewList(fn))

	if filter != nil {
		step = eh.GlobalCall(operators.Conditional, filter, step, accuExpr)
	}
	return eh.Fold(v, target, AccumulatorName, init, condition, step, accuExpr), nil
}

func makeFilter(eh ExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *common.Error) {
	v, found := extractIdent(args[0])
	if !found {
		return nil, &common.Error{Message: "argument is not an identifier"}
	}

	filter := args[1]
	accuExpr := eh.Ident(AccumulatorName)
	init := eh.NewList()
	condition := eh.LiteralBool(true)
	// TODO: use compiler internal method for faster, stateful add.
	step := eh.GlobalCall(operators.Add, accuExpr, eh.NewList(args[0]))
	step = eh.GlobalCall(operators.Conditional, filter, step, accuExpr)
	return eh.Fold(v, target, AccumulatorName, init, condition, step, accuExpr), nil
}

func extractIdent(e *exprpb.Expr) (string, bool) {
	switch e.ExprKind.(type) {
	case *exprpb.Expr_IdentExpr:
		return e.GetIdentExpr().GetName(), true
	}
	return "", false
}

func makeHas(eh ExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *common.Error) {
	if s, ok := args[0].ExprKind.(*exprpb.Expr_SelectExpr); ok {
		return eh.PresenceTest(s.SelectExpr.GetOperand(), s.SelectExpr.GetField()), nil
	}
	return nil, &common.Error{Message: "invalid argument to has() macro"}
}
