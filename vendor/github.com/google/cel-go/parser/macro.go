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
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/operators"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

// MacroOpt defines a functional option for configuring macro behavior.
type MacroOpt func(*macro) *macro

// MacroDocs configures a list of strings into a multiline description for the macro.
func MacroDocs(docs ...string) MacroOpt {
	return func(m *macro) *macro {
		m.doc = common.MultilineDescription(docs...)
		return m
	}
}

// MacroExamples configures a list of examples, either as a string or common.MultilineString,
// into an example set to be provided with the macro Documentation() call.
func MacroExamples(examples ...string) MacroOpt {
	return func(m *macro) *macro {
		m.examples = examples
		return m
	}
}

// NewGlobalMacro creates a Macro for a global function with the specified arg count.
func NewGlobalMacro(function string, argCount int, expander MacroExpander, opts ...MacroOpt) Macro {
	m := &macro{
		function: function,
		argCount: argCount,
		expander: expander}
	for _, opt := range opts {
		m = opt(m)
	}
	return m
}

// NewReceiverMacro creates a Macro for a receiver function matching the specified arg count.
func NewReceiverMacro(function string, argCount int, expander MacroExpander, opts ...MacroOpt) Macro {
	m := &macro{
		function:      function,
		argCount:      argCount,
		expander:      expander,
		receiverStyle: true}
	for _, opt := range opts {
		m = opt(m)
	}
	return m
}

// NewGlobalVarArgMacro creates a Macro for a global function with a variable arg count.
func NewGlobalVarArgMacro(function string, expander MacroExpander, opts ...MacroOpt) Macro {
	m := &macro{
		function:    function,
		expander:    expander,
		varArgStyle: true}
	for _, opt := range opts {
		m = opt(m)
	}
	return m
}

// NewReceiverVarArgMacro creates a Macro for a receiver function matching a variable arg count.
func NewReceiverVarArgMacro(function string, expander MacroExpander, opts ...MacroOpt) Macro {
	m := &macro{
		function:      function,
		expander:      expander,
		receiverStyle: true,
		varArgStyle:   true}
	for _, opt := range opts {
		m = opt(m)
	}
	return m
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
	doc           string
	examples      []string
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

// Documentation generates documentation and examples for the macro.
func (m *macro) Documentation() *common.Doc {
	examples := make([]*common.Doc, len(m.examples))
	for i, ex := range m.examples {
		examples[i] = common.NewExampleDoc(ex)
	}
	return common.NewMacroDoc(m.Function(), m.doc, examples...)
}

func makeMacroKey(name string, args int, receiverStyle bool) string {
	return fmt.Sprintf("%s:%d:%v", name, args, receiverStyle)
}

func makeVarArgMacroKey(name string, receiverStyle bool) string {
	return fmt.Sprintf("%s:*:%v", name, receiverStyle)
}

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
type MacroExpander func(eh ExprHelper, target ast.Expr, args []ast.Expr) (ast.Expr, *common.Error)

// ExprHelper assists with the creation of Expr values in a manner which is consistent
// the internal semantics and id generation behaviors of the parser and checker libraries.
type ExprHelper interface {
	// Copy the input expression with a brand new set of identifiers.
	Copy(ast.Expr) ast.Expr

	// Literal creates an Expr value for a scalar literal value.
	NewLiteral(value ref.Val) ast.Expr

	// NewList creates a list literal instruction with an optional set of elements.
	NewList(elems ...ast.Expr) ast.Expr

	// NewMap creates a CreateStruct instruction for a map where the map is comprised of the
	// optional set of key, value entries.
	NewMap(entries ...ast.EntryExpr) ast.Expr

	// NewMapEntry creates a Map Entry for the key, value pair.
	NewMapEntry(key ast.Expr, val ast.Expr, optional bool) ast.EntryExpr

	// NewStruct creates a struct literal expression with an optional set of field initializers.
	NewStruct(typeName string, fieldInits ...ast.EntryExpr) ast.Expr

	// NewStructField creates a new struct field initializer from the field name and value.
	NewStructField(field string, init ast.Expr, optional bool) ast.EntryExpr

	// NewComprehension creates a new one-variable comprehension instruction.
	//
	// - iterRange represents the expression that resolves to a list or map where the elements or
	//   keys (respectively) will be iterated over.
	// - iterVar is the variable name for the list element value, or the map key, depending on the
	//   range type.
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
	NewComprehension(iterRange ast.Expr,
		iterVar,
		accuVar string,
		accuInit,
		condition,
		step,
		result ast.Expr) ast.Expr

	// NewComprehensionTwoVar creates a new two-variable comprehension instruction.
	//
	// - iterRange represents the expression that resolves to a list or map where the elements or
	//   keys (respectively) will be iterated over.
	// - iterVar is the iteration variable assigned to the list index or the map key.
	// - iterVar2 is the iteration variable assigned to the list element value or the map key value.
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
	NewComprehensionTwoVar(iterRange ast.Expr,
		iterVar,
		iterVar2,
		accuVar string,
		accuInit,
		condition,
		step,
		result ast.Expr) ast.Expr

	// NewIdent creates an identifier Expr value.
	NewIdent(name string) ast.Expr

	// NewAccuIdent returns an accumulator identifier for use with comprehension results.
	NewAccuIdent() ast.Expr

	// AccuIdentName returns the name of the accumulator identifier.
	AccuIdentName() string

	// NewCall creates a function call Expr value for a global (free) function.
	NewCall(function string, args ...ast.Expr) ast.Expr

	// NewMemberCall creates a function call Expr value for a receiver-style function.
	NewMemberCall(function string, target ast.Expr, args ...ast.Expr) ast.Expr

	// NewPresenceTest creates a Select TestOnly Expr value for modelling has() semantics.
	NewPresenceTest(operand ast.Expr, field string) ast.Expr

	// NewSelect create a field traversal Expr value.
	NewSelect(operand ast.Expr, field string) ast.Expr

	// OffsetLocation returns the Location of the expression identifier.
	OffsetLocation(exprID int64) common.Location

	// NewError associates an error message with a given expression id.
	NewError(exprID int64, message string) *common.Error
}

var (
	// HasMacro expands "has(m.f)" which tests the presence of a field, avoiding the need to
	// specify the field as a string.
	HasMacro = NewGlobalMacro(operators.Has, 1, MakeHas,
		MacroDocs(
			`check a protocol buffer message for the presence of a field, or check a map`,
			`for the presence of a string key.`,
			`Only map accesses using the select notation are supported.`),
		MacroExamples(
			common.MultilineDescription(
				`// true if the 'address' field exists in the 'user' message`,
				`has(user.address)`),
			common.MultilineDescription(
				`// test whether the 'key_name' is set on the map which defines it`,
				`has({'key_name': 'value'}.key_name) // true`),
			common.MultilineDescription(
				`// test whether the 'id' field is set to a non-default value on the Expr{} message literal`,
				`has(Expr{}.id) // false`),
		))

	// AllMacro expands "range.all(var, predicate)" into a comprehension which ensures that all
	// elements in the range satisfy the predicate.
	AllMacro = NewReceiverMacro(operators.All, 2, MakeAll,
		MacroDocs(`tests whether all elements in the input list or all keys in a map`,
			`satisfy the given predicate. The all macro behaves in a manner consistent with`,
			`the Logical AND operator including in how it absorbs errors and short-circuits.`),
		MacroExamples(
			`[1, 2, 3].all(x, x > 0) // true`,
			`[1, 2, 0].all(x, x > 0) // false`,
			`['apple', 'banana', 'cherry'].all(fruit, fruit.size() > 3) // true`,
			`[3.14, 2.71, 1.61].all(num, num < 3.0) // false`,
			`{'a': 1, 'b': 2, 'c': 3}.all(key, key != 'b') // false`,
			common.MultilineDescription(
				`// an empty list or map as the range will result in a trivially true result`,
				`[].all(x, x > 0) // true`),
		))

	// ExistsMacro expands "range.exists(var, predicate)" into a comprehension which ensures that
	// some element in the range satisfies the predicate.
	ExistsMacro = NewReceiverMacro(operators.Exists, 2, MakeExists,
		MacroDocs(`tests whether any value in the list or any key in the map`,
			`satisfies the predicate expression. The exists macro behaves in a manner`,
			`consistent with the Logical OR operator including in how it absorbs errors and`,
			`short-circuits.`),
		MacroExamples(
			`[1, 2, 3].exists(i, i % 2 != 0) // true`,
			`[0, -1, 5].exists(num, num < 0) // true`,
			`{'x': 'foo', 'y': 'bar'}.exists(key, key.startsWith('z')) // false`,
			common.MultilineDescription(
				`// an empty list or map as the range will result in a trivially false result`,
				`[].exists(i, i > 0) // false`),
			common.MultilineDescription(
				`// test whether a key name equalling 'iss' exists in the map and the`,
				`// value contains the substring 'cel.dev'`,
				`// tokens = {'sub': 'me', 'iss': 'https://issuer.cel.dev'}`,
				`tokens.exists(k, k == 'iss' && tokens[k].contains('cel.dev'))`),
		))

	// ExistsOneMacro expands "range.exists_one(var, predicate)", which is true if for exactly one
	// element in range the predicate holds.
	// Deprecated: Use ExistsOneMacroNew
	ExistsOneMacro = NewReceiverMacro(operators.ExistsOne, 2, MakeExistsOne,
		MacroDocs(`tests whether exactly one list element or map key satisfies`,
			`the predicate expression. This macro does not short-circuit in order to remain`,
			`consistent with logical operators being the only operators which can absorb`,
			`errors within CEL.`),
		MacroExamples(
			`[1, 2, 2].exists_one(i, i < 2) // true`,
			`{'a': 'hello', 'aa': 'hellohello'}.exists_one(k, k.startsWith('a')) // false`,
			`[1, 2, 3, 4].exists_one(num, num % 2 == 0) // false`,
			common.MultilineDescription(
				`// ensure exactly one key in the map ends in @acme.co`,
				`{'wiley@acme.co': 'coyote', 'aa@milne.co': 'bear'}.exists_one(k, k.endsWith('@acme.co')) // true`),
		))

	// ExistsOneMacroNew expands "range.existsOne(var, predicate)", which is true if for exactly one
	// element in range the predicate holds.
	ExistsOneMacroNew = NewReceiverMacro("existsOne", 2, MakeExistsOne,
		MacroDocs(
			`tests whether exactly one list element or map key satisfies the predicate`,
			`expression. This macro does not short-circuit in order to remain consistent`,
			`with logical operators being the only operators which can absorb errors`,
			`within CEL.`),
		MacroExamples(
			`[1, 2, 2].existsOne(i, i < 2) // true`,
			`{'a': 'hello', 'aa': 'hellohello'}.existsOne(k, k.startsWith('a')) // false`,
			`[1, 2, 3, 4].existsOne(num, num % 2 == 0) // false`,
			common.MultilineDescription(
				`// ensure exactly one key in the map ends in @acme.co`,
				`{'wiley@acme.co': 'coyote', 'aa@milne.co': 'bear'}.existsOne(k, k.endsWith('@acme.co')) // true`),
		))

	// MapMacro expands "range.map(var, function)" into a comprehension which applies the function
	// to each element in the range to produce a new list.
	MapMacro = NewReceiverMacro(operators.Map, 2, MakeMap,
		MacroDocs("the three-argument form of map transforms all elements in the input range."),
		MacroExamples(
			`[1, 2, 3].map(x, x * 2) // [2, 4, 6]`,
			`[5, 10, 15].map(x, x / 5) // [1, 2, 3]`,
			`['apple', 'banana'].map(fruit, fruit.upperAscii()) // ['APPLE', 'BANANA']`,
			common.MultilineDescription(
				`// Combine all map key-value pairs into a list`,
				`{'hi': 'you', 'howzit': 'bruv'}.map(k,`,
				`    k + ":" + {'hi': 'you', 'howzit': 'bruv'}[k]) // ['hi:you', 'howzit:bruv']`),
		))

	// MapFilterMacro expands "range.map(var, predicate, function)" into a comprehension which
	// first filters the elements in the range by the predicate, then applies the transform function
	// to produce a new list.
	MapFilterMacro = NewReceiverMacro(operators.Map, 3, MakeMap,
		MacroDocs(`the four-argument form of the map transforms only elements which satisfy`,
			`the predicate which is equivalent to chaining the filter and three-argument`,
			`map macros together.`),
		MacroExamples(
			common.MultilineDescription(
				`// multiply only numbers divisible two, by 2`,
				`[1, 2, 3, 4].map(num, num % 2 == 0, num * 2) // [4, 8]`),
		))

	// FilterMacro expands "range.filter(var, predicate)" into a comprehension which filters
	// elements in the range, producing a new list from the elements that satisfy the predicate.
	FilterMacro = NewReceiverMacro(operators.Filter, 2, MakeFilter,
		MacroDocs(`returns a list containing only the elements from the input list`,
			`that satisfy the given predicate`),
		MacroExamples(
			`[1, 2, 3].filter(x, x > 1) // [2, 3]`,
			`['cat', 'dog', 'bird', 'fish'].filter(pet, pet.size() == 3) // ['cat', 'dog']`,
			`[{'a': 10, 'b': 5, 'c': 20}].map(m, m.filter(key, m[key] > 10)) // [['c']]`,
			common.MultilineDescription(
				`// filter a list to select only emails with the @cel.dev suffix`,
				`['alice@buf.io', 'tristan@cel.dev'].filter(v, v.endsWith('@cel.dev')) // ['tristan@cel.dev']`),
			common.MultilineDescription(
				`// filter a map into a list, selecting only the values for keys that start with 'http-auth'`,
				`{'http-auth-agent': 'secret', 'user-agent': 'mozilla'}.filter(k,`,
				`     k.startsWith('http-auth')) // ['secret']`),
		))

	// AllMacros includes the list of all spec-supported macros.
	AllMacros = []Macro{
		HasMacro,
		AllMacro,
		ExistsMacro,
		ExistsOneMacro,
		ExistsOneMacroNew,
		MapMacro,
		MapFilterMacro,
		FilterMacro,
	}

	// NoMacros list.
	NoMacros = []Macro{}
)

// AccumulatorName is the traditional variable name assigned to the fold accumulator variable.
const AccumulatorName = "__result__"

// HiddenAccumulatorName is a proposed update to the default fold accumlator variable.
// @result is not normally accessible from source, preventing accidental or intentional collisions
// in user expressions.
const HiddenAccumulatorName = "@result"

type quantifierKind int

const (
	quantifierAll quantifierKind = iota
	quantifierExists
	quantifierExistsOne
)

// MakeAll expands the input call arguments into a comprehension that returns true if all of the
// elements in the range match the predicate expressions:
// <iterRange>.all(<iterVar>, <predicate>)
func MakeAll(eh ExprHelper, target ast.Expr, args []ast.Expr) (ast.Expr, *common.Error) {
	return makeQuantifier(quantifierAll, eh, target, args)
}

// MakeExists expands the input call arguments into a comprehension that returns true if any of the
// elements in the range match the predicate expressions:
// <iterRange>.exists(<iterVar>, <predicate>)
func MakeExists(eh ExprHelper, target ast.Expr, args []ast.Expr) (ast.Expr, *common.Error) {
	return makeQuantifier(quantifierExists, eh, target, args)
}

// MakeExistsOne expands the input call arguments into a comprehension that returns true if exactly
// one of the elements in the range match the predicate expressions:
// <iterRange>.exists_one(<iterVar>, <predicate>)
func MakeExistsOne(eh ExprHelper, target ast.Expr, args []ast.Expr) (ast.Expr, *common.Error) {
	return makeQuantifier(quantifierExistsOne, eh, target, args)
}

// MakeMap expands the input call arguments into a comprehension that transforms each element in the
// input to produce an output list.
//
// There are two call patterns supported by map:
//
//	<iterRange>.map(<iterVar>, <transform>)
//	<iterRange>.map(<iterVar>, <predicate>, <transform>)
//
// In the second form only iterVar values which return true when provided to the predicate expression
// are transformed.
func MakeMap(eh ExprHelper, target ast.Expr, args []ast.Expr) (ast.Expr, *common.Error) {
	v, found := extractIdent(args[0])
	if !found {
		return nil, eh.NewError(args[0].ID(), "argument is not an identifier")
	}
	accu := eh.AccuIdentName()
	if v == accu || v == AccumulatorName {
		return nil, eh.NewError(args[0].ID(), "iteration variable overwrites accumulator variable")
	}

	var fn ast.Expr
	var filter ast.Expr

	if len(args) == 3 {
		filter = args[1]
		fn = args[2]
	} else {
		filter = nil
		fn = args[1]
	}

	init := eh.NewList()
	condition := eh.NewLiteral(types.True)
	step := eh.NewCall(operators.Add, eh.NewAccuIdent(), eh.NewList(fn))

	if filter != nil {
		step = eh.NewCall(operators.Conditional, filter, step, eh.NewAccuIdent())
	}
	return eh.NewComprehension(target, v, accu, init, condition, step, eh.NewAccuIdent()), nil
}

// MakeFilter expands the input call arguments into a comprehension which produces a list which contains
// only elements which match the provided predicate expression:
// <iterRange>.filter(<iterVar>, <predicate>)
func MakeFilter(eh ExprHelper, target ast.Expr, args []ast.Expr) (ast.Expr, *common.Error) {
	v, found := extractIdent(args[0])
	if !found {
		return nil, eh.NewError(args[0].ID(), "argument is not an identifier")
	}
	accu := eh.AccuIdentName()
	if v == accu || v == AccumulatorName {
		return nil, eh.NewError(args[0].ID(), "iteration variable overwrites accumulator variable")
	}

	filter := args[1]
	init := eh.NewList()
	condition := eh.NewLiteral(types.True)
	step := eh.NewCall(operators.Add, eh.NewAccuIdent(), eh.NewList(args[0]))
	step = eh.NewCall(operators.Conditional, filter, step, eh.NewAccuIdent())
	return eh.NewComprehension(target, v, accu, init, condition, step, eh.NewAccuIdent()), nil
}

// MakeHas expands the input call arguments into a presence test, e.g. has(<operand>.field)
func MakeHas(eh ExprHelper, target ast.Expr, args []ast.Expr) (ast.Expr, *common.Error) {
	if args[0].Kind() == ast.SelectKind {
		s := args[0].AsSelect()
		return eh.NewPresenceTest(s.Operand(), s.FieldName()), nil
	}
	return nil, eh.NewError(args[0].ID(), "invalid argument to has() macro")
}

func makeQuantifier(kind quantifierKind, eh ExprHelper, target ast.Expr, args []ast.Expr) (ast.Expr, *common.Error) {
	v, found := extractIdent(args[0])
	if !found {
		return nil, eh.NewError(args[0].ID(), "argument must be a simple name")
	}
	accu := eh.AccuIdentName()
	if v == accu || v == AccumulatorName {
		return nil, eh.NewError(args[0].ID(), "iteration variable overwrites accumulator variable")
	}

	var init ast.Expr
	var condition ast.Expr
	var step ast.Expr
	var result ast.Expr
	switch kind {
	case quantifierAll:
		init = eh.NewLiteral(types.True)
		condition = eh.NewCall(operators.NotStrictlyFalse, eh.NewAccuIdent())
		step = eh.NewCall(operators.LogicalAnd, eh.NewAccuIdent(), args[1])
		result = eh.NewAccuIdent()
	case quantifierExists:
		init = eh.NewLiteral(types.False)
		condition = eh.NewCall(
			operators.NotStrictlyFalse,
			eh.NewCall(operators.LogicalNot, eh.NewAccuIdent()))
		step = eh.NewCall(operators.LogicalOr, eh.NewAccuIdent(), args[1])
		result = eh.NewAccuIdent()
	case quantifierExistsOne:
		init = eh.NewLiteral(types.Int(0))
		condition = eh.NewLiteral(types.True)
		step = eh.NewCall(operators.Conditional, args[1],
			eh.NewCall(operators.Add, eh.NewAccuIdent(), eh.NewLiteral(types.Int(1))), eh.NewAccuIdent())
		result = eh.NewCall(operators.Equals, eh.NewAccuIdent(), eh.NewLiteral(types.Int(1)))
	default:
		return nil, eh.NewError(args[0].ID(), fmt.Sprintf("unrecognized quantifier '%v'", kind))
	}
	return eh.NewComprehension(target, v, accu, init, condition, step, result), nil
}

func extractIdent(e ast.Expr) (string, bool) {
	switch e.Kind() {
	case ast.IdentKind:
		return e.AsIdent(), true
	}
	return "", false
}
