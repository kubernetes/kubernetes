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
	"fmt"

	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/parser"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// Macro describes a function signature to match and the MacroExpander to apply.
//
// Note: when a Macro should apply to multiple overloads (based on arg count) of a given function,
// a Macro should be created per arg-count or as a var arg macro.
type Macro = parser.Macro

// MacroFactory defines an expansion function which converts a call and its arguments to a cel.Expr value.
type MacroFactory = parser.MacroExpander

// MacroExprFactory assists with the creation of Expr values in a manner which is consistent
// the internal semantics and id generation behaviors of the parser and checker libraries.
type MacroExprFactory = parser.ExprHelper

// MacroExpander converts a call and its associated arguments into a protobuf Expr representation.
//
// If the MacroExpander determines within the implementation that an expansion is not needed it may return
// a nil Expr value to indicate a non-match. However, if an expansion is to be performed, but the arguments
// are not well-formed, the result of the expansion will be an error.
//
// The MacroExpander accepts as arguments a MacroExprHelper as well as the arguments used in the function call
// and produces as output an Expr ast node.
//
// Note: when the Macro.IsReceiverStyle() method returns true, the target argument will be nil.
type MacroExpander func(eh MacroExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *Error)

// MacroExprHelper exposes helper methods for creating new expressions within a CEL abstract syntax tree.
// ExprHelper assists with the manipulation of proto-based Expr values in a manner which is
// consistent with the source position and expression id generation code leveraged by both
// the parser and type-checker.
type MacroExprHelper interface {
	// Copy the input expression with a brand new set of identifiers.
	Copy(*exprpb.Expr) *exprpb.Expr

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
	NewMapEntry(key *exprpb.Expr, val *exprpb.Expr, optional bool) *exprpb.Expr_CreateStruct_Entry

	// NewObject creates a CreateStruct instruction for an object with a given type name and
	// optional set of field initializers.
	NewObject(typeName string, fieldInits ...*exprpb.Expr_CreateStruct_Entry) *exprpb.Expr

	// NewObjectFieldInit creates a new Object field initializer from the field name and value.
	NewObjectFieldInit(field string, init *exprpb.Expr, optional bool) *exprpb.Expr_CreateStruct_Entry

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

	// AccuIdent returns an accumulator identifier for use with comprehension results.
	AccuIdent() *exprpb.Expr

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

	// NewError associates an error message with a given expression id.
	NewError(exprID int64, message string) *Error
}

// GlobalMacro creates a Macro for a global function with the specified arg count.
func GlobalMacro(function string, argCount int, factory MacroFactory) Macro {
	return parser.NewGlobalMacro(function, argCount, factory)
}

// ReceiverMacro creates a Macro for a receiver function matching the specified arg count.
func ReceiverMacro(function string, argCount int, factory MacroFactory) Macro {
	return parser.NewReceiverMacro(function, argCount, factory)
}

// GlobalVarArgMacro creates a Macro for a global function with a variable arg count.
func GlobalVarArgMacro(function string, factory MacroFactory) Macro {
	return parser.NewGlobalVarArgMacro(function, factory)
}

// ReceiverVarArgMacro creates a Macro for a receiver function matching a variable arg count.
func ReceiverVarArgMacro(function string, factory MacroFactory) Macro {
	return parser.NewReceiverVarArgMacro(function, factory)
}

// NewGlobalMacro creates a Macro for a global function with the specified arg count.
//
// Deprecated: use GlobalMacro
func NewGlobalMacro(function string, argCount int, expander MacroExpander) Macro {
	expand := adaptingExpander{expander}
	return parser.NewGlobalMacro(function, argCount, expand.Expander)
}

// NewReceiverMacro creates a Macro for a receiver function matching the specified arg count.
//
// Deprecated: use ReceiverMacro
func NewReceiverMacro(function string, argCount int, expander MacroExpander) Macro {
	expand := adaptingExpander{expander}
	return parser.NewReceiverMacro(function, argCount, expand.Expander)
}

// NewGlobalVarArgMacro creates a Macro for a global function with a variable arg count.
//
// Deprecated: use GlobalVarArgMacro
func NewGlobalVarArgMacro(function string, expander MacroExpander) Macro {
	expand := adaptingExpander{expander}
	return parser.NewGlobalVarArgMacro(function, expand.Expander)
}

// NewReceiverVarArgMacro creates a Macro for a receiver function matching a variable arg count.
//
// Deprecated: use ReceiverVarArgMacro
func NewReceiverVarArgMacro(function string, expander MacroExpander) Macro {
	expand := adaptingExpander{expander}
	return parser.NewReceiverVarArgMacro(function, expand.Expander)
}

// HasMacroExpander expands the input call arguments into a presence test, e.g. has(<operand>.field)
func HasMacroExpander(meh MacroExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *Error) {
	ph, err := toParserHelper(meh)
	if err != nil {
		return nil, err
	}
	arg, err := adaptToExpr(args[0])
	if err != nil {
		return nil, err
	}
	if arg.Kind() == ast.SelectKind {
		s := arg.AsSelect()
		return adaptToProto(ph.NewPresenceTest(s.Operand(), s.FieldName()))
	}
	return nil, ph.NewError(arg.ID(), "invalid argument to has() macro")
}

// ExistsMacroExpander expands the input call arguments into a comprehension that returns true if any of the
// elements in the range match the predicate expressions:
// <iterRange>.exists(<iterVar>, <predicate>)
func ExistsMacroExpander(meh MacroExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *Error) {
	ph, err := toParserHelper(meh)
	if err != nil {
		return nil, err
	}
	out, err := parser.MakeExists(ph, mustAdaptToExpr(target), mustAdaptToExprs(args))
	if err != nil {
		return nil, err
	}
	return adaptToProto(out)
}

// ExistsOneMacroExpander expands the input call arguments into a comprehension that returns true if exactly
// one of the elements in the range match the predicate expressions:
// <iterRange>.exists_one(<iterVar>, <predicate>)
func ExistsOneMacroExpander(meh MacroExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *Error) {
	ph, err := toParserHelper(meh)
	if err != nil {
		return nil, err
	}
	out, err := parser.MakeExistsOne(ph, mustAdaptToExpr(target), mustAdaptToExprs(args))
	if err != nil {
		return nil, err
	}
	return adaptToProto(out)
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
func MapMacroExpander(meh MacroExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *Error) {
	ph, err := toParserHelper(meh)
	if err != nil {
		return nil, err
	}
	out, err := parser.MakeMap(ph, mustAdaptToExpr(target), mustAdaptToExprs(args))
	if err != nil {
		return nil, err
	}
	return adaptToProto(out)
}

// FilterMacroExpander expands the input call arguments into a comprehension which produces a list which contains
// only elements which match the provided predicate expression:
// <iterRange>.filter(<iterVar>, <predicate>)
func FilterMacroExpander(meh MacroExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *Error) {
	ph, err := toParserHelper(meh)
	if err != nil {
		return nil, err
	}
	out, err := parser.MakeFilter(ph, mustAdaptToExpr(target), mustAdaptToExprs(args))
	if err != nil {
		return nil, err
	}
	return adaptToProto(out)
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

type adaptingExpander struct {
	legacyExpander MacroExpander
}

func (adapt *adaptingExpander) Expander(eh parser.ExprHelper, target ast.Expr, args []ast.Expr) (ast.Expr, *common.Error) {
	var legacyTarget *exprpb.Expr = nil
	var err *Error = nil
	if target != nil {
		legacyTarget, err = adaptToProto(target)
		if err != nil {
			return nil, err
		}
	}
	legacyArgs := make([]*exprpb.Expr, len(args))
	for i, arg := range args {
		legacyArgs[i], err = adaptToProto(arg)
		if err != nil {
			return nil, err
		}
	}
	ah := &adaptingHelper{modernHelper: eh}
	legacyExpr, err := adapt.legacyExpander(ah, legacyTarget, legacyArgs)
	if err != nil {
		return nil, err
	}
	ex, err := adaptToExpr(legacyExpr)
	if err != nil {
		return nil, err
	}
	return ex, nil
}

func wrapErr(id int64, message string, err error) *common.Error {
	return &common.Error{
		Location: common.NoLocation,
		Message:  fmt.Sprintf("%s: %v", message, err),
		ExprID:   id,
	}
}

type adaptingHelper struct {
	modernHelper parser.ExprHelper
}

// Copy the input expression with a brand new set of identifiers.
func (ah *adaptingHelper) Copy(e *exprpb.Expr) *exprpb.Expr {
	return mustAdaptToProto(ah.modernHelper.Copy(mustAdaptToExpr(e)))
}

// LiteralBool creates an Expr value for a bool literal.
func (ah *adaptingHelper) LiteralBool(value bool) *exprpb.Expr {
	return mustAdaptToProto(ah.modernHelper.NewLiteral(types.Bool(value)))
}

// LiteralBytes creates an Expr value for a byte literal.
func (ah *adaptingHelper) LiteralBytes(value []byte) *exprpb.Expr {
	return mustAdaptToProto(ah.modernHelper.NewLiteral(types.Bytes(value)))
}

// LiteralDouble creates an Expr value for double literal.
func (ah *adaptingHelper) LiteralDouble(value float64) *exprpb.Expr {
	return mustAdaptToProto(ah.modernHelper.NewLiteral(types.Double(value)))
}

// LiteralInt creates an Expr value for an int literal.
func (ah *adaptingHelper) LiteralInt(value int64) *exprpb.Expr {
	return mustAdaptToProto(ah.modernHelper.NewLiteral(types.Int(value)))
}

// LiteralString creates am Expr value for a string literal.
func (ah *adaptingHelper) LiteralString(value string) *exprpb.Expr {
	return mustAdaptToProto(ah.modernHelper.NewLiteral(types.String(value)))
}

// LiteralUint creates an Expr value for a uint literal.
func (ah *adaptingHelper) LiteralUint(value uint64) *exprpb.Expr {
	return mustAdaptToProto(ah.modernHelper.NewLiteral(types.Uint(value)))
}

// NewList creates a CreateList instruction where the list is comprised of the optional set
// of elements provided as arguments.
func (ah *adaptingHelper) NewList(elems ...*exprpb.Expr) *exprpb.Expr {
	return mustAdaptToProto(ah.modernHelper.NewList(mustAdaptToExprs(elems)...))
}

// NewMap creates a CreateStruct instruction for a map where the map is comprised of the
// optional set of key, value entries.
func (ah *adaptingHelper) NewMap(entries ...*exprpb.Expr_CreateStruct_Entry) *exprpb.Expr {
	adaptedEntries := make([]ast.EntryExpr, len(entries))
	for i, e := range entries {
		adaptedEntries[i] = mustAdaptToEntryExpr(e)
	}
	return mustAdaptToProto(ah.modernHelper.NewMap(adaptedEntries...))
}

// NewMapEntry creates a Map Entry for the key, value pair.
func (ah *adaptingHelper) NewMapEntry(key *exprpb.Expr, val *exprpb.Expr, optional bool) *exprpb.Expr_CreateStruct_Entry {
	return mustAdaptToProtoEntry(
		ah.modernHelper.NewMapEntry(mustAdaptToExpr(key), mustAdaptToExpr(val), optional))
}

// NewObject creates a CreateStruct instruction for an object with a given type name and
// optional set of field initializers.
func (ah *adaptingHelper) NewObject(typeName string, fieldInits ...*exprpb.Expr_CreateStruct_Entry) *exprpb.Expr {
	adaptedEntries := make([]ast.EntryExpr, len(fieldInits))
	for i, e := range fieldInits {
		adaptedEntries[i] = mustAdaptToEntryExpr(e)
	}
	return mustAdaptToProto(ah.modernHelper.NewStruct(typeName, adaptedEntries...))
}

// NewObjectFieldInit creates a new Object field initializer from the field name and value.
func (ah *adaptingHelper) NewObjectFieldInit(field string, init *exprpb.Expr, optional bool) *exprpb.Expr_CreateStruct_Entry {
	return mustAdaptToProtoEntry(
		ah.modernHelper.NewStructField(field, mustAdaptToExpr(init), optional))
}

// Fold creates a fold comprehension instruction.
//
//   - iterVar is the iteration variable name.
//   - iterRange represents the expression that resolves to a list or map where the elements or
//     keys (respectively) will be iterated over.
//   - accuVar is the accumulation variable name, typically parser.AccumulatorName.
//   - accuInit is the initial expression whose value will be set for the accuVar prior to
//     folding.
//   - condition is the expression to test to determine whether to continue folding.
//   - step is the expression to evaluation at the conclusion of a single fold iteration.
//   - result is the computation to evaluate at the conclusion of the fold.
//
// The accuVar should not shadow variable names that you would like to reference within the
// environment in the step and condition expressions. Presently, the name __result__ is commonly
// used by built-in macros but this may change in the future.
func (ah *adaptingHelper) Fold(iterVar string,
	iterRange *exprpb.Expr,
	accuVar string,
	accuInit *exprpb.Expr,
	condition *exprpb.Expr,
	step *exprpb.Expr,
	result *exprpb.Expr) *exprpb.Expr {
	return mustAdaptToProto(
		ah.modernHelper.NewComprehension(
			mustAdaptToExpr(iterRange),
			iterVar,
			accuVar,
			mustAdaptToExpr(accuInit),
			mustAdaptToExpr(condition),
			mustAdaptToExpr(step),
			mustAdaptToExpr(result),
		),
	)
}

// Ident creates an identifier Expr value.
func (ah *adaptingHelper) Ident(name string) *exprpb.Expr {
	return mustAdaptToProto(ah.modernHelper.NewIdent(name))
}

// AccuIdent returns an accumulator identifier for use with comprehension results.
func (ah *adaptingHelper) AccuIdent() *exprpb.Expr {
	return mustAdaptToProto(ah.modernHelper.NewAccuIdent())
}

// GlobalCall creates a function call Expr value for a global (free) function.
func (ah *adaptingHelper) GlobalCall(function string, args ...*exprpb.Expr) *exprpb.Expr {
	return mustAdaptToProto(ah.modernHelper.NewCall(function, mustAdaptToExprs(args)...))
}

// ReceiverCall creates a function call Expr value for a receiver-style function.
func (ah *adaptingHelper) ReceiverCall(function string, target *exprpb.Expr, args ...*exprpb.Expr) *exprpb.Expr {
	return mustAdaptToProto(
		ah.modernHelper.NewMemberCall(function, mustAdaptToExpr(target), mustAdaptToExprs(args)...))
}

// PresenceTest creates a Select TestOnly Expr value for modelling has() semantics.
func (ah *adaptingHelper) PresenceTest(operand *exprpb.Expr, field string) *exprpb.Expr {
	op := mustAdaptToExpr(operand)
	return mustAdaptToProto(ah.modernHelper.NewPresenceTest(op, field))
}

// Select create a field traversal Expr value.
func (ah *adaptingHelper) Select(operand *exprpb.Expr, field string) *exprpb.Expr {
	op := mustAdaptToExpr(operand)
	return mustAdaptToProto(ah.modernHelper.NewSelect(op, field))
}

// OffsetLocation returns the Location of the expression identifier.
func (ah *adaptingHelper) OffsetLocation(exprID int64) common.Location {
	return ah.modernHelper.OffsetLocation(exprID)
}

// NewError associates an error message with a given expression id.
func (ah *adaptingHelper) NewError(exprID int64, message string) *Error {
	return ah.modernHelper.NewError(exprID, message)
}

func mustAdaptToExprs(exprs []*exprpb.Expr) []ast.Expr {
	adapted := make([]ast.Expr, len(exprs))
	for i, e := range exprs {
		adapted[i] = mustAdaptToExpr(e)
	}
	return adapted
}

func mustAdaptToExpr(e *exprpb.Expr) ast.Expr {
	out, _ := adaptToExpr(e)
	return out
}

func adaptToExpr(e *exprpb.Expr) (ast.Expr, *Error) {
	if e == nil {
		return nil, nil
	}
	out, err := ast.ProtoToExpr(e)
	if err != nil {
		return nil, wrapErr(e.GetId(), "proto conversion failure", err)
	}
	return out, nil
}

func mustAdaptToEntryExpr(e *exprpb.Expr_CreateStruct_Entry) ast.EntryExpr {
	out, _ := ast.ProtoToEntryExpr(e)
	return out
}

func mustAdaptToProto(e ast.Expr) *exprpb.Expr {
	out, _ := adaptToProto(e)
	return out
}

func adaptToProto(e ast.Expr) (*exprpb.Expr, *Error) {
	if e == nil {
		return nil, nil
	}
	out, err := ast.ExprToProto(e)
	if err != nil {
		return nil, wrapErr(e.ID(), "expr conversion failure", err)
	}
	return out, nil
}

func mustAdaptToProtoEntry(e ast.EntryExpr) *exprpb.Expr_CreateStruct_Entry {
	out, _ := ast.EntryExprToProto(e)
	return out
}

func toParserHelper(meh MacroExprHelper) (parser.ExprHelper, *Error) {
	ah, ok := meh.(*adaptingHelper)
	if !ok {
		return nil, common.NewError(0,
			fmt.Sprintf("unsupported macro helper: %v (%T)", meh, meh),
			common.NoLocation)
	}
	return ah.modernHelper, nil
}
