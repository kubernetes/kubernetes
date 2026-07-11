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
	"sync"

	antlr "github.com/antlr4-go/antlr/v4"

	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

type parserHelper struct {
	exprFactory ast.ExprFactory
	source      common.Source
	sourceInfo  *ast.SourceInfo
	nextID      int64
}

func newParserHelper(source common.Source, fac ast.ExprFactory) *parserHelper {
	return &parserHelper{
		exprFactory: fac,
		source:      source,
		sourceInfo:  ast.NewSourceInfo(source),
		nextID:      1,
	}
}

func (p *parserHelper) getSourceInfo() *ast.SourceInfo {
	return p.sourceInfo
}

func (p *parserHelper) newLiteral(ctx any, value ref.Val) ast.Expr {
	return p.exprFactory.NewLiteral(p.newID(ctx), value)
}

func (p *parserHelper) newLiteralBool(ctx any, value bool) ast.Expr {
	return p.newLiteral(ctx, types.Bool(value))
}

func (p *parserHelper) newLiteralString(ctx any, value string) ast.Expr {
	return p.newLiteral(ctx, types.String(value))
}

func (p *parserHelper) newLiteralBytes(ctx any, value []byte) ast.Expr {
	return p.newLiteral(ctx, types.Bytes(value))
}

func (p *parserHelper) newLiteralInt(ctx any, value int64) ast.Expr {
	return p.newLiteral(ctx, types.Int(value))
}

func (p *parserHelper) newLiteralUint(ctx any, value uint64) ast.Expr {
	return p.newLiteral(ctx, types.Uint(value))
}

func (p *parserHelper) newLiteralDouble(ctx any, value float64) ast.Expr {
	return p.newLiteral(ctx, types.Double(value))
}

func (p *parserHelper) newIdent(ctx any, name string) ast.Expr {
	return p.exprFactory.NewIdent(p.newID(ctx), name)
}

func (p *parserHelper) newSelect(ctx any, operand ast.Expr, field string) ast.Expr {
	return p.exprFactory.NewSelect(p.newID(ctx), operand, field)
}

func (p *parserHelper) newPresenceTest(ctx any, operand ast.Expr, field string) ast.Expr {
	return p.exprFactory.NewPresenceTest(p.newID(ctx), operand, field)
}

func (p *parserHelper) newGlobalCall(ctx any, function string, args ...ast.Expr) ast.Expr {
	return p.exprFactory.NewCall(p.newID(ctx), function, args...)
}

func (p *parserHelper) newReceiverCall(ctx any, function string, target ast.Expr, args ...ast.Expr) ast.Expr {
	return p.exprFactory.NewMemberCall(p.newID(ctx), function, target, args...)
}

func (p *parserHelper) newList(ctx any, elements []ast.Expr, optionals ...int32) ast.Expr {
	return p.exprFactory.NewList(p.newID(ctx), elements, optionals)
}

func (p *parserHelper) newMap(ctx any, entries ...ast.EntryExpr) ast.Expr {
	return p.exprFactory.NewMap(p.newID(ctx), entries)
}

func (p *parserHelper) newMapEntry(entryID int64, key ast.Expr, value ast.Expr, optional bool) ast.EntryExpr {
	return p.exprFactory.NewMapEntry(entryID, key, value, optional)
}

func (p *parserHelper) newObject(ctx any, typeName string, fields ...ast.EntryExpr) ast.Expr {
	return p.exprFactory.NewStruct(p.newID(ctx), typeName, fields)
}

func (p *parserHelper) newObjectField(fieldID int64, field string, value ast.Expr, optional bool) ast.EntryExpr {
	return p.exprFactory.NewStructField(fieldID, field, value, optional)
}

func (p *parserHelper) newComprehension(ctx any,
	iterRange ast.Expr,
	iterVar,
	accuVar string,
	accuInit ast.Expr,
	condition ast.Expr,
	step ast.Expr,
	result ast.Expr) ast.Expr {
	return p.exprFactory.NewComprehension(
		p.newID(ctx), iterRange, iterVar, accuVar, accuInit, condition, step, result)
}

func (p *parserHelper) newComprehensionTwoVar(ctx any,
	iterRange ast.Expr,
	iterVar, iterVar2,
	accuVar string,
	accuInit ast.Expr,
	condition ast.Expr,
	step ast.Expr,
	result ast.Expr) ast.Expr {
	return p.exprFactory.NewComprehensionTwoVar(
		p.newID(ctx), iterRange, iterVar, iterVar2, accuVar, accuInit, condition, step, result)
}

func (p *parserHelper) newID(ctx any) int64 {
	if id, isID := ctx.(int64); isID {
		return id
	}
	return p.id(ctx)
}

func (p *parserHelper) newExpr(ctx any) ast.Expr {
	return p.exprFactory.NewUnspecifiedExpr(p.newID(ctx))
}

func (p *parserHelper) id(ctx any) int64 {
	var offset ast.OffsetRange
	switch c := ctx.(type) {
	case antlr.ParserRuleContext:
		start := c.GetStart()
		offset.Start = p.sourceInfo.ComputeOffset(int32(start.GetLine()), int32(start.GetColumn()))
		offset.Stop = offset.Start + int32(len(c.GetText()))
	case antlr.Token:
		offset.Start = p.sourceInfo.ComputeOffset(int32(c.GetLine()), int32(c.GetColumn()))
		offset.Stop = offset.Start + int32(len(c.GetText()))
	case common.Location:
		offset.Start = p.sourceInfo.ComputeOffsetAbsolute(int32(c.Line()), int32(c.Column()))
		offset.Stop = offset.Start
	case ast.OffsetRange:
		offset = c
	default:
		// This should only happen if the ctx is nil
		return -1
	}
	id := p.nextID
	p.sourceInfo.SetOffsetRange(id, offset)
	p.nextID++
	return id
}

func (p *parserHelper) deleteID(id int64) {
	p.sourceInfo.ClearOffsetRange(id)
	if id == p.nextID-1 {
		p.nextID--
	}
}

func (p *parserHelper) getLocation(id int64) common.Location {
	return p.sourceInfo.GetStartLocation(id)
}

func (p *parserHelper) getLocationByOffset(offset int32) common.Location {
	return p.getSourceInfo().GetLocationByOffset(offset)
}

// buildMacroCallArg iterates the expression and returns a new expression
// where all macros have been replaced by their IDs in MacroCalls
func (p *parserHelper) buildMacroCallArg(expr ast.Expr) ast.Expr {
	if _, found := p.sourceInfo.GetMacroCall(expr.ID()); found {
		return p.exprFactory.NewUnspecifiedExpr(expr.ID())
	}

	switch expr.Kind() {
	case ast.CallKind:
		// Iterate the AST from `expr` recursively looking for macros. Because we are at most
		// starting from the top level macro, this recursion is bounded by the size of the AST. This
		// means that the depth check on the AST during parsing will catch recursion overflows
		// before we get to here.
		call := expr.AsCall()
		macroArgs := make([]ast.Expr, len(call.Args()))
		for index, arg := range call.Args() {
			macroArgs[index] = p.buildMacroCallArg(arg)
		}
		if !call.IsMemberFunction() {
			return p.exprFactory.NewCall(expr.ID(), call.FunctionName(), macroArgs...)
		}
		macroTarget := p.buildMacroCallArg(call.Target())
		return p.exprFactory.NewMemberCall(expr.ID(), call.FunctionName(), macroTarget, macroArgs...)
	case ast.ListKind:
		list := expr.AsList()
		macroListArgs := make([]ast.Expr, list.Size())
		for i, elem := range list.Elements() {
			macroListArgs[i] = p.buildMacroCallArg(elem)
		}
		return p.exprFactory.NewList(expr.ID(), macroListArgs, list.OptionalIndices())
	}
	return expr
}

// addMacroCall adds the macro the the MacroCalls map in source info. If a macro has args/subargs/target
// that are macros, their ID will be stored instead for later self-lookups.
func (p *parserHelper) addMacroCall(exprID int64, function string, target ast.Expr, args ...ast.Expr) {
	macroArgs := make([]ast.Expr, len(args))
	for index, arg := range args {
		macroArgs[index] = p.buildMacroCallArg(arg)
	}
	if target == nil {
		p.sourceInfo.SetMacroCall(exprID, p.exprFactory.NewCall(0, function, macroArgs...))
		return
	}
	macroTarget := target
	if _, found := p.sourceInfo.GetMacroCall(target.ID()); found {
		macroTarget = p.exprFactory.NewUnspecifiedExpr(target.ID())
	} else {
		macroTarget = p.buildMacroCallArg(target)
	}
	p.sourceInfo.SetMacroCall(exprID, p.exprFactory.NewMemberCall(0, function, macroTarget, macroArgs...))
}

// logicManager compacts logical trees into a more efficient structure which is semantically
// equivalent with how the logic graph is constructed by the ANTLR parser.
//
// The purpose of the logicManager is to ensure a compact serialization format for the logical &&, ||
// operators which have a tendency to create long DAGs which are skewed in one direction. Since the
// operators are commutative re-ordering the terms *must not* affect the evaluation result.
//
// The logic manager will either render the terms to N-chained && / || operators as a single logical
// call with N-terms, or will rebalance the tree. Rebalancing the terms is a safe, if somewhat
// controversial choice as it alters the traditional order of execution assumptions present in most
// expressions.
type logicManager struct {
	exprFactory  ast.ExprFactory
	function     string
	terms        []ast.Expr
	ops          []int64
	variadicASTs bool
}

// newVariadicLogicManager creates a logic manager instance bound to a specific function and its first term.
func newVariadicLogicManager(fac ast.ExprFactory, function string, term ast.Expr) *logicManager {
	return &logicManager{
		exprFactory:  fac,
		function:     function,
		terms:        []ast.Expr{term},
		ops:          []int64{},
		variadicASTs: true,
	}
}

// newBalancingLogicManager creates a logic manager instance bound to a specific function and its first term.
func newBalancingLogicManager(fac ast.ExprFactory, function string, term ast.Expr) *logicManager {
	return &logicManager{
		exprFactory:  fac,
		function:     function,
		terms:        []ast.Expr{term},
		ops:          []int64{},
		variadicASTs: false,
	}
}

// addTerm adds an operation identifier and term to the set of terms to be balanced.
func (l *logicManager) addTerm(op int64, term ast.Expr) {
	l.terms = append(l.terms, term)
	l.ops = append(l.ops, op)
}

// toExpr renders the logic graph into an Expr value, either balancing a tree of logical
// operations or creating a variadic representation of the logical operator.
func (l *logicManager) toExpr() ast.Expr {
	if len(l.terms) == 1 {
		return l.terms[0]
	}
	if l.variadicASTs {
		return l.exprFactory.NewCall(l.ops[0], l.function, l.terms...)
	}
	return l.balancedTree(0, len(l.ops)-1)
}

// balancedTree recursively balances the terms provided to a commutative operator.
func (l *logicManager) balancedTree(lo, hi int) ast.Expr {
	mid := (lo + hi + 1) / 2

	var left ast.Expr
	if mid == lo {
		left = l.terms[mid]
	} else {
		left = l.balancedTree(lo, mid-1)
	}

	var right ast.Expr
	if mid == hi {
		right = l.terms[mid+1]
	} else {
		right = l.balancedTree(mid+1, hi)
	}
	return l.exprFactory.NewCall(l.ops[mid], l.function, left, right)
}

type exprHelper struct {
	*parserHelper
	id int64
}

func (e *exprHelper) nextMacroID() int64 {
	return e.parserHelper.id(e.parserHelper.getLocation(e.id))
}

// Copy implements the ExprHelper interface method by producing a copy of the input Expr value
// with a fresh set of numeric identifiers the Expr and all its descendants.
func (e *exprHelper) Copy(expr ast.Expr) ast.Expr {
	offsetRange, _ := e.parserHelper.sourceInfo.GetOffsetRange(expr.ID())
	copyID := e.parserHelper.newID(offsetRange)
	switch expr.Kind() {
	case ast.LiteralKind:
		return e.exprFactory.NewLiteral(copyID, expr.AsLiteral())
	case ast.IdentKind:
		return e.exprFactory.NewIdent(copyID, expr.AsIdent())
	case ast.SelectKind:
		sel := expr.AsSelect()
		op := e.Copy(sel.Operand())
		if sel.IsTestOnly() {
			return e.exprFactory.NewPresenceTest(copyID, op, sel.FieldName())
		}
		return e.exprFactory.NewSelect(copyID, op, sel.FieldName())
	case ast.CallKind:
		call := expr.AsCall()
		args := call.Args()
		argsCopy := make([]ast.Expr, len(args))
		for i, arg := range args {
			argsCopy[i] = e.Copy(arg)
		}
		if !call.IsMemberFunction() {
			return e.exprFactory.NewCall(copyID, call.FunctionName(), argsCopy...)
		}
		return e.exprFactory.NewMemberCall(copyID, call.FunctionName(), e.Copy(call.Target()), argsCopy...)
	case ast.ListKind:
		list := expr.AsList()
		elems := list.Elements()
		elemsCopy := make([]ast.Expr, len(elems))
		for i, elem := range elems {
			elemsCopy[i] = e.Copy(elem)
		}
		return e.exprFactory.NewList(copyID, elemsCopy, list.OptionalIndices())
	case ast.MapKind:
		m := expr.AsMap()
		entries := m.Entries()
		entriesCopy := make([]ast.EntryExpr, len(entries))
		for i, en := range entries {
			entry := en.AsMapEntry()
			entryID := e.nextMacroID()
			entriesCopy[i] = e.exprFactory.NewMapEntry(entryID,
				e.Copy(entry.Key()), e.Copy(entry.Value()), entry.IsOptional())
		}
		return e.exprFactory.NewMap(copyID, entriesCopy)
	case ast.StructKind:
		s := expr.AsStruct()
		fields := s.Fields()
		fieldsCopy := make([]ast.EntryExpr, len(fields))
		for i, f := range fields {
			field := f.AsStructField()
			fieldID := e.nextMacroID()
			fieldsCopy[i] = e.exprFactory.NewStructField(fieldID,
				field.Name(), e.Copy(field.Value()), field.IsOptional())
		}
		return e.exprFactory.NewStruct(copyID, s.TypeName(), fieldsCopy)
	case ast.ComprehensionKind:
		compre := expr.AsComprehension()
		iterRange := e.Copy(compre.IterRange())
		accuInit := e.Copy(compre.AccuInit())
		cond := e.Copy(compre.LoopCondition())
		step := e.Copy(compre.LoopStep())
		result := e.Copy(compre.Result())
		// All comprehensions can be represented by the two-variable comprehension since the
		// differentiation between one and two-variable is whether the iterVar2 value is non-empty.
		return e.exprFactory.NewComprehensionTwoVar(copyID,
			iterRange, compre.IterVar(), compre.IterVar2(), compre.AccuVar(), accuInit, cond, step, result)
	}
	return e.exprFactory.NewUnspecifiedExpr(copyID)
}

// NewLiteral implements the ExprHelper interface method.
func (e *exprHelper) NewLiteral(value ref.Val) ast.Expr {
	return e.exprFactory.NewLiteral(e.nextMacroID(), value)
}

// NewList implements the ExprHelper interface method.
func (e *exprHelper) NewList(elems ...ast.Expr) ast.Expr {
	return e.exprFactory.NewList(e.nextMacroID(), elems, []int32{})
}

// NewMap implements the ExprHelper interface method.
func (e *exprHelper) NewMap(entries ...ast.EntryExpr) ast.Expr {
	return e.exprFactory.NewMap(e.nextMacroID(), entries)
}

// NewMapEntry implements the ExprHelper interface method.
func (e *exprHelper) NewMapEntry(key ast.Expr, val ast.Expr, optional bool) ast.EntryExpr {
	return e.exprFactory.NewMapEntry(e.nextMacroID(), key, val, optional)
}

// NewStruct implements the ExprHelper interface method.
func (e *exprHelper) NewStruct(typeName string, fieldInits ...ast.EntryExpr) ast.Expr {
	return e.exprFactory.NewStruct(e.nextMacroID(), typeName, fieldInits)
}

// NewStructField implements the ExprHelper interface method.
func (e *exprHelper) NewStructField(field string, init ast.Expr, optional bool) ast.EntryExpr {
	return e.exprFactory.NewStructField(e.nextMacroID(), field, init, optional)
}

// NewComprehension implements the ExprHelper interface method.
func (e *exprHelper) NewComprehension(
	iterRange ast.Expr,
	iterVar string,
	accuVar string,
	accuInit ast.Expr,
	condition ast.Expr,
	step ast.Expr,
	result ast.Expr) ast.Expr {
	return e.exprFactory.NewComprehension(
		e.nextMacroID(), iterRange, iterVar, accuVar, accuInit, condition, step, result)
}

// NewComprehensionTwoVar implements the ExprHelper interface method.
func (e *exprHelper) NewComprehensionTwoVar(
	iterRange ast.Expr,
	iterVar,
	iterVar2,
	accuVar string,
	accuInit,
	condition,
	step,
	result ast.Expr) ast.Expr {
	return e.exprFactory.NewComprehensionTwoVar(
		e.nextMacroID(), iterRange, iterVar, iterVar2, accuVar, accuInit, condition, step, result)
}

// NewIdent implements the ExprHelper interface method.
func (e *exprHelper) NewIdent(name string) ast.Expr {
	return e.exprFactory.NewIdent(e.nextMacroID(), name)
}

// NewAccuIdent implements the ExprHelper interface method.
func (e *exprHelper) NewAccuIdent() ast.Expr {
	return e.exprFactory.NewAccuIdent(e.nextMacroID())
}

// AccuIdentName implements the ExprHelper interface method.
func (e *exprHelper) AccuIdentName() string {
	return e.exprFactory.AccuIdentName()
}

// NewGlobalCall implements the ExprHelper interface method.
func (e *exprHelper) NewCall(function string, args ...ast.Expr) ast.Expr {
	return e.exprFactory.NewCall(e.nextMacroID(), function, args...)
}

// NewMemberCall implements the ExprHelper interface method.
func (e *exprHelper) NewMemberCall(function string, target ast.Expr, args ...ast.Expr) ast.Expr {
	return e.exprFactory.NewMemberCall(e.nextMacroID(), function, target, args...)
}

// NewPresenceTest implements the ExprHelper interface method.
func (e *exprHelper) NewPresenceTest(operand ast.Expr, field string) ast.Expr {
	return e.exprFactory.NewPresenceTest(e.nextMacroID(), operand, field)
}

// NewSelect implements the ExprHelper interface method.
func (e *exprHelper) NewSelect(operand ast.Expr, field string) ast.Expr {
	return e.exprFactory.NewSelect(e.nextMacroID(), operand, field)
}

// OffsetLocation implements the ExprHelper interface method.
func (e *exprHelper) OffsetLocation(exprID int64) common.Location {
	return e.parserHelper.sourceInfo.GetStartLocation(exprID)
}

// NewError associates an error message with a given expression id, populating the source offset location of the error if possible.
func (e *exprHelper) NewError(exprID int64, message string) *common.Error {
	return common.NewError(exprID, message, e.OffsetLocation(exprID))
}

var (
	// Thread-safe pool of ExprHelper values to minimize alloc overhead of ExprHelper creations.
	exprHelperPool = &sync.Pool{
		New: func() any {
			return &exprHelper{}
		},
	}
)
