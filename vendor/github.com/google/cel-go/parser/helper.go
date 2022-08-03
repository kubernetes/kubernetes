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

	"github.com/antlr/antlr4/runtime/Go/antlr"
	"github.com/google/cel-go/common"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

type parserHelper struct {
	source     common.Source
	nextID     int64
	positions  map[int64]int32
	macroCalls map[int64]*exprpb.Expr
}

func newParserHelper(source common.Source) *parserHelper {
	return &parserHelper{
		source:     source,
		nextID:     1,
		positions:  make(map[int64]int32),
		macroCalls: make(map[int64]*exprpb.Expr),
	}
}

func (p *parserHelper) getSourceInfo() *exprpb.SourceInfo {
	return &exprpb.SourceInfo{
		Location:    p.source.Description(),
		Positions:   p.positions,
		LineOffsets: p.source.LineOffsets(),
		MacroCalls:  p.macroCalls}
}

func (p *parserHelper) newLiteral(ctx interface{}, value *exprpb.Constant) *exprpb.Expr {
	exprNode := p.newExpr(ctx)
	exprNode.ExprKind = &exprpb.Expr_ConstExpr{ConstExpr: value}
	return exprNode
}

func (p *parserHelper) newLiteralBool(ctx interface{}, value bool) *exprpb.Expr {
	return p.newLiteral(ctx,
		&exprpb.Constant{ConstantKind: &exprpb.Constant_BoolValue{BoolValue: value}})
}

func (p *parserHelper) newLiteralString(ctx interface{}, value string) *exprpb.Expr {
	return p.newLiteral(ctx,
		&exprpb.Constant{ConstantKind: &exprpb.Constant_StringValue{StringValue: value}})
}

func (p *parserHelper) newLiteralBytes(ctx interface{}, value []byte) *exprpb.Expr {
	return p.newLiteral(ctx,
		&exprpb.Constant{ConstantKind: &exprpb.Constant_BytesValue{BytesValue: value}})
}

func (p *parserHelper) newLiteralInt(ctx interface{}, value int64) *exprpb.Expr {
	return p.newLiteral(ctx,
		&exprpb.Constant{ConstantKind: &exprpb.Constant_Int64Value{Int64Value: value}})
}

func (p *parserHelper) newLiteralUint(ctx interface{}, value uint64) *exprpb.Expr {
	return p.newLiteral(ctx, &exprpb.Constant{ConstantKind: &exprpb.Constant_Uint64Value{Uint64Value: value}})
}

func (p *parserHelper) newLiteralDouble(ctx interface{}, value float64) *exprpb.Expr {
	return p.newLiteral(ctx,
		&exprpb.Constant{ConstantKind: &exprpb.Constant_DoubleValue{DoubleValue: value}})
}

func (p *parserHelper) newIdent(ctx interface{}, name string) *exprpb.Expr {
	exprNode := p.newExpr(ctx)
	exprNode.ExprKind = &exprpb.Expr_IdentExpr{IdentExpr: &exprpb.Expr_Ident{Name: name}}
	return exprNode
}

func (p *parserHelper) newSelect(ctx interface{}, operand *exprpb.Expr, field string) *exprpb.Expr {
	exprNode := p.newExpr(ctx)
	exprNode.ExprKind = &exprpb.Expr_SelectExpr{
		SelectExpr: &exprpb.Expr_Select{Operand: operand, Field: field}}
	return exprNode
}

func (p *parserHelper) newPresenceTest(ctx interface{}, operand *exprpb.Expr, field string) *exprpb.Expr {
	exprNode := p.newExpr(ctx)
	exprNode.ExprKind = &exprpb.Expr_SelectExpr{
		SelectExpr: &exprpb.Expr_Select{Operand: operand, Field: field, TestOnly: true}}
	return exprNode
}

func (p *parserHelper) newGlobalCall(ctx interface{}, function string, args ...*exprpb.Expr) *exprpb.Expr {
	exprNode := p.newExpr(ctx)
	exprNode.ExprKind = &exprpb.Expr_CallExpr{
		CallExpr: &exprpb.Expr_Call{Function: function, Args: args}}
	return exprNode
}

func (p *parserHelper) newReceiverCall(ctx interface{}, function string, target *exprpb.Expr, args ...*exprpb.Expr) *exprpb.Expr {
	exprNode := p.newExpr(ctx)
	exprNode.ExprKind = &exprpb.Expr_CallExpr{
		CallExpr: &exprpb.Expr_Call{Function: function, Target: target, Args: args}}
	return exprNode
}

func (p *parserHelper) newList(ctx interface{}, elements ...*exprpb.Expr) *exprpb.Expr {
	exprNode := p.newExpr(ctx)
	exprNode.ExprKind = &exprpb.Expr_ListExpr{
		ListExpr: &exprpb.Expr_CreateList{Elements: elements}}
	return exprNode
}

func (p *parserHelper) newMap(ctx interface{}, entries ...*exprpb.Expr_CreateStruct_Entry) *exprpb.Expr {
	exprNode := p.newExpr(ctx)
	exprNode.ExprKind = &exprpb.Expr_StructExpr{
		StructExpr: &exprpb.Expr_CreateStruct{Entries: entries}}
	return exprNode
}

func (p *parserHelper) newMapEntry(entryID int64, key *exprpb.Expr, value *exprpb.Expr) *exprpb.Expr_CreateStruct_Entry {
	return &exprpb.Expr_CreateStruct_Entry{
		Id:      entryID,
		KeyKind: &exprpb.Expr_CreateStruct_Entry_MapKey{MapKey: key},
		Value:   value}
}

func (p *parserHelper) newObject(ctx interface{},
	typeName string,
	entries ...*exprpb.Expr_CreateStruct_Entry) *exprpb.Expr {
	exprNode := p.newExpr(ctx)
	exprNode.ExprKind = &exprpb.Expr_StructExpr{
		StructExpr: &exprpb.Expr_CreateStruct{
			MessageName: typeName,
			Entries:     entries}}
	return exprNode
}

func (p *parserHelper) newObjectField(fieldID int64, field string, value *exprpb.Expr) *exprpb.Expr_CreateStruct_Entry {
	return &exprpb.Expr_CreateStruct_Entry{
		Id:      fieldID,
		KeyKind: &exprpb.Expr_CreateStruct_Entry_FieldKey{FieldKey: field},
		Value:   value}
}

func (p *parserHelper) newComprehension(ctx interface{}, iterVar string,
	iterRange *exprpb.Expr,
	accuVar string,
	accuInit *exprpb.Expr,
	condition *exprpb.Expr,
	step *exprpb.Expr,
	result *exprpb.Expr) *exprpb.Expr {
	exprNode := p.newExpr(ctx)
	exprNode.ExprKind = &exprpb.Expr_ComprehensionExpr{
		ComprehensionExpr: &exprpb.Expr_Comprehension{
			AccuVar:       accuVar,
			AccuInit:      accuInit,
			IterVar:       iterVar,
			IterRange:     iterRange,
			LoopCondition: condition,
			LoopStep:      step,
			Result:        result}}
	return exprNode
}

func (p *parserHelper) newExpr(ctx interface{}) *exprpb.Expr {
	id, isID := ctx.(int64)
	if isID {
		return &exprpb.Expr{Id: id}
	}
	return &exprpb.Expr{Id: p.id(ctx)}
}

func (p *parserHelper) id(ctx interface{}) int64 {
	var location common.Location
	switch ctx.(type) {
	case antlr.ParserRuleContext:
		token := (ctx.(antlr.ParserRuleContext)).GetStart()
		location = p.source.NewLocation(token.GetLine(), token.GetColumn())
	case antlr.Token:
		token := ctx.(antlr.Token)
		location = p.source.NewLocation(token.GetLine(), token.GetColumn())
	case common.Location:
		location = ctx.(common.Location)
	default:
		// This should only happen if the ctx is nil
		return -1
	}
	id := p.nextID
	p.positions[id], _ = p.source.LocationOffset(location)
	p.nextID++
	return id
}

func (p *parserHelper) getLocation(id int64) common.Location {
	characterOffset := p.positions[id]
	location, _ := p.source.OffsetLocation(characterOffset)
	return location
}

// buildMacroCallArg iterates the expression and returns a new expression
// where all macros have been replaced by their IDs in MacroCalls
func (p *parserHelper) buildMacroCallArg(expr *exprpb.Expr) *exprpb.Expr {
	if _, found := p.macroCalls[expr.GetId()]; found {
		return &exprpb.Expr{Id: expr.GetId()}
	}

	switch expr.ExprKind.(type) {
	case *exprpb.Expr_CallExpr:
		// Iterate the AST from `expr` recursively looking for macros. Because we are at most
		// starting from the top level macro, this recursion is bounded by the size of the AST. This
		// means that the depth check on the AST during parsing will catch recursion overflows
		// before we get to here.
		macroTarget := expr.GetCallExpr().GetTarget()
		if macroTarget != nil {
			macroTarget = p.buildMacroCallArg(macroTarget)
		}
		macroArgs := make([]*exprpb.Expr, len(expr.GetCallExpr().GetArgs()))
		for index, arg := range expr.GetCallExpr().GetArgs() {
			macroArgs[index] = p.buildMacroCallArg(arg)
		}
		return &exprpb.Expr{
			Id: expr.GetId(),
			ExprKind: &exprpb.Expr_CallExpr{
				CallExpr: &exprpb.Expr_Call{
					Target:   macroTarget,
					Function: expr.GetCallExpr().GetFunction(),
					Args:     macroArgs,
				},
			},
		}
	case *exprpb.Expr_ListExpr:
		listExpr := expr.GetListExpr()
		macroListArgs := make([]*exprpb.Expr, len(listExpr.GetElements()))
		for i, elem := range listExpr.GetElements() {
			macroListArgs[i] = p.buildMacroCallArg(elem)
		}
		return &exprpb.Expr{
			Id: expr.GetId(),
			ExprKind: &exprpb.Expr_ListExpr{
				ListExpr: &exprpb.Expr_CreateList{
					Elements: macroListArgs,
				},
			},
		}
	}

	return expr
}

// addMacroCall adds the macro the the MacroCalls map in source info. If a macro has args/subargs/target
// that are macros, their ID will be stored instead for later self-lookups.
func (p *parserHelper) addMacroCall(exprID int64, function string, target *exprpb.Expr, args ...*exprpb.Expr) {
	macroTarget := target
	if target != nil {
		if _, found := p.macroCalls[target.GetId()]; found {
			macroTarget = &exprpb.Expr{Id: target.GetId()}
		} else {
			macroTarget = p.buildMacroCallArg(target)
		}
	}

	macroArgs := make([]*exprpb.Expr, len(args))
	for index, arg := range args {
		macroArgs[index] = p.buildMacroCallArg(arg)
	}

	p.macroCalls[exprID] = &exprpb.Expr{
		ExprKind: &exprpb.Expr_CallExpr{
			CallExpr: &exprpb.Expr_Call{
				Target:   macroTarget,
				Function: function,
				Args:     macroArgs,
			},
		},
	}
}

// balancer performs tree balancing on operators whose arguments are of equal precedence.
//
// The purpose of the balancer is to ensure a compact serialization format for the logical &&, ||
// operators which have a tendency to create long DAGs which are skewed in one direction. Since the
// operators are commutative re-ordering the terms *must not* affect the evaluation result.
//
// Re-balancing the terms is a safe, if somewhat controversial choice. A better solution would be
// to make these functions variadic and update both the checker and interpreter to understand this;
// however, this is a more complex change.
//
// TODO: Consider replacing tree-balancing with variadic logical &&, || within the parser, checker,
// and interpreter.
type balancer struct {
	helper   *parserHelper
	function string
	terms    []*exprpb.Expr
	ops      []int64
}

// newBalancer creates a balancer instance bound to a specific function and its first term.
func newBalancer(h *parserHelper, function string, term *exprpb.Expr) *balancer {
	return &balancer{
		helper:   h,
		function: function,
		terms:    []*exprpb.Expr{term},
		ops:      []int64{},
	}
}

// addTerm adds an operation identifier and term to the set of terms to be balanced.
func (b *balancer) addTerm(op int64, term *exprpb.Expr) {
	b.terms = append(b.terms, term)
	b.ops = append(b.ops, op)
}

// balance creates a balanced tree from the sub-terms and returns the final Expr value.
func (b *balancer) balance() *exprpb.Expr {
	if len(b.terms) == 1 {
		return b.terms[0]
	}
	return b.balancedTree(0, len(b.ops)-1)
}

// balancedTree recursively balances the terms provided to a commutative operator.
func (b *balancer) balancedTree(lo, hi int) *exprpb.Expr {
	mid := (lo + hi + 1) / 2

	var left *exprpb.Expr
	if mid == lo {
		left = b.terms[mid]
	} else {
		left = b.balancedTree(lo, mid-1)
	}

	var right *exprpb.Expr
	if mid == hi {
		right = b.terms[mid+1]
	} else {
		right = b.balancedTree(mid+1, hi)
	}
	return b.helper.newGlobalCall(b.ops[mid], b.function, left, right)
}

type exprHelper struct {
	*parserHelper
	id int64
}

func (e *exprHelper) nextMacroID() int64 {
	return e.parserHelper.id(e.parserHelper.getLocation(e.id))
}

// LiteralBool implements the ExprHelper interface method.
func (e *exprHelper) LiteralBool(value bool) *exprpb.Expr {
	return e.parserHelper.newLiteralBool(e.nextMacroID(), value)
}

// LiteralBytes implements the ExprHelper interface method.
func (e *exprHelper) LiteralBytes(value []byte) *exprpb.Expr {
	return e.parserHelper.newLiteralBytes(e.nextMacroID(), value)
}

// LiteralDouble implements the ExprHelper interface method.
func (e *exprHelper) LiteralDouble(value float64) *exprpb.Expr {
	return e.parserHelper.newLiteralDouble(e.nextMacroID(), value)
}

// LiteralInt implements the ExprHelper interface method.
func (e *exprHelper) LiteralInt(value int64) *exprpb.Expr {
	return e.parserHelper.newLiteralInt(e.nextMacroID(), value)
}

// LiteralString implements the ExprHelper interface method.
func (e *exprHelper) LiteralString(value string) *exprpb.Expr {
	return e.parserHelper.newLiteralString(e.nextMacroID(), value)
}

// LiteralUint implements the ExprHelper interface method.
func (e *exprHelper) LiteralUint(value uint64) *exprpb.Expr {
	return e.parserHelper.newLiteralUint(e.nextMacroID(), value)
}

// NewList implements the ExprHelper interface method.
func (e *exprHelper) NewList(elems ...*exprpb.Expr) *exprpb.Expr {
	return e.parserHelper.newList(e.nextMacroID(), elems...)
}

// NewMap implements the ExprHelper interface method.
func (e *exprHelper) NewMap(entries ...*exprpb.Expr_CreateStruct_Entry) *exprpb.Expr {
	return e.parserHelper.newMap(e.nextMacroID(), entries...)
}

// NewMapEntry implements the ExprHelper interface method.
func (e *exprHelper) NewMapEntry(key *exprpb.Expr,
	val *exprpb.Expr) *exprpb.Expr_CreateStruct_Entry {
	return e.parserHelper.newMapEntry(e.nextMacroID(), key, val)
}

// NewObject implements the ExprHelper interface method.
func (e *exprHelper) NewObject(typeName string,
	fieldInits ...*exprpb.Expr_CreateStruct_Entry) *exprpb.Expr {
	return e.parserHelper.newObject(e.nextMacroID(), typeName, fieldInits...)
}

// NewObjectFieldInit implements the ExprHelper interface method.
func (e *exprHelper) NewObjectFieldInit(field string,
	init *exprpb.Expr) *exprpb.Expr_CreateStruct_Entry {
	return e.parserHelper.newObjectField(e.nextMacroID(), field, init)
}

// Fold implements the ExprHelper interface method.
func (e *exprHelper) Fold(iterVar string,
	iterRange *exprpb.Expr,
	accuVar string,
	accuInit *exprpb.Expr,
	condition *exprpb.Expr,
	step *exprpb.Expr,
	result *exprpb.Expr) *exprpb.Expr {
	return e.parserHelper.newComprehension(
		e.nextMacroID(), iterVar, iterRange, accuVar, accuInit, condition, step, result)
}

// Ident implements the ExprHelper interface method.
func (e *exprHelper) Ident(name string) *exprpb.Expr {
	return e.parserHelper.newIdent(e.nextMacroID(), name)
}

// AccuIdent implements the ExprHelper interface method.
func (e *exprHelper) AccuIdent() *exprpb.Expr {
	return e.parserHelper.newIdent(e.nextMacroID(), AccumulatorName)
}

// GlobalCall implements the ExprHelper interface method.
func (e *exprHelper) GlobalCall(function string, args ...*exprpb.Expr) *exprpb.Expr {
	return e.parserHelper.newGlobalCall(e.nextMacroID(), function, args...)
}

// ReceiverCall implements the ExprHelper interface method.
func (e *exprHelper) ReceiverCall(function string,
	target *exprpb.Expr, args ...*exprpb.Expr) *exprpb.Expr {
	return e.parserHelper.newReceiverCall(e.nextMacroID(), function, target, args...)
}

// PresenceTest implements the ExprHelper interface method.
func (e *exprHelper) PresenceTest(operand *exprpb.Expr, field string) *exprpb.Expr {
	return e.parserHelper.newPresenceTest(e.nextMacroID(), operand, field)
}

// Select implements the ExprHelper interface method.
func (e *exprHelper) Select(operand *exprpb.Expr, field string) *exprpb.Expr {
	return e.parserHelper.newSelect(e.nextMacroID(), operand, field)
}

// OffsetLocation implements the ExprHelper interface method.
func (e *exprHelper) OffsetLocation(exprID int64) common.Location {
	offset := e.parserHelper.positions[exprID]
	location, _ := e.parserHelper.source.OffsetLocation(offset)
	return location
}

var (
	// Thread-safe pool of ExprHelper values to minimize alloc overhead of ExprHelper creations.
	exprHelperPool = &sync.Pool{
		New: func() interface{} {
			return &exprHelper{}
		},
	}
)
