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

package interpreter

import (
	"github.com/google/cel-go/common/operators"
	"github.com/google/cel-go/common/overloads"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
	structpb "google.golang.org/protobuf/types/known/structpb"
)

type astPruner struct {
	expr       *exprpb.Expr
	macroCalls map[int64]*exprpb.Expr
	state      EvalState
	nextExprID int64
}

// TODO Consider having a separate walk of the AST that finds common
// subexpressions. This can be called before or after constant folding to find
// common subexpressions.

// PruneAst prunes the given AST based on the given EvalState and generates a new AST.
// Given AST is copied on write and a new AST is returned.
// Couple of typical use cases this interface would be:
//
// A)
// 1) Evaluate expr with some unknowns,
// 2) If result is unknown:
//
//	a) PruneAst
//	b) Goto 1
//
// Functional call results which are known would be effectively cached across
// iterations.
//
// B)
// 1) Compile the expression (maybe via a service and maybe after checking a
//
//	compiled expression does not exists in local cache)
//
// 2) Prepare the environment and the interpreter. Activation might be empty.
// 3) Eval the expression. This might return unknown or error or a concrete
//
//	value.
//
// 4) PruneAst
// 4) Maybe cache the expression
// This is effectively constant folding the expression. How the environment is
// prepared in step 2 is flexible. For example, If the caller caches the
// compiled and constant folded expressions, but is not willing to constant
// fold(and thus cache results of) some external calls, then they can prepare
// the overloads accordingly.
func PruneAst(expr *exprpb.Expr, macroCalls map[int64]*exprpb.Expr, state EvalState) *exprpb.ParsedExpr {
	pruneState := NewEvalState()
	for _, id := range state.IDs() {
		v, _ := state.Value(id)
		pruneState.SetValue(id, v)
	}
	pruner := &astPruner{
		expr:       expr,
		macroCalls: macroCalls,
		state:      pruneState,
		nextExprID: getMaxID(expr)}
	newExpr, _ := pruner.maybePrune(expr)
	return &exprpb.ParsedExpr{
		Expr:       newExpr,
		SourceInfo: &exprpb.SourceInfo{MacroCalls: pruner.macroCalls},
	}
}

func (p *astPruner) createLiteral(id int64, val *exprpb.Constant) *exprpb.Expr {
	return &exprpb.Expr{
		Id: id,
		ExprKind: &exprpb.Expr_ConstExpr{
			ConstExpr: val,
		},
	}
}

func (p *astPruner) maybeCreateLiteral(id int64, val ref.Val) (*exprpb.Expr, bool) {
	switch v := val.(type) {
	case types.Bool:
		p.state.SetValue(id, val)
		return p.createLiteral(id,
			&exprpb.Constant{ConstantKind: &exprpb.Constant_BoolValue{BoolValue: bool(v)}}), true
	case types.Bytes:
		p.state.SetValue(id, val)
		return p.createLiteral(id,
			&exprpb.Constant{ConstantKind: &exprpb.Constant_BytesValue{BytesValue: []byte(v)}}), true
	case types.Double:
		p.state.SetValue(id, val)
		return p.createLiteral(id,
			&exprpb.Constant{ConstantKind: &exprpb.Constant_DoubleValue{DoubleValue: float64(v)}}), true
	case types.Duration:
		p.state.SetValue(id, val)
		durationString := string(v.ConvertToType(types.StringType).(types.String))
		return &exprpb.Expr{
			Id: id,
			ExprKind: &exprpb.Expr_CallExpr{
				CallExpr: &exprpb.Expr_Call{
					Function: overloads.TypeConvertDuration,
					Args: []*exprpb.Expr{
						p.createLiteral(p.nextID(),
							&exprpb.Constant{ConstantKind: &exprpb.Constant_StringValue{StringValue: durationString}}),
					},
				},
			},
		}, true
	case types.Int:
		p.state.SetValue(id, val)
		return p.createLiteral(id,
			&exprpb.Constant{ConstantKind: &exprpb.Constant_Int64Value{Int64Value: int64(v)}}), true
	case types.Uint:
		p.state.SetValue(id, val)
		return p.createLiteral(id,
			&exprpb.Constant{ConstantKind: &exprpb.Constant_Uint64Value{Uint64Value: uint64(v)}}), true
	case types.String:
		p.state.SetValue(id, val)
		return p.createLiteral(id,
			&exprpb.Constant{ConstantKind: &exprpb.Constant_StringValue{StringValue: string(v)}}), true
	case types.Null:
		p.state.SetValue(id, val)
		return p.createLiteral(id,
			&exprpb.Constant{ConstantKind: &exprpb.Constant_NullValue{NullValue: v.Value().(structpb.NullValue)}}), true
	}

	// Attempt to build a list literal.
	if list, isList := val.(traits.Lister); isList {
		sz := list.Size().(types.Int)
		elemExprs := make([]*exprpb.Expr, sz)
		for i := types.Int(0); i < sz; i++ {
			elem := list.Get(i)
			if types.IsUnknownOrError(elem) {
				return nil, false
			}
			elemExpr, ok := p.maybeCreateLiteral(p.nextID(), elem)
			if !ok {
				return nil, false
			}
			elemExprs[i] = elemExpr
		}
		p.state.SetValue(id, val)
		return &exprpb.Expr{
			Id: id,
			ExprKind: &exprpb.Expr_ListExpr{
				ListExpr: &exprpb.Expr_CreateList{
					Elements: elemExprs,
				},
			},
		}, true
	}

	// Create a map literal if possible.
	if mp, isMap := val.(traits.Mapper); isMap {
		it := mp.Iterator()
		entries := make([]*exprpb.Expr_CreateStruct_Entry, mp.Size().(types.Int))
		i := 0
		for it.HasNext() != types.False {
			key := it.Next()
			val := mp.Get(key)
			if types.IsUnknownOrError(key) || types.IsUnknownOrError(val) {
				return nil, false
			}
			keyExpr, ok := p.maybeCreateLiteral(p.nextID(), key)
			if !ok {
				return nil, false
			}
			valExpr, ok := p.maybeCreateLiteral(p.nextID(), val)
			if !ok {
				return nil, false
			}
			entry := &exprpb.Expr_CreateStruct_Entry{
				Id: p.nextID(),
				KeyKind: &exprpb.Expr_CreateStruct_Entry_MapKey{
					MapKey: keyExpr,
				},
				Value: valExpr,
			}
			entries[i] = entry
			i++
		}
		p.state.SetValue(id, val)
		return &exprpb.Expr{
			Id: id,
			ExprKind: &exprpb.Expr_StructExpr{
				StructExpr: &exprpb.Expr_CreateStruct{
					Entries: entries,
				},
			},
		}, true
	}

	// TODO(issues/377) To construct message literals, the type provider will need to support
	// the enumeration the fields for a given message.
	return nil, false
}

func (p *astPruner) maybePruneOptional(elem *exprpb.Expr) (*exprpb.Expr, bool) {
	elemVal, found := p.value(elem.GetId())
	if found && elemVal.Type() == types.OptionalType {
		opt := elemVal.(*types.Optional)
		if !opt.HasValue() {
			return nil, true
		}
		if newElem, pruned := p.maybeCreateLiteral(elem.GetId(), opt.GetValue()); pruned {
			return newElem, true
		}
	}
	return elem, false
}

func (p *astPruner) maybePruneIn(node *exprpb.Expr) (*exprpb.Expr, bool) {
	// elem in list
	call := node.GetCallExpr()
	val, exists := p.maybeValue(call.GetArgs()[1].GetId())
	if !exists {
		return nil, false
	}
	if sz, ok := val.(traits.Sizer); ok && sz.Size() == types.IntZero {
		return p.maybeCreateLiteral(node.GetId(), types.False)
	}
	return nil, false
}

func (p *astPruner) maybePruneLogicalNot(node *exprpb.Expr) (*exprpb.Expr, bool) {
	call := node.GetCallExpr()
	arg := call.GetArgs()[0]
	val, exists := p.maybeValue(arg.GetId())
	if !exists {
		return nil, false
	}
	if b, ok := val.(types.Bool); ok {
		return p.maybeCreateLiteral(node.GetId(), !b)
	}
	return nil, false
}

func (p *astPruner) maybePruneOr(node *exprpb.Expr) (*exprpb.Expr, bool) {
	call := node.GetCallExpr()
	// We know result is unknown, so we have at least one unknown arg
	// and if one side is a known value, we know we can ignore it.
	if v, exists := p.maybeValue(call.GetArgs()[0].GetId()); exists {
		if v == types.True {
			return p.maybeCreateLiteral(node.GetId(), types.True)
		}
		return call.GetArgs()[1], true
	}
	if v, exists := p.maybeValue(call.GetArgs()[1].GetId()); exists {
		if v == types.True {
			return p.maybeCreateLiteral(node.GetId(), types.True)
		}
		return call.GetArgs()[0], true
	}
	return nil, false
}

func (p *astPruner) maybePruneAnd(node *exprpb.Expr) (*exprpb.Expr, bool) {
	call := node.GetCallExpr()
	// We know result is unknown, so we have at least one unknown arg
	// and if one side is a known value, we know we can ignore it.
	if v, exists := p.maybeValue(call.GetArgs()[0].GetId()); exists {
		if v == types.False {
			return p.maybeCreateLiteral(node.GetId(), types.False)
		}
		return call.GetArgs()[1], true
	}
	if v, exists := p.maybeValue(call.GetArgs()[1].GetId()); exists {
		if v == types.False {
			return p.maybeCreateLiteral(node.GetId(), types.False)
		}
		return call.GetArgs()[0], true
	}
	return nil, false
}

func (p *astPruner) maybePruneConditional(node *exprpb.Expr) (*exprpb.Expr, bool) {
	call := node.GetCallExpr()
	cond, exists := p.maybeValue(call.GetArgs()[0].GetId())
	if !exists {
		return nil, false
	}
	if cond.Value().(bool) {
		return call.GetArgs()[1], true
	}
	return call.GetArgs()[2], true
}

func (p *astPruner) maybePruneFunction(node *exprpb.Expr) (*exprpb.Expr, bool) {
	if _, exists := p.value(node.GetId()); !exists {
		return nil, false
	}
	call := node.GetCallExpr()
	if call.Function == operators.LogicalOr {
		return p.maybePruneOr(node)
	}
	if call.Function == operators.LogicalAnd {
		return p.maybePruneAnd(node)
	}
	if call.Function == operators.Conditional {
		return p.maybePruneConditional(node)
	}
	if call.Function == operators.In {
		return p.maybePruneIn(node)
	}
	if call.Function == operators.LogicalNot {
		return p.maybePruneLogicalNot(node)
	}
	return nil, false
}

func (p *astPruner) maybePrune(node *exprpb.Expr) (*exprpb.Expr, bool) {
	return p.prune(node)
}

func (p *astPruner) prune(node *exprpb.Expr) (*exprpb.Expr, bool) {
	if node == nil {
		return node, false
	}
	val, valueExists := p.maybeValue(node.GetId())
	if valueExists {
		if newNode, ok := p.maybeCreateLiteral(node.GetId(), val); ok {
			delete(p.macroCalls, node.GetId())
			return newNode, true
		}
	}
	if macro, found := p.macroCalls[node.GetId()]; found {
		// prune the expression in terms of the macro call instead of the expanded form.
		if newMacro, pruned := p.prune(macro); pruned {
			p.macroCalls[node.GetId()] = newMacro
		}
	}

	// We have either an unknown/error value, or something we don't want to
	// transform, or expression was not evaluated. If possible, drill down
	// more.
	switch node.GetExprKind().(type) {
	case *exprpb.Expr_SelectExpr:
		if operand, pruned := p.maybePrune(node.GetSelectExpr().GetOperand()); pruned {
			return &exprpb.Expr{
				Id: node.GetId(),
				ExprKind: &exprpb.Expr_SelectExpr{
					SelectExpr: &exprpb.Expr_Select{
						Operand:  operand,
						Field:    node.GetSelectExpr().GetField(),
						TestOnly: node.GetSelectExpr().GetTestOnly(),
					},
				},
			}, true
		}
	case *exprpb.Expr_CallExpr:
		var prunedCall bool
		call := node.GetCallExpr()
		args := call.GetArgs()
		newArgs := make([]*exprpb.Expr, len(args))
		newCall := &exprpb.Expr_Call{
			Function: call.GetFunction(),
			Target:   call.GetTarget(),
			Args:     newArgs,
		}
		for i, arg := range args {
			newArgs[i] = arg
			if newArg, prunedArg := p.maybePrune(arg); prunedArg {
				prunedCall = true
				newArgs[i] = newArg
			}
		}
		if newTarget, prunedTarget := p.maybePrune(call.GetTarget()); prunedTarget {
			prunedCall = true
			newCall.Target = newTarget
		}
		newNode := &exprpb.Expr{
			Id: node.GetId(),
			ExprKind: &exprpb.Expr_CallExpr{
				CallExpr: newCall,
			},
		}
		if newExpr, pruned := p.maybePruneFunction(newNode); pruned {
			newExpr, _ = p.maybePrune(newExpr)
			return newExpr, true
		}
		if prunedCall {
			return newNode, true
		}
	case *exprpb.Expr_ListExpr:
		elems := node.GetListExpr().GetElements()
		optIndices := node.GetListExpr().GetOptionalIndices()
		optIndexMap := map[int32]bool{}
		for _, i := range optIndices {
			optIndexMap[i] = true
		}
		newOptIndexMap := make(map[int32]bool, len(optIndexMap))
		newElems := make([]*exprpb.Expr, 0, len(elems))
		var prunedList bool

		prunedIdx := 0
		for i, elem := range elems {
			_, isOpt := optIndexMap[int32(i)]
			if isOpt {
				newElem, pruned := p.maybePruneOptional(elem)
				if pruned {
					prunedList = true
					if newElem != nil {
						newElems = append(newElems, newElem)
						prunedIdx++
					}
					continue
				}
				newOptIndexMap[int32(prunedIdx)] = true
			}
			if newElem, prunedElem := p.maybePrune(elem); prunedElem {
				newElems = append(newElems, newElem)
				prunedList = true
			} else {
				newElems = append(newElems, elem)
			}
			prunedIdx++
		}
		optIndices = make([]int32, len(newOptIndexMap))
		idx := 0
		for i := range newOptIndexMap {
			optIndices[idx] = i
			idx++
		}
		if prunedList {
			return &exprpb.Expr{
				Id: node.GetId(),
				ExprKind: &exprpb.Expr_ListExpr{
					ListExpr: &exprpb.Expr_CreateList{
						Elements:        newElems,
						OptionalIndices: optIndices,
					},
				},
			}, true
		}
	case *exprpb.Expr_StructExpr:
		var prunedStruct bool
		entries := node.GetStructExpr().GetEntries()
		messageType := node.GetStructExpr().GetMessageName()
		newEntries := make([]*exprpb.Expr_CreateStruct_Entry, len(entries))
		for i, entry := range entries {
			newEntries[i] = entry
			newKey, prunedKey := p.maybePrune(entry.GetMapKey())
			newValue, prunedValue := p.maybePrune(entry.GetValue())
			if !prunedKey && !prunedValue {
				continue
			}
			prunedStruct = true
			newEntry := &exprpb.Expr_CreateStruct_Entry{
				Value: newValue,
			}
			if messageType != "" {
				newEntry.KeyKind = &exprpb.Expr_CreateStruct_Entry_FieldKey{
					FieldKey: entry.GetFieldKey(),
				}
			} else {
				newEntry.KeyKind = &exprpb.Expr_CreateStruct_Entry_MapKey{
					MapKey: newKey,
				}
			}
			newEntry.OptionalEntry = entry.GetOptionalEntry()
			newEntries[i] = newEntry
		}
		if prunedStruct {
			return &exprpb.Expr{
				Id: node.GetId(),
				ExprKind: &exprpb.Expr_StructExpr{
					StructExpr: &exprpb.Expr_CreateStruct{
						MessageName: messageType,
						Entries:     newEntries,
					},
				},
			}, true
		}
	}
	return node, false
}

func (p *astPruner) value(id int64) (ref.Val, bool) {
	val, found := p.state.Value(id)
	return val, (found && val != nil)
}

func (p *astPruner) maybeValue(id int64) (ref.Val, bool) {
	val, found := p.value(id)
	if !found || types.IsUnknownOrError(val) {
		return nil, false
	}
	return val, true
}

func (p *astPruner) nextID() int64 {
	next := p.nextExprID
	p.nextExprID++
	return next
}

type astVisitor struct {
	// visitEntry is called on every expr node, including those within a map/struct entry.
	visitExpr func(expr *exprpb.Expr)
	// visitEntry is called before entering the key, value of a map/struct entry.
	visitEntry func(entry *exprpb.Expr_CreateStruct_Entry)
}

func getMaxID(expr *exprpb.Expr) int64 {
	maxID := int64(1)
	visit(expr, maxIDVisitor(&maxID))
	return maxID
}

func maxIDVisitor(maxID *int64) astVisitor {
	return astVisitor{
		visitExpr: func(e *exprpb.Expr) {
			if e.GetId() >= *maxID {
				*maxID = e.GetId() + 1
			}
		},
		visitEntry: func(e *exprpb.Expr_CreateStruct_Entry) {
			if e.GetId() >= *maxID {
				*maxID = e.GetId() + 1
			}
		},
	}
}

func visit(expr *exprpb.Expr, visitor astVisitor) {
	exprs := []*exprpb.Expr{expr}
	for len(exprs) != 0 {
		e := exprs[0]
		visitor.visitExpr(e)
		exprs = exprs[1:]
		switch e.GetExprKind().(type) {
		case *exprpb.Expr_SelectExpr:
			exprs = append(exprs, e.GetSelectExpr().GetOperand())
		case *exprpb.Expr_CallExpr:
			call := e.GetCallExpr()
			if call.GetTarget() != nil {
				exprs = append(exprs, call.GetTarget())
			}
			exprs = append(exprs, call.GetArgs()...)
		case *exprpb.Expr_ComprehensionExpr:
			compre := e.GetComprehensionExpr()
			exprs = append(exprs,
				compre.GetIterRange(),
				compre.GetAccuInit(),
				compre.GetLoopCondition(),
				compre.GetLoopStep(),
				compre.GetResult())
		case *exprpb.Expr_ListExpr:
			list := e.GetListExpr()
			exprs = append(exprs, list.GetElements()...)
		case *exprpb.Expr_StructExpr:
			for _, entry := range e.GetStructExpr().GetEntries() {
				visitor.visitEntry(entry)
				if entry.GetMapKey() != nil {
					exprs = append(exprs, entry.GetMapKey())
				}
				exprs = append(exprs, entry.GetValue())
			}
		}
	}
}
