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
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
	structpb "google.golang.org/protobuf/types/known/structpb"
)

type astPruner struct {
	expr       *exprpb.Expr
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
//   a) PruneAst
//   b) Goto 1
// Functional call results which are known would be effectively cached across
// iterations.
//
// B)
// 1) Compile the expression (maybe via a service and maybe after checking a
//    compiled expression does not exists in local cache)
// 2) Prepare the environment and the interpreter. Activation might be empty.
// 3) Eval the expression. This might return unknown or error or a concrete
//    value.
// 4) PruneAst
// 4) Maybe cache the expression
// This is effectively constant folding the expression. How the environment is
// prepared in step 2 is flexible. For example, If the caller caches the
// compiled and constant folded expressions, but is not willing to constant
// fold(and thus cache results of) some external calls, then they can prepare
// the overloads accordingly.
func PruneAst(expr *exprpb.Expr, state EvalState) *exprpb.Expr {
	pruner := &astPruner{
		expr:       expr,
		state:      state,
		nextExprID: 1}
	newExpr, _ := pruner.prune(expr)
	return newExpr
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
	switch val.Type() {
	case types.BoolType:
		return p.createLiteral(id,
			&exprpb.Constant{ConstantKind: &exprpb.Constant_BoolValue{BoolValue: val.Value().(bool)}}), true
	case types.IntType:
		return p.createLiteral(id,
			&exprpb.Constant{ConstantKind: &exprpb.Constant_Int64Value{Int64Value: val.Value().(int64)}}), true
	case types.UintType:
		return p.createLiteral(id,
			&exprpb.Constant{ConstantKind: &exprpb.Constant_Uint64Value{Uint64Value: val.Value().(uint64)}}), true
	case types.StringType:
		return p.createLiteral(id,
			&exprpb.Constant{ConstantKind: &exprpb.Constant_StringValue{StringValue: val.Value().(string)}}), true
	case types.DoubleType:
		return p.createLiteral(id,
			&exprpb.Constant{ConstantKind: &exprpb.Constant_DoubleValue{DoubleValue: val.Value().(float64)}}), true
	case types.BytesType:
		return p.createLiteral(id,
			&exprpb.Constant{ConstantKind: &exprpb.Constant_BytesValue{BytesValue: val.Value().([]byte)}}), true
	case types.NullType:
		return p.createLiteral(id,
			&exprpb.Constant{ConstantKind: &exprpb.Constant_NullValue{NullValue: val.Value().(structpb.NullValue)}}), true
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

func (p *astPruner) maybePruneAndOr(node *exprpb.Expr) (*exprpb.Expr, bool) {
	if !p.existsWithUnknownValue(node.GetId()) {
		return nil, false
	}

	call := node.GetCallExpr()
	// We know result is unknown, so we have at least one unknown arg
	// and if one side is a known value, we know we can ignore it.
	if p.existsWithKnownValue(call.Args[0].GetId()) {
		return call.Args[1], true
	}
	if p.existsWithKnownValue(call.Args[1].GetId()) {
		return call.Args[0], true
	}
	return nil, false
}

func (p *astPruner) maybePruneConditional(node *exprpb.Expr) (*exprpb.Expr, bool) {
	if !p.existsWithUnknownValue(node.GetId()) {
		return nil, false
	}

	call := node.GetCallExpr()
	condVal, condValueExists := p.value(call.Args[0].GetId())
	if !condValueExists || types.IsUnknownOrError(condVal) {
		return nil, false
	}

	if condVal.Value().(bool) {
		return call.Args[1], true
	}
	return call.Args[2], true
}

func (p *astPruner) maybePruneFunction(node *exprpb.Expr) (*exprpb.Expr, bool) {
	call := node.GetCallExpr()
	if call.Function == operators.LogicalOr || call.Function == operators.LogicalAnd {
		return p.maybePruneAndOr(node)
	}
	if call.Function == operators.Conditional {
		return p.maybePruneConditional(node)
	}

	return nil, false
}

func (p *astPruner) prune(node *exprpb.Expr) (*exprpb.Expr, bool) {
	if node == nil {
		return node, false
	}
	val, valueExists := p.value(node.GetId())
	if valueExists && !types.IsUnknownOrError(val) {
		if newNode, ok := p.maybeCreateLiteral(node.GetId(), val); ok {
			return newNode, true
		}
	}

	// We have either an unknown/error value, or something we dont want to
	// transform, or expression was not evaluated. If possible, drill down
	// more.

	switch node.ExprKind.(type) {
	case *exprpb.Expr_SelectExpr:
		if operand, pruned := p.prune(node.GetSelectExpr().Operand); pruned {
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
		if newExpr, pruned := p.maybePruneFunction(node); pruned {
			newExpr, _ = p.prune(newExpr)
			return newExpr, true
		}
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
			if newArg, prunedArg := p.prune(arg); prunedArg {
				prunedCall = true
				newArgs[i] = newArg
			}
		}
		if newTarget, prunedTarget := p.prune(call.GetTarget()); prunedTarget {
			prunedCall = true
			newCall.Target = newTarget
		}
		if prunedCall {
			return &exprpb.Expr{
				Id: node.GetId(),
				ExprKind: &exprpb.Expr_CallExpr{
					CallExpr: newCall,
				},
			}, true
		}
	case *exprpb.Expr_ListExpr:
		elems := node.GetListExpr().GetElements()
		newElems := make([]*exprpb.Expr, len(elems))
		var prunedList bool
		for i, elem := range elems {
			newElems[i] = elem
			if newElem, prunedElem := p.prune(elem); prunedElem {
				newElems[i] = newElem
				prunedList = true
			}
		}
		if prunedList {
			return &exprpb.Expr{
				Id: node.GetId(),
				ExprKind: &exprpb.Expr_ListExpr{
					ListExpr: &exprpb.Expr_CreateList{
						Elements: newElems,
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
			newKey, prunedKey := p.prune(entry.GetMapKey())
			newValue, prunedValue := p.prune(entry.GetValue())
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
	case *exprpb.Expr_ComprehensionExpr:
		compre := node.GetComprehensionExpr()
		// Only the range of the comprehension is pruned since the state tracking only records
		// the last iteration of the comprehension and not each step in the evaluation which
		// means that the any residuals computed in between might be inaccurate.
		if newRange, pruned := p.prune(compre.GetIterRange()); pruned {
			return &exprpb.Expr{
				Id: node.GetId(),
				ExprKind: &exprpb.Expr_ComprehensionExpr{
					ComprehensionExpr: &exprpb.Expr_Comprehension{
						IterVar:       compre.GetIterVar(),
						IterRange:     newRange,
						AccuVar:       compre.GetAccuVar(),
						AccuInit:      compre.GetAccuInit(),
						LoopCondition: compre.GetLoopCondition(),
						LoopStep:      compre.GetLoopStep(),
						Result:        compre.GetResult(),
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

func (p *astPruner) existsWithUnknownValue(id int64) bool {
	val, valueExists := p.value(id)
	return valueExists && types.IsUnknown(val)
}

func (p *astPruner) existsWithKnownValue(id int64) bool {
	val, valueExists := p.value(id)
	return valueExists && !types.IsUnknown(val)
}

func (p *astPruner) nextID() int64 {
	for {
		_, found := p.state.Value(p.nextExprID)
		if !found {
			next := p.nextExprID
			p.nextExprID++
			return next
		}
		p.nextExprID++
	}
}
