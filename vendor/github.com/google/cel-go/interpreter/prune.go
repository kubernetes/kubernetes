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
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/operators"
	"github.com/google/cel-go/common/overloads"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
)

type astPruner struct {
	ast.ExprFactory
	expr       ast.Expr
	macroCalls map[int64]ast.Expr
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
func PruneAst(expr ast.Expr, macroCalls map[int64]ast.Expr, state EvalState) *ast.AST {
	pruneState := NewEvalState()
	for _, id := range state.IDs() {
		v, _ := state.Value(id)
		pruneState.SetValue(id, v)
	}
	pruner := &astPruner{
		ExprFactory: ast.NewExprFactory(),
		expr:        expr,
		macroCalls:  macroCalls,
		state:       pruneState,
		nextExprID:  getMaxID(expr)}
	newExpr, _ := pruner.maybePrune(expr)
	newInfo := ast.NewSourceInfo(nil)
	for id, call := range pruner.macroCalls {
		newInfo.SetMacroCall(id, call)
	}
	return ast.NewAST(newExpr, newInfo)
}

func (p *astPruner) maybeCreateLiteral(id int64, val ref.Val) (ast.Expr, bool) {
	switch v := val.(type) {
	case types.Bool, types.Bytes, types.Double, types.Int, types.Null, types.String, types.Uint:
		p.state.SetValue(id, val)
		return p.NewLiteral(id, val), true
	case types.Duration:
		p.state.SetValue(id, val)
		durationString := v.ConvertToType(types.StringType).(types.String)
		return p.NewCall(id, overloads.TypeConvertDuration, p.NewLiteral(p.nextID(), durationString)), true
	case types.Timestamp:
		timestampString := v.ConvertToType(types.StringType).(types.String)
		return p.NewCall(id, overloads.TypeConvertTimestamp, p.NewLiteral(p.nextID(), timestampString)), true
	}

	// Attempt to build a list literal.
	if list, isList := val.(traits.Lister); isList {
		sz := list.Size().(types.Int)
		elemExprs := make([]ast.Expr, sz)
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
		return p.NewList(id, elemExprs, []int32{}), true
	}

	// Create a map literal if possible.
	if mp, isMap := val.(traits.Mapper); isMap {
		it := mp.Iterator()
		entries := make([]ast.EntryExpr, mp.Size().(types.Int))
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
			entry := p.NewMapEntry(p.nextID(), keyExpr, valExpr, false)
			entries[i] = entry
			i++
		}
		p.state.SetValue(id, val)
		return p.NewMap(id, entries), true
	}

	// TODO(issues/377) To construct message literals, the type provider will need to support
	// the enumeration the fields for a given message.
	return nil, false
}

func (p *astPruner) maybePruneOptional(elem ast.Expr) (ast.Expr, bool) {
	elemVal, found := p.value(elem.ID())
	if found && elemVal.Type() == types.OptionalType {
		opt := elemVal.(*types.Optional)
		if !opt.HasValue() {
			return nil, true
		}
		if newElem, pruned := p.maybeCreateLiteral(elem.ID(), opt.GetValue()); pruned {
			return newElem, true
		}
	}
	return elem, false
}

func (p *astPruner) maybePruneIn(node ast.Expr) (ast.Expr, bool) {
	// elem in list
	call := node.AsCall()
	val, exists := p.maybeValue(call.Args()[1].ID())
	if !exists {
		return nil, false
	}
	if sz, ok := val.(traits.Sizer); ok && sz.Size() == types.IntZero {
		return p.maybeCreateLiteral(node.ID(), types.False)
	}
	return nil, false
}

func (p *astPruner) maybePruneLogicalNot(node ast.Expr) (ast.Expr, bool) {
	call := node.AsCall()
	arg := call.Args()[0]
	val, exists := p.maybeValue(arg.ID())
	if !exists {
		return nil, false
	}
	if b, ok := val.(types.Bool); ok {
		return p.maybeCreateLiteral(node.ID(), !b)
	}
	return nil, false
}

func (p *astPruner) maybePruneOr(node ast.Expr) (ast.Expr, bool) {
	call := node.AsCall()
	// We know result is unknown, so we have at least one unknown arg
	// and if one side is a known value, we know we can ignore it.
	if v, exists := p.maybeValue(call.Args()[0].ID()); exists {
		if v == types.True {
			return p.maybeCreateLiteral(node.ID(), types.True)
		}
		return call.Args()[1], true
	}
	if v, exists := p.maybeValue(call.Args()[1].ID()); exists {
		if v == types.True {
			return p.maybeCreateLiteral(node.ID(), types.True)
		}
		return call.Args()[0], true
	}
	return nil, false
}

func (p *astPruner) maybePruneAnd(node ast.Expr) (ast.Expr, bool) {
	call := node.AsCall()
	// We know result is unknown, so we have at least one unknown arg
	// and if one side is a known value, we know we can ignore it.
	if v, exists := p.maybeValue(call.Args()[0].ID()); exists {
		if v == types.False {
			return p.maybeCreateLiteral(node.ID(), types.False)
		}
		return call.Args()[1], true
	}
	if v, exists := p.maybeValue(call.Args()[1].ID()); exists {
		if v == types.False {
			return p.maybeCreateLiteral(node.ID(), types.False)
		}
		return call.Args()[0], true
	}
	return nil, false
}

func (p *astPruner) maybePruneConditional(node ast.Expr) (ast.Expr, bool) {
	call := node.AsCall()
	cond, exists := p.maybeValue(call.Args()[0].ID())
	if !exists {
		return nil, false
	}
	if cond.Value().(bool) {
		return call.Args()[1], true
	}
	return call.Args()[2], true
}

func (p *astPruner) maybePruneFunction(node ast.Expr) (ast.Expr, bool) {
	if _, exists := p.value(node.ID()); !exists {
		return nil, false
	}
	call := node.AsCall()
	if call.FunctionName() == operators.LogicalOr {
		return p.maybePruneOr(node)
	}
	if call.FunctionName() == operators.LogicalAnd {
		return p.maybePruneAnd(node)
	}
	if call.FunctionName() == operators.Conditional {
		return p.maybePruneConditional(node)
	}
	if call.FunctionName() == operators.In {
		return p.maybePruneIn(node)
	}
	if call.FunctionName() == operators.LogicalNot {
		return p.maybePruneLogicalNot(node)
	}
	return nil, false
}

func (p *astPruner) maybePrune(node ast.Expr) (ast.Expr, bool) {
	return p.prune(node)
}

func (p *astPruner) prune(node ast.Expr) (ast.Expr, bool) {
	if node == nil {
		return node, false
	}
	val, valueExists := p.maybeValue(node.ID())
	if valueExists {
		if newNode, ok := p.maybeCreateLiteral(node.ID(), val); ok {
			delete(p.macroCalls, node.ID())
			return newNode, true
		}
	}
	if macro, found := p.macroCalls[node.ID()]; found {
		// Ensure that intermediate values for the comprehension are cleared during pruning
		if node.Kind() == ast.ComprehensionKind {
			compre := node.AsComprehension()
			visit(macro, clearIterVarVisitor(compre.IterVar(), p.state))
		}
		// prune the expression in terms of the macro call instead of the expanded form.
		if newMacro, pruned := p.prune(macro); pruned {
			p.macroCalls[node.ID()] = newMacro
		}
	}

	// We have either an unknown/error value, or something we don't want to
	// transform, or expression was not evaluated. If possible, drill down
	// more.
	switch node.Kind() {
	case ast.SelectKind:
		sel := node.AsSelect()
		if operand, isPruned := p.maybePrune(sel.Operand()); isPruned {
			if sel.IsTestOnly() {
				return p.NewPresenceTest(node.ID(), operand, sel.FieldName()), true
			}
			return p.NewSelect(node.ID(), operand, sel.FieldName()), true
		}
	case ast.CallKind:
		argsPruned := false
		call := node.AsCall()
		args := call.Args()
		newArgs := make([]ast.Expr, len(args))
		for i, a := range args {
			newArgs[i] = a
			if arg, isPruned := p.maybePrune(a); isPruned {
				argsPruned = true
				newArgs[i] = arg
			}
		}
		if !call.IsMemberFunction() {
			newCall := p.NewCall(node.ID(), call.FunctionName(), newArgs...)
			if prunedCall, isPruned := p.maybePruneFunction(newCall); isPruned {
				return prunedCall, true
			}
			return newCall, argsPruned
		}
		newTarget := call.Target()
		targetPruned := false
		if prunedTarget, isPruned := p.maybePrune(call.Target()); isPruned {
			targetPruned = true
			newTarget = prunedTarget
		}
		newCall := p.NewMemberCall(node.ID(), call.FunctionName(), newTarget, newArgs...)
		if prunedCall, isPruned := p.maybePruneFunction(newCall); isPruned {
			return prunedCall, true
		}
		return newCall, targetPruned || argsPruned
	case ast.ListKind:
		l := node.AsList()
		elems := l.Elements()
		optIndices := l.OptionalIndices()
		optIndexMap := map[int32]bool{}
		for _, i := range optIndices {
			optIndexMap[i] = true
		}
		newOptIndexMap := make(map[int32]bool, len(optIndexMap))
		newElems := make([]ast.Expr, 0, len(elems))
		var listPruned bool
		prunedIdx := 0
		for i, elem := range elems {
			_, isOpt := optIndexMap[int32(i)]
			if isOpt {
				newElem, pruned := p.maybePruneOptional(elem)
				if pruned {
					listPruned = true
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
				listPruned = true
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
		if listPruned {
			return p.NewList(node.ID(), newElems, optIndices), true
		}
	case ast.MapKind:
		var mapPruned bool
		m := node.AsMap()
		entries := m.Entries()
		newEntries := make([]ast.EntryExpr, len(entries))
		for i, entry := range entries {
			newEntries[i] = entry
			e := entry.AsMapEntry()
			newKey, keyPruned := p.maybePrune(e.Key())
			newValue, valuePruned := p.maybePrune(e.Value())
			if !keyPruned && !valuePruned {
				continue
			}
			mapPruned = true
			newEntry := p.NewMapEntry(entry.ID(), newKey, newValue, e.IsOptional())
			newEntries[i] = newEntry
		}
		if mapPruned {
			return p.NewMap(node.ID(), newEntries), true
		}
	case ast.StructKind:
		var structPruned bool
		obj := node.AsStruct()
		fields := obj.Fields()
		newFields := make([]ast.EntryExpr, len(fields))
		for i, field := range fields {
			newFields[i] = field
			f := field.AsStructField()
			newValue, prunedValue := p.maybePrune(f.Value())
			if !prunedValue {
				continue
			}
			structPruned = true
			newEntry := p.NewStructField(field.ID(), f.Name(), newValue, f.IsOptional())
			newFields[i] = newEntry
		}
		if structPruned {
			return p.NewStruct(node.ID(), obj.TypeName(), newFields), true
		}
	case ast.ComprehensionKind:
		compre := node.AsComprehension()
		// Only the range of the comprehension is pruned since the state tracking only records
		// the last iteration of the comprehension and not each step in the evaluation which
		// means that the any residuals computed in between might be inaccurate.
		if newRange, pruned := p.maybePrune(compre.IterRange()); pruned {
			return p.NewComprehension(
				node.ID(),
				newRange,
				compre.IterVar(),
				compre.AccuVar(),
				compre.AccuInit(),
				compre.LoopCondition(),
				compre.LoopStep(),
				compre.Result(),
			), true
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
	visitExpr func(expr ast.Expr)
	// visitEntry is called before entering the key, value of a map/struct entry.
	visitEntry func(entry ast.EntryExpr)
}

func getMaxID(expr ast.Expr) int64 {
	maxID := int64(1)
	visit(expr, maxIDVisitor(&maxID))
	return maxID
}

func clearIterVarVisitor(varName string, state EvalState) astVisitor {
	return astVisitor{
		visitExpr: func(e ast.Expr) {
			if e.Kind() == ast.IdentKind && e.AsIdent() == varName {
				state.SetValue(e.ID(), nil)
			}
		},
	}
}

func maxIDVisitor(maxID *int64) astVisitor {
	return astVisitor{
		visitExpr: func(e ast.Expr) {
			if e.ID() >= *maxID {
				*maxID = e.ID() + 1
			}
		},
		visitEntry: func(e ast.EntryExpr) {
			if e.ID() >= *maxID {
				*maxID = e.ID() + 1
			}
		},
	}
}

func visit(expr ast.Expr, visitor astVisitor) {
	exprs := []ast.Expr{expr}
	for len(exprs) != 0 {
		e := exprs[0]
		if visitor.visitExpr != nil {
			visitor.visitExpr(e)
		}
		exprs = exprs[1:]
		switch e.Kind() {
		case ast.SelectKind:
			exprs = append(exprs, e.AsSelect().Operand())
		case ast.CallKind:
			call := e.AsCall()
			if call.Target() != nil {
				exprs = append(exprs, call.Target())
			}
			exprs = append(exprs, call.Args()...)
		case ast.ComprehensionKind:
			compre := e.AsComprehension()
			exprs = append(exprs,
				compre.IterRange(),
				compre.AccuInit(),
				compre.LoopCondition(),
				compre.LoopStep(),
				compre.Result())
		case ast.ListKind:
			list := e.AsList()
			exprs = append(exprs, list.Elements()...)
		case ast.MapKind:
			for _, entry := range e.AsMap().Entries() {
				e := entry.AsMapEntry()
				if visitor.visitEntry != nil {
					visitor.visitEntry(entry)
				}
				exprs = append(exprs, e.Key())
				exprs = append(exprs, e.Value())
			}
		case ast.StructKind:
			for _, entry := range e.AsStruct().Fields() {
				f := entry.AsStructField()
				if visitor.visitEntry != nil {
					visitor.visitEntry(entry)
				}
				exprs = append(exprs, f.Value())
			}
		}
	}
}
