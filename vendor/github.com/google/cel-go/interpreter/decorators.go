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
	"github.com/google/cel-go/common/overloads"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
)

// InterpretableDecorator is a functional interface for decorating or replacing
// Interpretable expression nodes at construction time.
type InterpretableDecorator func(Interpretable) (Interpretable, error)

// evalObserver is a functional interface that accepts an expression id and an observed value.
type evalObserver func(int64, ref.Val)

// decObserveEval records evaluation state into an EvalState object.
func decObserveEval(observer evalObserver) InterpretableDecorator {
	return func(i Interpretable) (Interpretable, error) {
		switch inst := i.(type) {
		case *evalWatch, *evalWatchAttr, *evalWatchConst:
			// these instruction are already watching, return straight-away.
			return i, nil
		case InterpretableAttribute:
			return &evalWatchAttr{
				InterpretableAttribute: inst,
				observer:               observer,
			}, nil
		case InterpretableConst:
			return &evalWatchConst{
				InterpretableConst: inst,
				observer:           observer,
			}, nil
		default:
			return &evalWatch{
				Interpretable: i,
				observer:      observer,
			}, nil
		}
	}
}

// decDisableShortcircuits ensures that all branches of an expression will be evaluated, no short-circuiting.
func decDisableShortcircuits() InterpretableDecorator {
	return func(i Interpretable) (Interpretable, error) {
		switch expr := i.(type) {
		case *evalOr:
			return &evalExhaustiveOr{
				id:  expr.id,
				lhs: expr.lhs,
				rhs: expr.rhs,
			}, nil
		case *evalAnd:
			return &evalExhaustiveAnd{
				id:  expr.id,
				lhs: expr.lhs,
				rhs: expr.rhs,
			}, nil
		case *evalFold:
			return &evalExhaustiveFold{
				id:        expr.id,
				accu:      expr.accu,
				accuVar:   expr.accuVar,
				iterRange: expr.iterRange,
				iterVar:   expr.iterVar,
				cond:      expr.cond,
				step:      expr.step,
				result:    expr.result,
			}, nil
		case InterpretableAttribute:
			cond, isCond := expr.Attr().(*conditionalAttribute)
			if isCond {
				return &evalExhaustiveConditional{
					id:      cond.id,
					attr:    cond,
					adapter: expr.Adapter(),
				}, nil
			}
		}
		return i, nil
	}
}

// decOptimize optimizes the program plan by looking for common evaluation patterns and
// conditionally precomputating the result.
// - build list and map values with constant elements.
// - convert 'in' operations to set membership tests if possible.
func decOptimize() InterpretableDecorator {
	return func(i Interpretable) (Interpretable, error) {
		switch inst := i.(type) {
		case *evalList:
			return maybeBuildListLiteral(i, inst)
		case *evalMap:
			return maybeBuildMapLiteral(i, inst)
		case InterpretableCall:
			if inst.OverloadID() == overloads.InList {
				return maybeOptimizeSetMembership(i, inst)
			}
			if overloads.IsTypeConversionFunction(inst.Function()) {
				return maybeOptimizeConstUnary(i, inst)
			}
		}
		return i, nil
	}
}

func maybeOptimizeConstUnary(i Interpretable, call InterpretableCall) (Interpretable, error) {
	args := call.Args()
	if len(args) != 1 {
		return i, nil
	}
	_, isConst := args[0].(InterpretableConst)
	if !isConst {
		return i, nil
	}
	val := call.Eval(EmptyActivation())
	if types.IsError(val) {
		return nil, val.(*types.Err)
	}
	return NewConstValue(call.ID(), val), nil
}

func maybeBuildListLiteral(i Interpretable, l *evalList) (Interpretable, error) {
	for _, elem := range l.elems {
		_, isConst := elem.(InterpretableConst)
		if !isConst {
			return i, nil
		}
	}
	return NewConstValue(l.ID(), l.Eval(EmptyActivation())), nil
}

func maybeBuildMapLiteral(i Interpretable, mp *evalMap) (Interpretable, error) {
	for idx, key := range mp.keys {
		_, isConst := key.(InterpretableConst)
		if !isConst {
			return i, nil
		}
		_, isConst = mp.vals[idx].(InterpretableConst)
		if !isConst {
			return i, nil
		}
	}
	return NewConstValue(mp.ID(), mp.Eval(EmptyActivation())), nil
}

// maybeOptimizeSetMembership may convert an 'in' operation against a list to map key membership
// test if the following conditions are true:
// - the list is a constant with homogeneous element types.
// - the elements are all of primitive type.
func maybeOptimizeSetMembership(i Interpretable, inlist InterpretableCall) (Interpretable, error) {
	args := inlist.Args()
	lhs := args[0]
	rhs := args[1]
	l, isConst := rhs.(InterpretableConst)
	if !isConst {
		return i, nil
	}
	// When the incoming binary call is flagged with as the InList overload, the value will
	// always be convertible to a `traits.Lister` type.
	list := l.Value().(traits.Lister)
	if list.Size() == types.IntZero {
		return NewConstValue(inlist.ID(), types.False), nil
	}
	it := list.Iterator()
	var typ ref.Type
	valueSet := make(map[ref.Val]ref.Val)
	for it.HasNext() == types.True {
		elem := it.Next()
		if !types.IsPrimitiveType(elem) {
			// Note, non-primitive type are not yet supported.
			return i, nil
		}
		if typ == nil {
			typ = elem.Type()
		} else if typ.TypeName() != elem.Type().TypeName() {
			return i, nil
		}
		valueSet[elem] = types.True
	}
	return &evalSetMembership{
		inst:        inlist,
		arg:         lhs,
		argTypeName: typ.TypeName(),
		valueSet:    valueSet,
	}, nil
}
