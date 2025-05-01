/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package fields

import (
	"fmt"
	"github.com/google/cel-go/common/ast"
	pointer "k8s.io/utils/ptr"
	"maps"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v4/value"
)

// ReachableFields returns the set of fields that can be accessed from the given
// expression. If a path to a non-leaf value is returned, this indicates that the
// value at the path was either a target or argument of a function call. In such
// cases, the exact set of fields reachable by the expression cannot be precisely
// determined, since the implementation of the function is opaque, and the caller must
// assume that all fields contained in the value are reachable by the expression.
//
// Wildcard may be included in FieldElements of the result to identify where an
// object, map or list is indexed by unknown value.
func ReachableFields(e ast.Expr) *fieldpath.Set {
	t := newTracker()
	returns := t.search(e)
	return t.observed.Union(returns)
}

// Search returns resultReachable which identifies the fields that are reachable from
// the evaluated result of the expression. Fields that are reachable by the expression
// but not reachable from the result of the expression are tracked as observed.
// For example, consider the `a.b[c.d]` subexpression of `a.b[c.d].e`. This
// observes `c.d` and returns a resultReachable of `a.b[*]`. This resultReachable
// is then used to construct `a.b[*].e` as the resultReachable from the full
// expression.
func (t *tracker) search(expr ast.Expr) (resultReachable *fieldpath.Set) {
	if expr == nil {
		return fieldpath.NewSet()
	}
	switch expr.Kind() {
	case ast.IdentKind:
		return t.lookupVar(expr.AsIdent())
	case ast.CallKind:
		call := expr.AsCall()

		// The index operator must be handled as a special case to formulate the path the expression evaluates to.
		if call.FunctionName() == "_[_]" && len(call.Args()) == 2 {
			indexableArg := call.Args()[0]
			indexArg := call.Args()[1]

			t.observe(t.search(indexArg))
			indexed := fieldpath.NewSet()
			t.search(indexableArg).Iterate(func(path fieldpath.Path) {
				indexed.Insert(append(path, Wildcard)) // We don't attempt to provide data literal indices due to lack of typing
			})
			return indexed
		}
		// The + operator must be handled as a special case to combine the operands into the result. This is important for
		// proper handling of map and list concatenation.
		if call.FunctionName() == "_+_" && len(call.Args()) == 2 {
			lhs := t.search(call.Args()[0])
			rhs := t.search(call.Args()[1])
			return lhs.Union(rhs)
		}

		// All other function calls are handled as an observation of the target and args.
		t.observe(t.search(call.Target()))
		for _, arg := range call.Args() {
			t.observe(t.search(arg))
		}
		return fieldpath.NewSet()
	case ast.ComprehensionKind:
		t.pushScope()
		defer t.popScope()
		comprehension := expr.AsComprehension()

		iterRangePaths := t.search(comprehension.IterRange())

		// map keys are required to be strings in the Kubernetes API, so we don't need a KeyWildcard element.
		t.addIterVar(comprehension.IterVar(), appendAll(iterRangePaths, Wildcard)) // map key or array index
		if comprehension.HasIterVar2() {
			t.addIterVar(comprehension.IterVar2(), appendAll(iterRangePaths, Wildcard)) // map value or array element
		}

		accuPath := fieldpath.Path{fieldpath.PathElement{FieldName: pointer.To(comprehension.AccuVar())}}
		t.currentScope().accVars.Insert(accuPath)
		result := t.search(comprehension.LoopStep())
		result = result.Difference(fieldpath.NewSet(accuPath))
		return result
	case ast.ListKind:
		list := expr.AsList()
		result := fieldpath.NewSet()
		for _, elem := range list.Elements() {
			result = result.Union(t.search(elem))
		}
		return result
	case ast.LiteralKind:
		return fieldpath.NewSet() // nothing to do for scalar data literals
	case ast.MapKind:
		m := expr.AsMap()
		result := fieldpath.NewSet()
		for _, entry := range m.Entries() {
			t.observe(t.search(entry.AsMapEntry().Key()))
			result = result.Union(t.search(entry.AsMapEntry().Value()))
		}
		return result
	case ast.SelectKind:
		selectExpr := expr.AsSelect()
		result := fieldpath.NewSet()
		operandPaths := t.search(selectExpr.Operand())
		operandPaths.Iterate(func(path fieldpath.Path) {
			result.Insert(append(path, fieldpath.PathElement{FieldName: pointer.To(selectExpr.FieldName())}))
		})
		return result
	case ast.UnspecifiedExprKind:
		return fieldpath.NewSet()
	case ast.StructKind:
		panic("object initialization not supported")
	default:
		panic(fmt.Sprintf("unknown expression kind: %v", expr.Kind()))
	}
}

func newTracker() *tracker {
	return &tracker{observed: fieldpath.NewSet(), stack: []scope{newScope()}}
}

type tracker struct {
	observed *fieldpath.Set // tracks fields that are referenced in the expression but are not part of the returns
	stack    []scope
}

func (t *tracker) observe(sets ...*fieldpath.Set) {
	for _, s := range sets {
		s.Iterate(func(path fieldpath.Path) {
			if !t.currentScope().accVars.Has(path) {
				t.observed.Insert(path)
			}
		})
	}
}

func (t *tracker) addIterVar(name string, rangePaths *fieldpath.Set) {
	t.currentScope().iterVars[name] = rangePaths
}

func (t *tracker) lookupVar(ident string) *fieldpath.Set {
	if iterVars, ok := t.currentScope().iterVars[ident]; ok {
		return iterVars
	}
	return fieldpath.NewSet(fieldpath.Path{fieldpath.PathElement{FieldName: pointer.To(ident)}})
}

func (t *tracker) currentScope() scope {
	return t.stack[len(t.stack)-1]
}

func (t *tracker) pushScope() {
	top := t.stack[len(t.stack)-1]
	t.stack = append(t.stack, top.push())
}

func (t *tracker) popScope() {
	t.stack = t.stack[:len(t.stack)-1]
}

func newScope() scope {
	return scope{accVars: fieldpath.NewSet(), iterVars: map[string]*fieldpath.Set{}}
}

type scope struct {
	accVars  *fieldpath.Set            // Comprehension accumulator variable names
	iterVars map[string]*fieldpath.Set // Map from comprehension iterator variable names to the field paths they are bound to
}

// Push creates a new scope above the current scope. Variables in the new scope shadow those in the current scope.
func (v scope) push() scope {
	// We COULD avoid the copies here, but it's probably not worth it. Variable counts are bounded to iterator and
	// accumulator vars (max 3 per scope), and CEL scope depth is bounded to 12.
	return scope{
		accVars:  v.accVars.Union(fieldpath.NewSet()), // TODO: Switch to .Copy once available.
		iterVars: maps.Clone(v.iterVars),
	}
}

func appendAll(set *fieldpath.Set, el fieldpath.PathElement) *fieldpath.Set {
	result := fieldpath.NewSet()
	set.Iterate(func(path fieldpath.Path) {
		result.Insert(append(path, el))
	})
	return result
}

// Wildcard represents access to either a map entry or an array element.
var Wildcard = fieldpath.PathElement{Value: wildcard}

var wildcard = pointer.To(value.NewValueInterface("*"))
