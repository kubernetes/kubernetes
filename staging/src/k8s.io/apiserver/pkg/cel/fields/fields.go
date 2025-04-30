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
	"k8s.io/apimachinery/pkg/util/sets"
	"maps"
)

// ReachableFields returns the set of fields that can be accessed from the given expression.
// If a non-scalar path is returned, this indicates that the value at the path was either a target or argument
// of a function call. In this case, the exact set of fields that can be accessed from the expression is not known.
func ReachableFields(e ast.Expr) sets.Set[string] {
	t := newTracker()
	returns := t.paths(e, newScope())
	return t.observed.Union(returns)
}

func newTracker() *tracker {
	return &tracker{observed: sets.New[string]()}
}

type tracker struct {
	observed sets.Set[string] // observed tracks fields that are referenced in the expression but are not part of the returns
}

func newScope() scope {
	return scope{accVars: sets.New[string](), iterVars: map[string]string{}}
}

type scope struct {
	accVars  sets.Set[string]  // Comprehension accumulator variables
	iterVars map[string]string // Comprehension iterator variables
}

// Push creates a new scope above the current scope. Variables in the new scope shadow the current scope.
func (v scope) push() scope {
	// We COULD use a stack of scopes. Might not be worth it. Variable counts are bounded to iter vars and
	// accr vars (max 3 per scope), and CEL scope depth is bounded to 12.
	return scope{
		accVars:  v.accVars.Clone(),
		iterVars: maps.Clone(v.iterVars),
	}
}

func (t *tracker) paths(e ast.Expr, scope scope) sets.Set[string] {
	if e == nil {
		return nil
	}
	switch e.Kind() {
	case ast.CallKind:
		call := e.AsCall()
		targetPath := t.paths(call.Target(), scope)

		// Index operator is implemented as a function where
		// the indexable and index scope are just args
		if call.FunctionName() == "_[_]" && len(call.Args()) == 2 {
			indexable := t.paths(call.Args()[0], scope)
			indexed := sets.New[string]()
			for path := range indexable {
				indexed.Insert(path + ".@index")
			}
			index := t.paths(call.Args()[1], scope)
			t.observed = t.observed.Union(index)
			return indexed
		}
		argPaths := sets.New[string]()
		for _, arg := range call.Args() {
			argPaths = argPaths.Union(t.paths(arg, scope))
		}
		return targetPath.Union(argPaths)
	case ast.ComprehensionKind:
		vars := scope.push()
		comprehension := e.AsComprehension()
		for path := range t.paths(comprehension.IterRange(), vars) {
			if !comprehension.HasIterVar2() {
				vars.iterVars[comprehension.IterVar()] = path + ".@item" // @item is a list element or map key
			} else {
				vars.iterVars[comprehension.IterVar()] = path + ".@index"  // @index is a list index or map key
				vars.iterVars[comprehension.IterVar2()] = path + ".@value" // @value is a list element or map value
			}
		}
		vars.accVars.Insert(comprehension.AccuVar())
		result := t.paths(comprehension.LoopStep(), vars)
		result = result.Delete(comprehension.AccuVar())
		return result
	case ast.IdentKind:
		ident := e.AsIdent()
		if i, ok := scope.iterVars[ident]; ok {
			return sets.New[string](i)
		}
		return sets.New[string](ident)
	case ast.ListKind:
		list := e.AsList()
		result := sets.New[string]()
		for _, elem := range list.Elements() {
			result = result.Union(t.paths(elem, scope))
		}
		return result
	case ast.LiteralKind:
		return sets.New[string]() // nothing to do for scalar data literals
	case ast.MapKind:
		m := e.AsMap()
		result := sets.New[string]()
		for _, entry := range m.Entries() {
			t.observed = t.observed.Union(t.paths(entry.AsMapEntry().Key(), scope))
			result = result.Union(t.paths(entry.AsMapEntry().Value(), scope))
		}
		return result
	case ast.SelectKind:
		selectExpr := e.AsSelect()
		result := sets.New[string]()
		for path := range t.paths(selectExpr.Operand(), scope) {
			result.Insert(path + "." + selectExpr.FieldName())
		}
		return result
	case ast.UnspecifiedExprKind:
		return sets.New[string]()
	case ast.StructKind:
		panic("object initialization not supported")
	default:
		panic(fmt.Sprintf("unknown expression kind: %v", e.Kind()))
	}
}
