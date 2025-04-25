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
func ReachableFields(e ast.Expr) sets.Set[string] {
	ac := &tracker{observed: sets.New[string]()}
	scope := scope{accVars: sets.New[string](), iterVars: map[string]string{}}
	returns := ac.paths(e, scope)
	return ac.observed.Union(returns)
}

type tracker struct {
	observed sets.Set[string] // observed tracks fields that are referenced in the expression but are not part of the returns
}

type scope struct {
	accVars  sets.Set[string]  // Comprehension accumulator variables
	iterVars map[string]string // Comprehension iterator variables
}

func (v scope) newScope() scope {
	return scope{
		accVars:  v.accVars.Clone(),
		iterVars: maps.Clone(v.iterVars),
	}
}

func (ac *tracker) paths(e ast.Expr, scope scope) sets.Set[string] {
	if e == nil {
		return nil
	}
	switch e.Kind() {
	case ast.CallKind:
		call := e.AsCall()
		targetPath := ac.paths(call.Target(), scope)

		// Index operator is implemented as a function where
		// the indexable and index scope are just args
		if call.FunctionName() == "_[_]" && len(call.Args()) == 2 {
			indexable := ac.paths(call.Args()[0], scope)
			indexed := sets.New[string]()
			for path := range indexable {
				indexed.Insert(path + ".@index")
			}
			index := ac.paths(call.Args()[1], scope)
			ac.observed = ac.observed.Union(index)
			return indexed
		}
		argPaths := sets.New[string]()
		for _, arg := range call.Args() {
			argPaths = argPaths.Union(ac.paths(arg, scope))
		}
		return targetPath.Union(argPaths)
	case ast.ComprehensionKind:
		vars := scope.newScope()
		comprehension := e.AsComprehension()
		for path := range ac.paths(comprehension.IterRange(), vars) {
			if !comprehension.HasIterVar2() {
				vars.iterVars[comprehension.IterVar()] = path + ".@item" // @item is a list element or map key
			} else {
				vars.iterVars[comprehension.IterVar()] = path + ".@index"  // @index is a list index or map key
				vars.iterVars[comprehension.IterVar2()] = path + ".@value" // @value is a list element or map value
			}
		}
		vars.accVars.Insert(comprehension.AccuVar())
		result := ac.paths(comprehension.LoopStep(), vars)
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
			result = result.Union(ac.paths(elem, scope))
		}
		return result
	case ast.LiteralKind:
		return sets.New[string]() // nothing to do for scalar data literals
	case ast.MapKind:
		m := e.AsMap()
		result := sets.New[string]()
		for _, entry := range m.Entries() {
			ac.observed = ac.observed.Union(ac.paths(entry.AsMapEntry().Key(), scope))
			result = result.Union(ac.paths(entry.AsMapEntry().Value(), scope))
		}
		return result
	case ast.SelectKind:
		selectExpr := e.AsSelect()
		result := sets.New[string]()
		for path := range ac.paths(selectExpr.Operand(), scope) {
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
