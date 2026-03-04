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

package checker

import (
	"sort"

	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/debug"
)

type semanticAdorner struct {
	checked *ast.AST
}

var _ debug.Adorner = &semanticAdorner{}

func (a *semanticAdorner) GetMetadata(elem any) string {
	result := ""
	e, isExpr := elem.(ast.Expr)
	if !isExpr {
		return result
	}
	t := a.checked.TypeMap()[e.ID()]
	if t != nil {
		result += "~"
		result += FormatCELType(t)
	}

	switch e.Kind() {
	case ast.IdentKind,
		ast.CallKind,
		ast.ListKind,
		ast.StructKind,
		ast.SelectKind:
		if ref, found := a.checked.ReferenceMap()[e.ID()]; found {
			if len(ref.OverloadIDs) == 0 {
				result += "^" + ref.Name
			} else {
				sort.Strings(ref.OverloadIDs)
				for i, overload := range ref.OverloadIDs {
					if i == 0 {
						result += "^"
					} else {
						result += "|"
					}
					result += overload
				}
			}
		}
	}

	return result
}

// Print returns a string representation of the Expr message,
// annotated with types from the CheckedExpr.  The Expr must
// be a sub-expression embedded in the CheckedExpr.
func Print(e ast.Expr, checked *ast.AST) string {
	a := &semanticAdorner{checked: checked}
	return debug.ToAdornedDebugString(e, a)
}
