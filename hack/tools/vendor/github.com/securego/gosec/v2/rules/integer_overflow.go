// (c) Copyright 2016 Hewlett Packard Enterprise Development LP
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rules

import (
	"fmt"
	"go/ast"

	"github.com/securego/gosec/v2"
)

type integerOverflowCheck struct {
	gosec.MetaData
	calls gosec.CallList
}

func (i *integerOverflowCheck) ID() string {
	return i.MetaData.ID
}

func (i *integerOverflowCheck) Match(node ast.Node, ctx *gosec.Context) (*gosec.Issue, error) {
	var atoiVarObj map[*ast.Object]ast.Node

	// To check multiple lines, ctx.PassedValues is used to store temporary data.
	if _, ok := ctx.PassedValues[i.ID()]; !ok {
		atoiVarObj = make(map[*ast.Object]ast.Node)
		ctx.PassedValues[i.ID()] = atoiVarObj
	} else if pv, ok := ctx.PassedValues[i.ID()].(map[*ast.Object]ast.Node); ok {
		atoiVarObj = pv
	} else {
		return nil, fmt.Errorf("PassedValues[%s] of Context is not map[*ast.Object]ast.Node, but %T", i.ID(), ctx.PassedValues[i.ID()])
	}

	// strconv.Atoi is a common function.
	// To reduce false positives, This rule detects code which is converted to int32/int16 only.
	switch n := node.(type) {
	case *ast.AssignStmt:
		for _, expr := range n.Rhs {
			if callExpr, ok := expr.(*ast.CallExpr); ok && i.calls.ContainsPkgCallExpr(callExpr, ctx, false) != nil {
				if idt, ok := n.Lhs[0].(*ast.Ident); ok && idt.Name != "_" {
					// Example:
					//  v, _ := strconv.Atoi("1111")
					// Add v's Obj to atoiVarObj map
					atoiVarObj[idt.Obj] = n
				}
			}
		}
	case *ast.CallExpr:
		if fun, ok := n.Fun.(*ast.Ident); ok {
			if fun.Name == "int32" || fun.Name == "int16" {
				if idt, ok := n.Args[0].(*ast.Ident); ok {
					if n, ok := atoiVarObj[idt.Obj]; ok {
						// Detect int32(v) and int16(v)
						return gosec.NewIssue(ctx, n, i.ID(), i.What, i.Severity, i.Confidence), nil
					}
				}
			}
		}
	}

	return nil, nil
}

// NewIntegerOverflowCheck detects if there is potential Integer OverFlow
func NewIntegerOverflowCheck(id string, conf gosec.Config) (gosec.Rule, []ast.Node) {
	calls := gosec.NewCallList()
	calls.Add("strconv", "Atoi")
	return &integerOverflowCheck{
		MetaData: gosec.MetaData{
			ID:         id,
			Severity:   gosec.High,
			Confidence: gosec.Medium,
			What:       "Potential Integer overflow made by strconv.Atoi result conversion to int16/32",
		},
		calls: calls,
	}, []ast.Node{(*ast.FuncDecl)(nil), (*ast.AssignStmt)(nil), (*ast.CallExpr)(nil)}
}
