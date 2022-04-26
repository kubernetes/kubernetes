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

package gosec

import "go/ast"

func resolveIdent(n *ast.Ident, c *Context) bool {
	if n.Obj == nil || n.Obj.Kind != ast.Var {
		return true
	}
	if node, ok := n.Obj.Decl.(ast.Node); ok {
		return TryResolve(node, c)
	}
	return false
}

func resolveValueSpec(n *ast.ValueSpec, c *Context) bool {
	if len(n.Values) == 0 {
		return false
	}
	for _, value := range n.Values {
		if !TryResolve(value, c) {
			return false
		}
	}
	return true
}

func resolveAssign(n *ast.AssignStmt, c *Context) bool {
	if len(n.Rhs) == 0 {
		return false
	}
	for _, arg := range n.Rhs {
		if !TryResolve(arg, c) {
			return false
		}
	}
	return true
}

func resolveCompLit(n *ast.CompositeLit, c *Context) bool {
	if len(n.Elts) == 0 {
		return false
	}
	for _, arg := range n.Elts {
		if !TryResolve(arg, c) {
			return false
		}
	}
	return true
}

func resolveBinExpr(n *ast.BinaryExpr, c *Context) bool {
	return (TryResolve(n.X, c) && TryResolve(n.Y, c))
}

func resolveCallExpr(n *ast.CallExpr, c *Context) bool {
	// TODO(tkelsey): next step, full function resolution
	return false
}

// TryResolve will attempt, given a subtree starting at some AST node, to resolve
// all values contained within to a known constant. It is used to check for any
// unknown values in compound expressions.
func TryResolve(n ast.Node, c *Context) bool {
	switch node := n.(type) {
	case *ast.BasicLit:
		return true
	case *ast.CompositeLit:
		return resolveCompLit(node, c)
	case *ast.Ident:
		return resolveIdent(node, c)
	case *ast.ValueSpec:
		return resolveValueSpec(node, c)
	case *ast.AssignStmt:
		return resolveAssign(node, c)
	case *ast.CallExpr:
		return resolveCallExpr(node, c)
	case *ast.BinaryExpr:
		return resolveBinExpr(node, c)
	}
	return false
}
