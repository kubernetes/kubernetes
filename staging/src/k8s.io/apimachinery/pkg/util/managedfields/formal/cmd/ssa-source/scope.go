/*
Copyright 2026 The Kubernetes Authors.

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

package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"strconv"
	"strings"
)

type binding struct {
	id   string
	kind string
}

// Unique lexical identities let the interpreter preserve updates without leaking shadows.
type scope struct {
	parent   *scope
	bindings map[string]binding
	serial   *int
	depth    int
	readOnly map[string]bool
}

func rootScope(function functionSpec) *scope {
	serial := 0
	globals := &scope{bindings: map[string]binding{}, serial: &serial, depth: -1}
	for _, name := range []string{"fmt", "nil", "fieldpath", "delete"} {
		globals.bindings[name] = binding{name, "builtin"}
	}
	root := globals.child()
	for _, parameter := range function.parameters {
		root.bindings[parameter] = binding{parameter, function.maps[parameter]}
	}
	root.bindings["s"] = binding{"s", ""}
	return root
}

func (s *scope) child() *scope {
	return &scope{parent: s, bindings: map[string]binding{}, serial: s.serial, depth: s.depth + 1, readOnly: map[string]bool{}}
}

func (s *scope) resolve(name string) (binding, bool) {
	if value, ok := s.bindings[name]; ok {
		return value, true
	}
	if s.parent != nil {
		return s.parent.resolve(name)
	}
	return binding{}, false
}

func (s *scope) declare(name, kind string) binding {
	if current, ok := s.bindings[name]; ok {
		return current
	}
	id := name
	_, shadows := s.resolve(name)
	if s.depth != 0 || shadows {
		*s.serial = *s.serial + 1
		id = fmt.Sprintf("%s$%d", name, *s.serial)
	}
	value := binding{id, kind}
	s.bindings[name] = value
	return value
}

func (s *scope) canWrite(id string) bool {
	if s.readOnly[id] {
		return false
	}
	return s.parent == nil || s.parent.canWrite(id)
}

func mapName(expression ast.Expr, context *scope, writable bool) (binding, error) {
	name, ok := expression.(*ast.Ident)
	if !ok {
		return binding{}, fmt.Errorf("unsupported Go map receiver %T", expression)
	}
	value, ok := context.resolve(name.Name)
	if !ok || (value.kind != "managed" && value.kind != "versions") {
		return binding{}, fmt.Errorf("unsupported Go map %q", name.Name)
	}
	if writable && (value.kind != "versions" || !context.canWrite(value.id)) {
		return binding{}, fmt.Errorf("mutation of read-only or iterated Go map %q", name.Name)
	}
	return value, nil
}

func renderExpr(expression ast.Expr, context *scope) (string, error) {
	switch e := expression.(type) {
	case *ast.Ident:
		value, ok := context.resolve(e.Name)
		if !ok {
			return "", fmt.Errorf("unsupported global or unbound Go name %q", e.Name)
		}
		name, err := quoted(value.id)
		return "(.var " + name + ")", err
	case *ast.BasicLit:
		if e.Kind != token.STRING {
			return "", fmt.Errorf("unsupported Go literal %s", e.Value)
		}
		value, err := strconv.Unquote(e.Value)
		if err != nil {
			return "", err
		}
		text, err := quoted(value)
		return "(.text " + text + ")", err
	case *ast.SelectorExpr:
		receiver, err := renderExpr(e.X, context)
		if err != nil {
			return "", err
		}
		field, err := quoted(e.Sel.Name)
		return "(.select " + receiver + " " + field + ")", err
	case *ast.CallExpr:
		method, ok := e.Fun.(*ast.SelectorExpr)
		if !ok || e.Ellipsis.IsValid() {
			return "", fmt.Errorf("unsupported Go call %T", e.Fun)
		}
		receiver, err := renderExpr(method.X, context)
		if err != nil {
			return "", err
		}
		args, err := renderExpressions(e.Args, context)
		if err != nil {
			return "", err
		}
		name, err := quoted(method.Sel.Name)
		return "(.call " + receiver + " " + name + " " + args + ")", err
	case *ast.BinaryExpr:
		if e.Op == token.LOR {
			left, err := renderExpr(e.X, context)
			if err != nil {
				return "", err
			}
			right, err := renderExpr(e.Y, context)
			return "(.orElse " + left + " " + right + ")", err
		}
		nilName, ok := e.Y.(*ast.Ident)
		if (e.Op != token.NEQ && e.Op != token.EQL) || !ok || nilName.Name != "nil" {
			return "", fmt.Errorf("unsupported Go comparison %s", e.Op)
		}
		if value, exists := context.resolve("nil"); !exists || value.id != "nil" || value.kind != "builtin" {
			return "", fmt.Errorf("unsupported comparison against a shadowed nil identifier")
		}
		value, err := renderExpr(e.X, context)
		if e.Op == token.EQL {
			return "(.eqNil " + value + ")", err
		}
		return "(.neNil " + value + ")", err
	case *ast.UnaryExpr:
		if e.Op != token.NOT {
			return "", fmt.Errorf("unsupported Go unary operator %s", e.Op)
		}
		value, err := renderExpr(e.X, context)
		return "(.notFlag " + value + ")", err
	case *ast.IndexExpr:
		if _, err := mapName(e.X, context, false); err != nil {
			return "", err
		}
		receiver, err := renderExpr(e.X, context)
		if err != nil {
			return "", err
		}
		key, err := renderExpr(e.Index, context)
		return "(.index " + receiver + " " + key + ")", err
	case *ast.CompositeLit:
		var typ bytes.Buffer
		if err := format.Node(&typ, token.NewFileSet(), e.Type); err != nil {
			return "", err
		}
		if typ.String() != "map[fieldpath.APIVersion]*fieldpath.Set" || len(e.Elts) != 0 {
			return "", fmt.Errorf("unsupported Go composite literal %s", typ.String())
		}
		return "(.emptyVersionMap)", nil
	default:
		return "", fmt.Errorf("unsupported Go expression %T", expression)
	}
}

func renderExpressions(input []ast.Expr, context *scope) (string, error) {
	items := make([]string, 0, len(input))
	for _, input := range input {
		item, err := renderExpr(input, context)
		if err != nil {
			return "", err
		}
		items = append(items, item)
	}
	return "[" + strings.Join(items, ", ") + "]", nil
}

func renderStatements(input []ast.Stmt, context *scope) (string, error) {
	items := make([]string, 0, len(input))
	for _, statement := range input {
		switch s := statement.(type) {
		case *ast.DeclStmt:
			declaration, ok := s.Decl.(*ast.GenDecl)
			if !ok || declaration.Tok != token.VAR || len(declaration.Specs) != 1 {
				return "", fmt.Errorf("unsupported Go declaration")
			}
			value, ok := declaration.Specs[0].(*ast.ValueSpec)
			if !ok || len(value.Names) != 1 || len(value.Values) != 0 {
				return "", fmt.Errorf("unsupported Go variable declaration")
			}
			typ, ok := value.Type.(*ast.Ident)
			if !ok || typ.Name != "error" || value.Names[0].Name == "_" {
				return "", fmt.Errorf("unsupported Go zero-value type")
			}
			name := value.Names[0].Name
			if _, exists := context.bindings[name]; exists {
				return "", fmt.Errorf("duplicate Go variable declaration %q", name)
			}
			text, err := quoted(context.declare(name, "").id)
			if err != nil {
				return "", err
			}
			items = append(items, "(.declareError "+text+")")
		case *ast.AssignStmt:
			if s.Tok != token.ASSIGN && s.Tok != token.DEFINE {
				return "", fmt.Errorf("unsupported Go assignment operator %s", s.Tok)
			}
			if len(s.Lhs) == 1 {
				if index, ok := s.Lhs[0].(*ast.IndexExpr); ok {
					if s.Tok != token.ASSIGN || len(s.Rhs) != 1 {
						return "", fmt.Errorf("unsupported Go map assignment")
					}
					mapVar, err := mapName(index.X, context, true)
					if err != nil {
						return "", err
					}
					key, err := renderExpr(index.Index, context)
					if err != nil {
						return "", err
					}
					value, err := renderExpr(s.Rhs[0], context)
					if err != nil {
						return "", err
					}
					name, _ := quoted(mapVar.id)
					items = append(items, "(.mapSet "+name+" "+key+" "+value+")")
					continue
				}
			}
			// Map aliases require heap semantics and are deliberately outside this subset.
			for _, rhs := range s.Rhs {
				if name, ok := rhs.(*ast.Ident); ok {
					if value, exists := context.resolve(name.Name); exists && value.kind != "" && value.kind != "builtin" {
						return "", fmt.Errorf("unsupported Go map alias %q", name.Name)
					}
				}
			}
			values, err := renderExpressions(s.Rhs, context)
			if err != nil {
				return "", err
			}
			if len(s.Lhs) == 2 && len(s.Rhs) == 1 {
				if index, ok := s.Rhs[0].(*ast.IndexExpr); ok {
					receiver, _ := renderExpr(index.X, context)
					key, _ := renderExpr(index.Index, context)
					values = "[(.indexTwo " + receiver + " " + key + ")]"
				}
			}
			mapKind := ""
			if len(s.Lhs) == 1 && len(s.Rhs) == 1 {
				if _, composite := s.Rhs[0].(*ast.CompositeLit); composite {
					mapKind = "versions"
				}
			}
			names := make([]string, 0, len(s.Lhs))
			for _, target := range s.Lhs {
				name, ok := target.(*ast.Ident)
				if !ok {
					return "", fmt.Errorf("unsupported Go assignment target %T", target)
				}
				id := "_"
				if name.Name != "_" {
					var value binding
					if s.Tok == token.DEFINE {
						value = context.declare(name.Name, mapKind)
					} else {
						var exists bool
						value, exists = context.resolve(name.Name)
						if !exists || value.kind != "" {
							return "", fmt.Errorf("unsupported Go variable assignment %q", name.Name)
						}
					}
					id = value.id
				}
				text, err := quoted(id)
				if err != nil {
					return "", err
				}
				names = append(names, text)
			}
			items = append(items, "(.assign ["+strings.Join(names, ", ")+"] "+values+")")
		case *ast.IfStmt:
			if s.Else != nil {
				return "", fmt.Errorf("unsupported Go else")
			}
			conditionScope := context.child()
			if s.Init != nil {
				init, err := renderStatements([]ast.Stmt{s.Init}, conditionScope)
				if err != nil {
					return "", err
				}
				// Flatten only the initializer; fresh lexical identities retain its scope.
				items = append(items, strings.TrimSuffix(strings.TrimPrefix(init, "[\n  "), "\n]"))
			}
			condition, err := renderExpr(s.Cond, conditionScope)
			if err != nil {
				return "", err
			}
			body, err := renderStatements(s.Body.List, conditionScope.child())
			if err != nil {
				return "", err
			}
			items = append(items, "(.branch "+condition+" "+body+")")
		case *ast.RangeStmt:
			if s.Tok != token.DEFINE || s.Key == nil || s.Value == nil {
				return "", fmt.Errorf("unsupported Go range form")
			}
			mapVar, err := mapName(s.X, context, false)
			if err != nil {
				return "", err
			}
			collection, err := renderExpr(s.X, context)
			if err != nil {
				return "", err
			}
			iteration := context.child()
			iteration.readOnly[mapVar.id] = true
			names := []string{}
			for _, variable := range []ast.Expr{s.Key, s.Value} {
				name, ok := variable.(*ast.Ident)
				if !ok {
					return "", fmt.Errorf("unsupported Go range variable %T", variable)
				}
				id := "_"
				if name.Name != "_" {
					id = iteration.declare(name.Name, "").id
				}
				text, err := quoted(id)
				if err != nil {
					return "", err
				}
				names = append(names, text)
			}
			body, err := renderStatements(s.Body.List, iteration.child())
			if err != nil {
				return "", err
			}
			items = append(items, "(.rangeMap "+names[0]+" "+names[1]+" "+collection+" "+body+")")
		case *ast.ExprStmt:
			call, ok := s.X.(*ast.CallExpr)
			if !ok || len(call.Args) != 2 {
				return "", fmt.Errorf("unsupported Go expression statement")
			}
			name, ok := call.Fun.(*ast.Ident)
			builtin, resolved := context.resolve("delete")
			if !ok || name.Name != "delete" || !resolved || builtin.kind != "builtin" {
				return "", fmt.Errorf("unsupported Go expression statement")
			}
			mapVar, err := mapName(call.Args[0], context, true)
			if err != nil {
				return "", err
			}
			key, err := renderExpr(call.Args[1], context)
			if err != nil {
				return "", err
			}
			id, _ := quoted(mapVar.id)
			items = append(items, "(.mapDelete "+id+" "+key+")")
		case *ast.ReturnStmt:
			if len(s.Results) == 0 {
				return "", fmt.Errorf("unsupported Go named return")
			}
			values, err := renderExpressions(s.Results, context)
			if err != nil {
				return "", err
			}
			items = append(items, "(.ret "+values+")")
		default:
			return "", fmt.Errorf("unsupported Go statement %T", statement)
		}
	}
	return "[\n  " + strings.Join(items, ",\n  ") + "\n]", nil
}
