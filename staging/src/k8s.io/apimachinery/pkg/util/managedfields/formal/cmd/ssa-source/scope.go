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
	parent      *scope
	bindings    map[string]binding
	serial      *int
	depth       int
	readOnly    map[string]bool
	elements    bool
	updater     bool
	inLoop      bool
	currentKeys map[string]string
	sets        bool
}

func rootScope(function functionSpec) *scope {
	serial := 0
	globals := &scope{bindings: map[string]binding{}, serial: &serial, depth: -1}
	for _, name := range []string{"fmt", "nil", "true", "false", "fieldpath", "value", "delete"} {
		globals.bindings[name] = binding{name, "builtin"}
	}
	root := globals.child()
	root.elements = function.elements
	root.updater = function.updater
	root.sets = function.sets
	if function.updater {
		for _, name := range []string{"len", "ConflictsFromManagers"} {
			globals.bindings[name] = binding{name, "builtin"}
		}
	}
	if function.elements {
		for _, name := range []string{"len", "append"} {
			globals.bindings[name] = binding{name, "builtin"}
		}
	}
	for _, parameter := range function.parameters {
		root.bindings[parameter] = binding{parameter, function.maps[parameter]}
	}
	root.bindings["s"] = binding{"s", ""}
	if function.elements {
		root.bindings["s"] = binding{"s", "element-input"}
	}
	return root
}

func (s *scope) child() *scope {
	return &scope{parent: s, bindings: map[string]binding{}, serial: s.serial, depth: s.depth + 1, readOnly: map[string]bool{}, elements: s.elements, updater: s.updater, inLoop: s.inLoop, currentKeys: map[string]string{}, sets: s.sets}
}

func (s *scope) currentKey(mapID string) string {
	if key := s.currentKeys[mapID]; key != "" {
		return key
	}
	if s.parent != nil {
		return s.parent.currentKey(mapID)
	}
	return ""
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
	if !ok || (value.kind != "managed" && value.kind != "managed-write" && value.kind != "versions" && value.kind != "comparisons") {
		return binding{}, fmt.Errorf("unsupported Go map %q", name.Name)
	}
	if writable && ((value.kind != "versions" && value.kind != "managed-write" && value.kind != "comparisons") || !context.canWrite(value.id)) {
		return binding{}, fmt.Errorf("mutation of read-only or iterated Go map %q", name.Name)
	}
	return value, nil
}

func readableMap(expression ast.Expr, context *scope) error {
	if selector, ok := expression.(*ast.SelectorExpr); ok {
		if receiver, ok := selector.X.(*ast.Ident); ok && receiver.Name == "s" &&
			(selector.Sel.Name == "IgnoredFields" || selector.Sel.Name == "IgnoreFilter") {
			if bound, found := context.resolve("s"); found && bound.id == "s" {
				return nil
			}
		}
	}
	_, err := mapName(expression, context, false)
	return err
}

func renderExpr(expression ast.Expr, context *scope) (string, error) {
	switch e := expression.(type) {
	case *ast.Ident:
		value, ok := context.resolve(e.Name)
		if !ok {
			return "", fmt.Errorf("unsupported global or unbound Go name %q", e.Name)
		}
		if value.kind == "builtin" && (value.id == "true" || value.id == "false") {
			return "(.flag " + value.id + ")", nil
		}
		name, err := quoted(value.id)
		return "(.var " + name + ")", err
	case *ast.BasicLit:
		if (context.elements || context.updater) && e.Kind == token.INT {
			if len(e.Value) > 1 && e.Value[0] == '0' {
				return "", fmt.Errorf("unsupported Go non-decimal integer %q", e.Value)
			}
			value, err := strconv.ParseUint(e.Value, 10, 64)
			if err != nil {
				return "", fmt.Errorf("unsupported Go non-decimal or negative integer %q", e.Value)
			}
			return fmt.Sprintf("(.number %d)", value), nil
		}
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
		if name, ok := e.Fun.(*ast.Ident); ok && (context.elements || context.updater) {
			bound, found := context.resolve(name.Name)
			if !found || bound.kind != "builtin" {
				return "", fmt.Errorf("unsupported or shadowed element-set builtin %q", name.Name)
			}
			if name.Name == "len" && len(e.Args) == 1 && !e.Ellipsis.IsValid() {
				value, err := renderExpr(e.Args[0], context)
				return "(.length " + value + ")", err
			}
			if context.updater && name.Name == "ConflictsFromManagers" && len(e.Args) == 1 && !e.Ellipsis.IsValid() {
				value, err := renderExpr(e.Args[0], context)
				return "(.conflictsFromManagers " + value + ")", err
			}
			if context.elements && name.Name == "append" && len(e.Args) == 2 {
				left, err := renderExpr(e.Args[0], context)
				if err != nil {
					return "", err
				}
				right, err := renderExpr(e.Args[1], context)
				constructor := "appendOne"
				if e.Ellipsis.IsValid() {
					constructor = "concatenate"
				}
				return "(." + constructor + " " + left + " " + right + ")", err
			}
			return "", fmt.Errorf("unsupported element-set builtin call %q", name.Name)
		}
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
		if context.updater && (e.Op == token.EQL || e.Op == token.NEQ) {
			if name, nilComparison := e.Y.(*ast.Ident); !nilComparison || name.Name != "nil" {
				left, err := renderExpr(e.X, context)
				if err != nil {
					return "", err
				}
				right, err := renderExpr(e.Y, context)
				if err != nil {
					return "", err
				}
				comparison := "(.equalScalars " + left + " " + right + ")"
				if e.Op == token.NEQ {
					comparison = "(.notFlag " + comparison + ")"
				}
				return comparison, nil
			}
		}
		if context.elements && e.Op == token.LSS {
			left, err := renderExpr(e.X, context)
			if err != nil {
				return "", err
			}
			right, err := renderExpr(e.Y, context)
			return "(.less " + left + " " + right + ")", err
		}
		if e.Op == token.LOR || e.Op == token.LAND {
			left, err := renderExpr(e.X, context)
			if err != nil {
				return "", err
			}
			right, err := renderExpr(e.Y, context)
			operator := "orElse"
			if e.Op == token.LAND {
				operator = "andAlso"
			}
			return "(." + operator + " " + left + " " + right + ")", err
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
		if context.sets && e.Op == token.AND {
			value, err := renderExpr(e.X, context)
			return "(.addressOf " + value + ")", err
		}
		if context.elements && e.Op == token.AND {
			if composite, ok := e.X.(*ast.CompositeLit); ok && len(composite.Elts) == 0 {
				if name, ok := composite.Type.(*ast.Ident); ok && name.Name == "PathElementSet" {
					return "(.emptyElementSet)", nil
				}
			}
		}
		if e.Op != token.NOT {
			return "", fmt.Errorf("unsupported Go unary operator %s", e.Op)
		}
		value, err := renderExpr(e.X, context)
		return "(.notFlag " + value + ")", err
	case *ast.StarExpr:
		if !context.sets {
			return "", fmt.Errorf("Go dereference requires the recursive-set interpreter")
		}
		value, err := renderExpr(e.X, context)
		return "(.dereference " + value + ")", err
	case *ast.IndexExpr:
		if !context.elements {
			if err := readableMap(e.X, context); err != nil {
				return "", err
			}
		}
		receiver, err := renderExpr(e.X, context)
		if err != nil {
			return "", err
		}
		key, err := renderExpr(e.Index, context)
		return "(.index " + receiver + " " + key + ")", err
	case *ast.SliceExpr:
		if !context.elements || e.Low == nil || e.High != nil || e.Max != nil || e.Slice3 {
			return "", fmt.Errorf("unsupported Go slice expression")
		}
		value, err := renderExpr(e.X, context)
		if err != nil {
			return "", err
		}
		index, err := renderExpr(e.Low, context)
		return "(.sliceFrom " + value + " " + index + ")", err
	case *ast.CompositeLit:
		var typ bytes.Buffer
		if err := format.Node(&typ, token.NewFileSet(), e.Type); err != nil {
			return "", err
		}
		if context.sets {
			if typ.String() == "SetNodeMap" && len(e.Elts) == 0 {
				return "(.emptyNodeMap)", nil
			}
			fields := []string{}
			constructor := ""
			if typ.String() == "Set" {
				fields, constructor = []string{"Members", "Children"}, "setRecord"
			}
			if typ.String() == "setNode" {
				fields, constructor = []string{"pathElement", "set"}, "setNode"
			}
			if constructor != "" {
				if len(e.Elts) != len(fields) {
					return "", fmt.Errorf("%s literal must initialize exactly %v", typ.String(), fields)
				}
				values := make([]string, len(fields))
				for index, field := range fields {
					entry, ok := e.Elts[index].(*ast.KeyValueExpr)
					if !ok {
						return "", fmt.Errorf("%s literal requires named fields in evaluation order", typ.String())
					}
					name, ok := entry.Key.(*ast.Ident)
					if !ok || name.Name != field {
						return "", fmt.Errorf("%s literal field %d must be %s", typ.String(), index, field)
					}
					value, err := renderExpr(entry.Value, context)
					if err != nil {
						return "", err
					}
					values[index] = value
				}
				return "(." + constructor + " " + strings.Join(values, " ") + ")", nil
			}
		}
		if context.updater && typ.String() == "map[fieldpath.APIVersion]*typed.Comparison" && len(e.Elts) == 1 {
			entry, ok := e.Elts[0].(*ast.KeyValueExpr)
			if !ok {
				return "", fmt.Errorf("comparison cache literal requires an explicit key")
			}
			key, err := renderExpr(entry.Key, context)
			if err != nil {
				return "", err
			}
			value, err := renderExpr(entry.Value, context)
			return "(.singletonComparisonMap " + key + " " + value + ")", err
		}
		if len(e.Elts) != 0 {
			return "", fmt.Errorf("unsupported Go composite literal %s", typ.String())
		}
		switch typ.String() {
		case "map[fieldpath.APIVersion]*fieldpath.Set":
			return "(.emptyVersionMap)", nil
		case "fieldpath.ManagedFields":
			return "(.emptyManagedMap)", nil
		default:
			return "", fmt.Errorf("unsupported Go composite literal %s", typ.String())
		}
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
			var typ bytes.Buffer
			if err := format.Node(&typ, token.NewFileSet(), value.Type); err != nil {
				return "", err
			}
			comparisonMap := context.updater && typ.String() == "map[fieldpath.APIVersion]*typed.Comparison"
			if (typ.String() != "error" && typ.String() != "fieldpath.Filter" && !comparisonMap) || value.Names[0].Name == "_" {
				return "", fmt.Errorf("unsupported Go zero-value type")
			}
			name := value.Names[0].Name
			if _, exists := context.bindings[name]; exists {
				return "", fmt.Errorf("duplicate Go variable declaration %q", name)
			}
			kind := ""
			if comparisonMap {
				kind = "comparisons"
			}
			text, err := quoted(context.declare(name, kind).id)
			if err != nil {
				return "", err
			}
			constructor := "declareError"
			if typ.String() == "fieldpath.Filter" || comparisonMap {
				constructor = "declareNil"
			}
			items = append(items, "(."+constructor+" "+text+")")
		case *ast.AssignStmt:
			if s.Tok != token.ASSIGN && s.Tok != token.DEFINE {
				return "", fmt.Errorf("unsupported Go assignment operator %s", s.Tok)
			}
			if len(s.Lhs) == 1 {
				if field, ok := s.Lhs[0].(*ast.SelectorExpr); ok && context.elements {
					name, ok := field.X.(*ast.Ident)
					if !ok || s.Tok != token.ASSIGN || len(s.Rhs) != 1 || field.Sel.Name != "members" {
						return "", fmt.Errorf("unsupported element-set field assignment")
					}
					bound, found := context.resolve(name.Name)
					if !found || bound.kind != "element-fresh" {
						return "", fmt.Errorf("element-set writes require a fresh local allocation")
					}
					// Appending only to the fresh output prevents input backing-array aliases.
					call, ok := s.Rhs[0].(*ast.CallExpr)
					if !ok || len(call.Args) == 0 {
						return "", fmt.Errorf("element-set member writes must append to their own output")
					}
					appendName, ok := call.Fun.(*ast.Ident)
					base, selected := call.Args[0].(*ast.SelectorExpr)
					if !ok || appendName.Name != "append" || !selected || base.Sel.Name != "members" {
						return "", fmt.Errorf("element-set member writes must append to their own output")
					}
					baseName, ok := base.X.(*ast.Ident)
					if !ok {
						return "", fmt.Errorf("unsupported element-set append receiver")
					}
					baseBinding, found := context.resolve(baseName.Name)
					if !found || baseBinding.id != bound.id {
						return "", fmt.Errorf("element-set append may not reuse input storage")
					}
					value, err := renderExpr(s.Rhs[0], context)
					if err != nil {
						return "", err
					}
					id, _ := quoted(bound.id)
					items = append(items, "(.fieldSet "+id+" \"members\" "+value+")")
					continue
				}
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
					if context.updater {
						var typ bytes.Buffer
						if err := format.Node(&typ, token.NewFileSet(), s.Rhs[0].(*ast.CompositeLit).Type); err != nil {
							return "", err
						}
						if typ.String() == "fieldpath.ManagedFields" {
							mapKind = "managed-write"
						}
						if typ.String() == "map[fieldpath.APIVersion]*typed.Comparison" {
							mapKind = "comparisons"
						}
					}
				}
				if context.elements {
					if fresh, ok := s.Rhs[0].(*ast.UnaryExpr); ok && fresh.Op == token.AND {
						if _, allocated := fresh.X.(*ast.CompositeLit); !allocated {
							return "", fmt.Errorf("local set addresses require a fresh composite allocation")
						}
						mapKind = "element-fresh"
					}
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
						if !exists || (value.kind != "" && value.kind != "managed-write" && !(context.updater && value.kind == "comparisons" && mapKind == "comparisons")) {
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
		case *ast.IncDecStmt:
			name, ok := s.X.(*ast.Ident)
			if !context.elements || !ok || s.Tok != token.INC {
				return "", fmt.Errorf("unsupported Go increment/decrement")
			}
			bound, found := context.resolve(name.Name)
			if !found || bound.kind != "" {
				return "", fmt.Errorf("unsupported Go increment target")
			}
			id, _ := quoted(bound.id)
			items = append(items, "(.increment "+id+")")
		case *ast.ForStmt:
			if !context.elements || s.Init != nil || s.Post != nil || s.Cond == nil {
				return "", fmt.Errorf("unsupported Go for loop")
			}
			condition, err := renderExpr(s.Cond, context)
			if err != nil {
				return "", err
			}
			body, err := renderStatements(s.Body.List, context.child())
			if err != nil {
				return "", err
			}
			items = append(items, "(.whileLoop "+condition+" "+body+")")
		case *ast.IfStmt:
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
			if s.Else == nil {
				items = append(items, "(.branch "+condition+" "+body+")")
			} else {
				alternative, ok := s.Else.(*ast.BlockStmt)
				if !ok {
					return "", fmt.Errorf("unsupported Go else form %T", s.Else)
				}
				other, err := renderStatements(alternative.List, conditionScope.child())
				if err != nil {
					return "", err
				}
				items = append(items, "(.branchElse "+condition+" "+body+" "+other+")")
			}
		case *ast.RangeStmt:
			if s.Tok != token.DEFINE || s.Key == nil || (s.Value == nil && !context.updater) {
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
			iteration.inLoop = true
			names := []string{}
			for _, variable := range []ast.Expr{s.Key, s.Value} {
				if variable == nil {
					names = append(names, `"_"`)
					continue
				}
				name, ok := variable.(*ast.Ident)
				if !ok {
					return "", fmt.Errorf("unsupported Go range variable %T", variable)
				}
				id := "_"
				if name.Name != "_" {
					id = iteration.declare(name.Name, "").id
				}
				if len(names) == 0 && id != "_" {
					iteration.currentKeys[mapVar.id] = id
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
		case *ast.BranchStmt:
			if !context.updater || !context.inLoop || s.Tok != token.CONTINUE || s.Label != nil {
				return "", fmt.Errorf("unsupported Go branch statement")
			}
			items = append(items, "(.continueLoop)")
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
			mapVar, err := mapName(call.Args[0], context, false)
			if err != nil {
				return "", err
			}
			currentDelete := false
			if context.updater && mapVar.kind == "managed-write" {
				if name, ok := call.Args[1].(*ast.Ident); ok {
					key, found := context.resolve(name.Name)
					currentDelete = found && key.id == context.currentKey(mapVar.id)
				}
			}
			if !currentDelete {
				if _, err := mapName(call.Args[0], context, true); err != nil {
					return "", err
				}
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
