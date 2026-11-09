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
	"fmt"
	"go/ast"
	"go/token"
	"reflect"
	"strconv"
	"strings"
)

// Retain structural syntax even for branches the specialized interpreter refuses to execute.
// Positions carry only presence; parser symbol links and comments are not executable syntax.
func renderRaw(value reflect.Value) (string, error) {
	if !value.IsValid() {
		return ".absent", nil
	}
	if value.Kind() == reflect.Interface || value.Kind() == reflect.Pointer {
		if value.IsNil() {
			return ".absent", nil
		}
		return renderRaw(value.Elem())
	}
	if value.Type() == reflect.TypeOf(token.Pos(0)) {
		return "(.bit " + strconv.FormatBool(value.Int() != 0) + ")", nil
	}
	if value.Type() == reflect.TypeOf(token.ILLEGAL) {
		text, err := quoted(token.Token(value.Int()).String())
		return "(.atom " + text + ")", err
	}
	switch value.Kind() {
	case reflect.String:
		text, err := quoted(value.String())
		return "(.atom " + text + ")", err
	case reflect.Bool:
		return "(.bit " + strconv.FormatBool(value.Bool()) + ")", nil
	case reflect.Slice:
		items := make([]string, value.Len())
		for index := range items {
			item, err := renderRaw(value.Index(index))
			if err != nil {
				return "", err
			}
			items[index] = item
		}
		return "(.sequence [" + strings.Join(items, ",") + "])", nil
	case reflect.Struct:
		if value.Type().PkgPath() != "go/ast" {
			return "", fmt.Errorf("unsupported structural AST record %s", value.Type())
		}
		fields := []string{}
		for index := 0; index < value.NumField(); index++ {
			name := value.Type().Field(index).Name
			if name == "Obj" || name == "Doc" || name == "Comment" {
				continue
			}
			item, err := renderRaw(value.Field(index))
			if err != nil {
				return "", err
			}
			field, _ := quoted(name)
			fields = append(fields, "("+field+","+item+")")
		}
		// Keep the original token alongside Go's decoded constant for expression evaluation.
		if literal, ok := value.Interface().(ast.BasicLit); ok && literal.Kind == token.STRING {
			decoded, err := strconv.Unquote(literal.Value)
			if err != nil {
				return "", err
			}
			text, err := quoted(decoded)
			if err != nil {
				return "", err
			}
			fields = append(fields, "(\"DecodedString\",(.atom "+text+"))")
		}
		tag, _ := quoted(value.Type().Name())
		return "(.record " + tag + " [" + strings.Join(fields, ",") + "])", nil
	default:
		return "", fmt.Errorf("unsupported structural AST value %s", value.Type())
	}
}

var atomicRemoveFunctions = []functionSpec{
	{name: "removeItemsWithSchema", leanName: "factoryProgram", standalone: true, signature: "func removeItemsWithSchema(val value.Value, toRemove *fieldpath.Set, schema *schema.Schema, typeRef schema.TypeRef, shouldExtract bool) value.Value", rawSyntax: true},
	{name: "doScalar", leanName: "scalarProgram", receiver: "*removingWalker", signature: "func (w *removingWalker) doScalar(t *schema.Scalar) ValidationErrors", rawSyntax: true},
	{name: "doList", leanName: "listProgram", receiver: "*removingWalker", signature: "func (w *removingWalker) doList(t *schema.List) (errs ValidationErrors)", rawSyntax: true},
	{name: "doMap", leanName: "mapProgram", receiver: "*removingWalker", signature: "func (w *removingWalker) doMap(t *schema.Map) ValidationErrors", rawSyntax: true},
}

var schemaDispatchFunctions = []functionSpec{
	{name: "resolveSchema", leanName: "resolveProgram", standalone: true, signature: "func resolveSchema(s *schema.Schema, tr schema.TypeRef, v value.Value, ah atomHandler) ValidationErrors", rawSyntax: true},
	{name: "deduceAtom", leanName: "deduceProgram", standalone: true, signature: "func deduceAtom(atom schema.Atom, val value.Value) schema.Atom", rawSyntax: true},
	{name: "handleAtom", leanName: "handleProgram", standalone: true, signature: "func handleAtom(a schema.Atom, tr schema.TypeRef, ah atomHandler) ValidationErrors", rawSyntax: true},
}

// A raw tree must not hide variable aliases behind the interpreter's finite frame slots.
func validateRawBindings(function *ast.FuncDecl) error {
	localNodes := map[ast.Node]bool{}
	ast.Inspect(function.Body, func(node ast.Node) bool {
		if node != nil {
			localNodes[node] = true
		}
		return true
	})
	parameters := map[*ast.Object]bool{}
	for _, fields := range []*ast.FieldList{function.Recv, function.Type.Params, function.Type.Results} {
		if fields == nil {
			continue
		}
		for _, field := range fields.List {
			for _, name := range field.Names {
				parameters[name.Obj] = true
			}
		}
	}
	bindings := map[string]*ast.Object{}
	for object := range parameters {
		if object != nil {
			bindings[object.Name] = object
		}
	}
	// Go's parser represents range declarations with synthetic assignment nodes.
	ast.Inspect(function.Body, func(node ast.Node) bool {
		if loop, ok := node.(*ast.RangeStmt); ok && loop.Tok == token.DEFINE {
			for _, expression := range []ast.Expr{loop.Key, loop.Value} {
				if name, ok := expression.(*ast.Ident); ok && name.Obj != nil {
					parameters[name.Obj] = true
				}
			}
		}
		return true
	})
	fieldNames := map[*ast.Ident]bool{}
	ast.Inspect(function.Body, func(node ast.Node) bool {
		switch node := node.(type) {
		case *ast.SelectorExpr:
			fieldNames[node.Sel] = true
		case *ast.KeyValueExpr:
			if name, ok := node.Key.(*ast.Ident); ok {
				fieldNames[name] = true
			}
		}
		return true
	})
	reserved := map[string]bool{}
	for _, name := range []string{"w", "t", "l", "m", "v", "val", "s", "schema", "value", "tr", "ah", "atom", "a", "ok", "name", "typeName", "toRemove", "typeRef", "shouldExtract"} {
		reserved[name] = true
	}
	builtins := map[string]bool{"nil": true, "true": true, "false": true, "len": true, "append": true, "make": true, "new": true}
	packages := map[string]bool{"schema": true, "value": true, "fieldpath": true, "fmt": true, "errors": true, "strings": true}
	var failure error
	ast.Inspect(function.Body, func(node ast.Node) bool {
		if failure != nil {
			return false
		}
		name, ok := node.(*ast.Ident)
		if !ok || fieldNames[name] || name.Name == "_" {
			return true
		}
		object := name.Obj
		if object == nil {
			if reserved[name.Name] && !packages[name.Name] {
				failure = fmt.Errorf("raw frame variable %q is not locally bound", name.Name)
			}
			return true
		}
		if builtins[name.Name] {
			failure = fmt.Errorf("raw syntax cannot shadow builtin %q", name.Name)
			return false
		}
		if object.Kind != ast.Var && object.Kind != ast.Con {
			return true
		}
		declaration, ok := object.Decl.(ast.Node)
		if !parameters[object] && (!ok || !localNodes[declaration]) {
			failure = fmt.Errorf("raw syntax cannot read global variable or constant %q", name.Name)
			return false
		}
		if previous, found := bindings[name.Name]; found && previous != object {
			failure = fmt.Errorf("raw syntax cannot shadow variable %q", name.Name)
			return false
		}
		bindings[name.Name] = object
		return true
	})
	return failure
}
