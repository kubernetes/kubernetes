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
