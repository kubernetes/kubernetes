/* Copyright 2016 The Bazel Authors. All rights reserved.

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

package rule

import (
	"fmt"
	"log"
	"reflect"
	"sort"

	bzl "github.com/bazelbuild/buildtools/build"
)

// KeyValue represents a key-value pair. This gets converted into a
// rule attribute, i.e., a Skylark keyword argument.
type KeyValue struct {
	Key   string
	Value interface{}
}

// GlobValue represents a Bazel glob expression.
type GlobValue struct {
	Patterns []string
	Excludes []string
}

// BzlExprValue is implemented by types that have custom translations
// to Starlark values.
type BzlExprValue interface {
	BzlExpr() bzl.Expr
}

// SelectStringListValue is a value that can be translated to a Bazel
// select expression that picks a string list based on a string condition.
type SelectStringListValue map[string][]string

func (s SelectStringListValue) BzlExpr() bzl.Expr {
	defaultKey := "//conditions:default"
	keys := make([]string, 0, len(s))
	haveDefaultKey := false
	for key := range s {
		if key == defaultKey {
			haveDefaultKey = true
		} else {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	if haveDefaultKey {
		keys = append(keys, defaultKey)
	}

	args := make([]bzl.Expr, 0, len(s))
	for _, key := range keys {
		value := ExprFromValue(s[key])
		if key != defaultKey {
			value.(*bzl.ListExpr).ForceMultiLine = true
		}
		args = append(args, &bzl.KeyValueExpr{
			Key:   &bzl.StringExpr{Value: key},
			Value: value,
		})
	}
	sel := &bzl.CallExpr{
		X:    &bzl.Ident{Name: "select"},
		List: []bzl.Expr{&bzl.DictExpr{List: args, ForceMultiLine: true}},
	}
	return sel
}

// ExprFromValue converts a value into an expression that can be written into
// a Bazel build file. The following types of values can be converted:
//
//   * bools, integers, floats, strings.
//   * slices, arrays (converted to lists).
//   * maps (converted to select expressions; keys must be rules in
//     @io_bazel_rules_go//go/platform).
//   * GlobValue (converted to glob expressions).
//   * PlatformStrings (converted to a concatenation of a list and selects).
//
// Converting unsupported types will cause a panic.
func ExprFromValue(val interface{}) bzl.Expr {
	if e, ok := val.(bzl.Expr); ok {
		return e
	}
	if be, ok := val.(BzlExprValue); ok {
		return be.BzlExpr()
	}

	rv := reflect.ValueOf(val)
	switch rv.Kind() {
	case reflect.Bool:
		tok := "False"
		if rv.Bool() {
			tok = "True"
		}
		return &bzl.LiteralExpr{Token: tok}

	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return &bzl.LiteralExpr{Token: fmt.Sprintf("%d", val)}

	case reflect.Float32, reflect.Float64:
		return &bzl.LiteralExpr{Token: fmt.Sprintf("%f", val)}

	case reflect.String:
		return &bzl.StringExpr{Value: val.(string)}

	case reflect.Slice, reflect.Array:
		var list []bzl.Expr
		for i := 0; i < rv.Len(); i++ {
			elem := ExprFromValue(rv.Index(i).Interface())
			list = append(list, elem)
		}
		return &bzl.ListExpr{List: list}

	case reflect.Map:
		rkeys := rv.MapKeys()
		sort.Sort(byString(rkeys))
		args := make([]bzl.Expr, len(rkeys))
		for i, rk := range rkeys {
			k := &bzl.StringExpr{Value: mapKeyString(rk)}
			v := ExprFromValue(rv.MapIndex(rk).Interface())
			if l, ok := v.(*bzl.ListExpr); ok {
				l.ForceMultiLine = true
			}
			args[i] = &bzl.KeyValueExpr{Key: k, Value: v}
		}
		return &bzl.DictExpr{List: args, ForceMultiLine: true}

	case reflect.Struct:
		switch val := val.(type) {
		case GlobValue:
			patternsValue := ExprFromValue(val.Patterns)
			globArgs := []bzl.Expr{patternsValue}
			if len(val.Excludes) > 0 {
				excludesValue := ExprFromValue(val.Excludes)
				globArgs = append(globArgs, &bzl.KeyValueExpr{
					Key:   &bzl.StringExpr{Value: "excludes"},
					Value: excludesValue,
				})
			}
			return &bzl.CallExpr{
				X:    &bzl.LiteralExpr{Token: "glob"},
				List: globArgs,
			}
		}
	}

	log.Panicf("type not supported: %T", val)
	return nil
}

func mapKeyString(k reflect.Value) string {
	switch s := k.Interface().(type) {
	case string:
		return s
	default:
		log.Panicf("unexpected map key: %v", k)
		return ""
	}
}

type byString []reflect.Value

var _ sort.Interface = byString{}

func (s byString) Len() int {
	return len(s)
}

func (s byString) Less(i, j int) bool {
	return mapKeyString(s[i]) < mapKeyString(s[j])
}

func (s byString) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
