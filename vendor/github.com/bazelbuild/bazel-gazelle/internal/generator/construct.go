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

package generator

import (
	"fmt"
	"log"
	"reflect"
	"sort"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
	"github.com/bazelbuild/bazel-gazelle/internal/packages"
	bf "github.com/bazelbuild/buildtools/build"
	bt "github.com/bazelbuild/buildtools/tables"
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

// EmptyRule generates an empty rule with the given kind and name.
func EmptyRule(kind, name string) *bf.CallExpr {
	return NewRule(kind, []KeyValue{{"name", name}})
}

// NewRule generates a rule of the given kind with the given attributes.
func NewRule(kind string, kwargs []KeyValue) *bf.CallExpr {
	sort.Sort(byAttrName(kwargs))

	var list []bf.Expr
	for _, arg := range kwargs {
		expr := newValue(arg.Value)
		list = append(list, &bf.BinaryExpr{
			X:  &bf.LiteralExpr{Token: arg.Key},
			Op: "=",
			Y:  expr,
		})
	}

	return &bf.CallExpr{
		X:    &bf.LiteralExpr{Token: kind},
		List: list,
	}
}

// newValue converts a Go value into the corresponding expression in Bazel BUILD file.
func newValue(val interface{}) bf.Expr {
	rv := reflect.ValueOf(val)
	switch rv.Kind() {
	case reflect.Bool:
		tok := "False"
		if rv.Bool() {
			tok = "True"
		}
		return &bf.LiteralExpr{Token: tok}

	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return &bf.LiteralExpr{Token: fmt.Sprintf("%d", val)}

	case reflect.Float32, reflect.Float64:
		return &bf.LiteralExpr{Token: fmt.Sprintf("%f", val)}

	case reflect.String:
		return &bf.StringExpr{Value: val.(string)}

	case reflect.Slice, reflect.Array:
		var list []bf.Expr
		for i := 0; i < rv.Len(); i++ {
			elem := newValue(rv.Index(i).Interface())
			list = append(list, elem)
		}
		return &bf.ListExpr{List: list}

	case reflect.Map:
		rkeys := rv.MapKeys()
		sort.Sort(byString(rkeys))
		args := make([]bf.Expr, len(rkeys))
		for i, rk := range rkeys {
			label := fmt.Sprintf("@%s//go/platform:%s", config.RulesGoRepoName, mapKeyString(rk))
			k := &bf.StringExpr{Value: label}
			v := newValue(rv.MapIndex(rk).Interface())
			if l, ok := v.(*bf.ListExpr); ok {
				l.ForceMultiLine = true
			}
			args[i] = &bf.KeyValueExpr{Key: k, Value: v}
		}
		args = append(args, &bf.KeyValueExpr{
			Key:   &bf.StringExpr{Value: "//conditions:default"},
			Value: &bf.ListExpr{},
		})
		sel := &bf.CallExpr{
			X:    &bf.LiteralExpr{Token: "select"},
			List: []bf.Expr{&bf.DictExpr{List: args, ForceMultiLine: true}},
		}
		return sel

	case reflect.Struct:
		switch val := val.(type) {
		case GlobValue:
			patternsValue := newValue(val.Patterns)
			globArgs := []bf.Expr{patternsValue}
			if len(val.Excludes) > 0 {
				excludesValue := newValue(val.Excludes)
				globArgs = append(globArgs, &bf.KeyValueExpr{
					Key:   &bf.StringExpr{Value: "excludes"},
					Value: excludesValue,
				})
			}
			return &bf.CallExpr{
				X:    &bf.LiteralExpr{Token: "glob"},
				List: globArgs,
			}

		case packages.PlatformStrings:
			var pieces []bf.Expr
			if len(val.Generic) > 0 {
				pieces = append(pieces, newValue(val.Generic))
			}
			if len(val.OS) > 0 {
				pieces = append(pieces, newValue(val.OS))
			}
			if len(val.Arch) > 0 {
				pieces = append(pieces, newValue(val.Arch))
			}
			if len(val.Platform) > 0 {
				pieces = append(pieces, newValue(val.Platform))
			}
			if len(pieces) == 0 {
				return &bf.ListExpr{}
			} else if len(pieces) == 1 {
				return pieces[0]
			} else {
				e := pieces[0]
				if list, ok := e.(*bf.ListExpr); ok {
					list.ForceMultiLine = true
				}
				for _, piece := range pieces[1:] {
					e = &bf.BinaryExpr{X: e, Y: piece, Op: "+"}
				}
				return e
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
	case config.Platform:
		return s.String()
	default:
		log.Panicf("unexpected map key: %v", k)
		return ""
	}
}

type byAttrName []KeyValue

var _ sort.Interface = byAttrName{}

func (s byAttrName) Len() int {
	return len(s)
}

func (s byAttrName) Less(i, j int) bool {
	if cmp := bt.NamePriority[s[i].Key] - bt.NamePriority[s[j].Key]; cmp != 0 {
		return cmp < 0
	}
	return s[i].Key < s[j].Key
}

func (s byAttrName) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
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
