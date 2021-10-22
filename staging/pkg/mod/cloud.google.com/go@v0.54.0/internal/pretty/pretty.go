// Copyright 2016 Google LLC
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

// Package pretty implements a simple pretty-printer. It is intended for
// debugging the output of tests.
//
// It follows pointers and produces multi-line output for complex values like
// slices, maps and structs.
package pretty

import (
	"fmt"
	"io"
	"reflect"
	"sort"
	"strings"
	"time"
)

// Indent is the string output at each level of indentation.
var Indent = "    "

// Value returns a value that will print prettily when used as an
// argument for the %v or %s format specifiers.
// With no flags, struct fields and map keys with default values are omitted.
// With the '+' or '#' flags, all values are displayed.
//
// This package does not detect cycles. Attempting to print a Value that
// contains cycles will result in unbounded recursion.
func Value(v interface{}) val { return val{v: v} }

// val is a value.
type val struct{ v interface{} }

// Format implements the fmt.Formatter interface.
func (v val) Format(s fmt.State, c rune) {
	if c == 'v' || c == 's' {
		fprint(s, reflect.ValueOf(v.v), state{
			defaults: s.Flag('+') || s.Flag('#'),
		})
	} else {
		fmt.Fprintf(s, "%%!%c(pretty.val)", c)
	}
}

type state struct {
	level          int
	prefix, suffix string
	defaults       bool
}

const maxLevel = 100

var typeOfTime = reflect.TypeOf(time.Time{})

func fprint(w io.Writer, v reflect.Value, s state) {
	if s.level > maxLevel {
		fmt.Fprintln(w, "pretty: max nested depth exceeded")
		return
	}
	indent := strings.Repeat(Indent, s.level)
	fmt.Fprintf(w, "%s%s", indent, s.prefix)
	if isNil(v) {
		fmt.Fprintf(w, "nil%s", s.suffix)
		return
	}
	if v.Type().Kind() == reflect.Interface {
		v = v.Elem()
	}
	if v.Type() == typeOfTime {
		fmt.Fprintf(w, "%s%s", v.Interface(), s.suffix)
		return
	}
	for v.Type().Kind() == reflect.Ptr {
		fmt.Fprintf(w, "&")
		v = v.Elem()
	}
	switch v.Type().Kind() {
	default:
		fmt.Fprintf(w, "%s%s", short(v), s.suffix)

	case reflect.Array:
		fmt.Fprintf(w, "%s{\n", v.Type())
		for i := 0; i < v.Len(); i++ {
			fprint(w, v.Index(i), state{
				level:    s.level + 1,
				prefix:   "",
				suffix:   ",",
				defaults: s.defaults,
			})
			fmt.Fprintln(w)
		}
		fmt.Fprintf(w, "%s}", indent)

	case reflect.Slice:
		fmt.Fprintf(w, "%s{", v.Type())
		if v.Len() > 0 {
			fmt.Fprintln(w)
			for i := 0; i < v.Len(); i++ {
				fprint(w, v.Index(i), state{
					level:    s.level + 1,
					prefix:   "",
					suffix:   ",",
					defaults: s.defaults,
				})
				fmt.Fprintln(w)
			}
		}
		fmt.Fprintf(w, "%s}%s", indent, s.suffix)

	case reflect.Map:
		fmt.Fprintf(w, "%s{", v.Type())
		if v.Len() > 0 {
			fmt.Fprintln(w)
			keys := v.MapKeys()
			maybeSort(keys, v.Type().Key())
			for _, key := range keys {
				val := v.MapIndex(key)
				if s.defaults || !isDefault(val) {
					fprint(w, val, state{
						level:    s.level + 1,
						prefix:   short(key) + ": ",
						suffix:   ",",
						defaults: s.defaults,
					})
					fmt.Fprintln(w)
				}
			}
		}
		fmt.Fprintf(w, "%s}%s", indent, s.suffix)

	case reflect.Struct:
		t := v.Type()
		fmt.Fprintf(w, "%s{\n", t)
		for i := 0; i < t.NumField(); i++ {
			f := v.Field(i)
			if s.defaults || !isDefault(f) {
				fprint(w, f, state{
					level:    s.level + 1,
					prefix:   t.Field(i).Name + ": ",
					suffix:   ",",
					defaults: s.defaults,
				})
				fmt.Fprintln(w)
			}
		}
		fmt.Fprintf(w, "%s}%s", indent, s.suffix)
	}
}

func isNil(v reflect.Value) bool {
	if !v.IsValid() {
		return true
	}
	switch v.Type().Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		return v.IsNil()
	default:
		return false
	}
}

func isDefault(v reflect.Value) bool {
	if !v.IsValid() {
		return true
	}
	t := v.Type()
	switch t.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		return v.IsNil()
	default:
		if !v.CanInterface() {
			return false
		}
		return t.Comparable() && v.Interface() == reflect.Zero(t).Interface()
	}
}

// short returns a short, one-line string for v.
func short(v reflect.Value) string {
	if !v.IsValid() {
		return "nil"
	}
	if v.Type().Kind() == reflect.String {
		return fmt.Sprintf("%q", v)
	}
	return fmt.Sprintf("%v", v)
}

func maybeSort(vs []reflect.Value, t reflect.Type) {
	if less := lessFunc(t); less != nil {
		sort.Sort(&sorter{vs, less})
	}
}

// lessFunc returns a function that implements the "<" operator
// for the given type, or nil if the type doesn't support "<" .
func lessFunc(t reflect.Type) func(v1, v2 interface{}) bool {
	switch t.Kind() {
	case reflect.String:
		return func(v1, v2 interface{}) bool { return v1.(string) < v2.(string) }
	case reflect.Int:
		return func(v1, v2 interface{}) bool { return v1.(int) < v2.(int) }
	case reflect.Int8:
		return func(v1, v2 interface{}) bool { return v1.(int8) < v2.(int8) }
	case reflect.Int16:
		return func(v1, v2 interface{}) bool { return v1.(int16) < v2.(int16) }
	case reflect.Int32:
		return func(v1, v2 interface{}) bool { return v1.(int32) < v2.(int32) }
	case reflect.Int64:
		return func(v1, v2 interface{}) bool { return v1.(int64) < v2.(int64) }
	case reflect.Uint:
		return func(v1, v2 interface{}) bool { return v1.(uint) < v2.(uint) }
	case reflect.Uint8:
		return func(v1, v2 interface{}) bool { return v1.(uint8) < v2.(uint8) }
	case reflect.Uint16:
		return func(v1, v2 interface{}) bool { return v1.(uint16) < v2.(uint16) }
	case reflect.Uint32:
		return func(v1, v2 interface{}) bool { return v1.(uint32) < v2.(uint32) }
	case reflect.Uint64:
		return func(v1, v2 interface{}) bool { return v1.(uint64) < v2.(uint64) }
	case reflect.Float32:
		return func(v1, v2 interface{}) bool { return v1.(float32) < v2.(float32) }
	case reflect.Float64:
		return func(v1, v2 interface{}) bool { return v1.(float64) < v2.(float64) }
	default:
		return nil
	}
}

type sorter struct {
	vs   []reflect.Value
	less func(v1, v2 interface{}) bool
}

func (s *sorter) Len() int           { return len(s.vs) }
func (s *sorter) Swap(i, j int)      { s.vs[i], s.vs[j] = s.vs[j], s.vs[i] }
func (s *sorter) Less(i, j int) bool { return s.less(s.vs[i].Interface(), s.vs[j].Interface()) }
