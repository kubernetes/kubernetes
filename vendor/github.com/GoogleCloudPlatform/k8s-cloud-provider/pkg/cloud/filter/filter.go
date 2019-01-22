/*
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package filter encapsulates the filter argument to compute API calls.
//
//  // List all global addresses (no filter).
//  c.GlobalAddresses().List(ctx, filter.None)
//
//  // List global addresses filtering for name matching "abc.*".
//  c.GlobalAddresses().List(ctx, filter.Regexp("name", "abc.*"))
//
//  // List on multiple conditions.
//  f := filter.Regexp("name", "homer.*").AndNotRegexp("name", "homers")
//  c.GlobalAddresses().List(ctx, f)
package filter

import (
	"errors"
	"fmt"
	"reflect"
	"regexp"
	"strings"

	"k8s.io/klog"
)

var (
	// None indicates that the List result set should not be filter (i.e.
	// return all values).
	None *F
)

// Regexp returns a filter for fieldName matches regexp v.
func Regexp(fieldName, v string) *F {
	return (&F{}).AndRegexp(fieldName, v)
}

// NotRegexp returns a filter for fieldName not matches regexp v.
func NotRegexp(fieldName, v string) *F {
	return (&F{}).AndNotRegexp(fieldName, v)
}

// EqualInt returns a filter for fieldName ~ v.
func EqualInt(fieldName string, v int) *F {
	return (&F{}).AndEqualInt(fieldName, v)
}

// NotEqualInt returns a filter for fieldName != v.
func NotEqualInt(fieldName string, v int) *F {
	return (&F{}).AndNotEqualInt(fieldName, v)
}

// EqualBool returns a filter for fieldName == v.
func EqualBool(fieldName string, v bool) *F {
	return (&F{}).AndEqualBool(fieldName, v)
}

// NotEqualBool returns a filter for fieldName != v.
func NotEqualBool(fieldName string, v bool) *F {
	return (&F{}).AndNotEqualBool(fieldName, v)
}

// F is a filter to be used with List() operations.
//
// From the compute API description:
//
// Sets a filter {expression} for filtering listed resources. Your {expression}
// must be in the format: field_name comparison_string literal_string.
//
// The field_name is the name of the field you want to compare. Only atomic field
// types are supported (string, number, boolean). The comparison_string must be
// either eq (equals) or ne (not equals). The literal_string is the string value
// to filter to. The literal value must be valid for the type of field you are
// filtering by (string, number, boolean). For string fields, the literal value is
// interpreted as a regular expression using RE2 syntax. The literal value must
// match the entire field.
//
// For example, to filter for instances that do not have a name of
// example-instance, you would use name ne example-instance.
//
// You can filter on nested fields. For example, you could filter on instances
// that have set the scheduling.automaticRestart field to true. Use filtering on
// nested fields to take advantage of labels to organize and search for results
// based on label values.
//
// To filter on multiple expressions, provide each separate expression within
// parentheses. For example, (scheduling.automaticRestart eq true)
// (zone eq us-central1-f). Multiple expressions are treated as AND expressions,
// meaning that resources must match all expressions to pass the filters.
type F struct {
	predicates []filterPredicate
}

// And joins two filters together.
func (fl *F) And(rest *F) *F {
	fl.predicates = append(fl.predicates, rest.predicates...)
	return fl
}

// AndRegexp adds a field match string predicate.
func (fl *F) AndRegexp(fieldName, v string) *F {
	fl.predicates = append(fl.predicates, filterPredicate{fieldName: fieldName, op: equals, s: &v})
	return fl
}

// AndNotRegexp adds a field not match string predicate.
func (fl *F) AndNotRegexp(fieldName, v string) *F {
	fl.predicates = append(fl.predicates, filterPredicate{fieldName: fieldName, op: notEquals, s: &v})
	return fl
}

// AndEqualInt adds a field == int predicate.
func (fl *F) AndEqualInt(fieldName string, v int) *F {
	fl.predicates = append(fl.predicates, filterPredicate{fieldName: fieldName, op: equals, i: &v})
	return fl
}

// AndNotEqualInt adds a field != int predicate.
func (fl *F) AndNotEqualInt(fieldName string, v int) *F {
	fl.predicates = append(fl.predicates, filterPredicate{fieldName: fieldName, op: notEquals, i: &v})
	return fl
}

// AndEqualBool adds a field == bool predicate.
func (fl *F) AndEqualBool(fieldName string, v bool) *F {
	fl.predicates = append(fl.predicates, filterPredicate{fieldName: fieldName, op: equals, b: &v})
	return fl
}

// AndNotEqualBool adds a field != bool predicate.
func (fl *F) AndNotEqualBool(fieldName string, v bool) *F {
	fl.predicates = append(fl.predicates, filterPredicate{fieldName: fieldName, op: notEquals, b: &v})
	return fl
}

func (fl *F) String() string {
	if len(fl.predicates) == 1 {
		return fl.predicates[0].String()
	}

	var pl []string
	for _, p := range fl.predicates {
		pl = append(pl, "("+p.String()+")")
	}
	return strings.Join(pl, " ")
}

// Match returns true if the F as specifies matches the given object. This
// is used by the Mock implementations to perform filtering and SHOULD NOT be
// used in production code as it is not well-tested to be equivalent to the
// actual compute API.
func (fl *F) Match(obj interface{}) bool {
	if fl == nil {
		return true
	}
	for _, p := range fl.predicates {
		if !p.match(obj) {
			return false
		}
	}
	return true
}

type filterOp int

const (
	equals    filterOp = iota
	notEquals filterOp = iota
)

// filterPredicate is an individual predicate for a fieldName and value.
type filterPredicate struct {
	fieldName string

	op filterOp
	s  *string
	i  *int
	b  *bool
}

func (fp *filterPredicate) String() string {
	var op string
	switch fp.op {
	case equals:
		op = "eq"
	case notEquals:
		op = "ne"
	default:
		op = "invalidOp"
	}

	var value string
	switch {
	case fp.s != nil:
		// There does not seem to be any sort of escaping as specified in the
		// document. This means it's possible to create malformed expressions.
		value = *fp.s
	case fp.i != nil:
		value = fmt.Sprintf("%d", *fp.i)
	case fp.b != nil:
		value = fmt.Sprintf("%t", *fp.b)
	default:
		value = "invalidValue"
	}

	return fmt.Sprintf("%s %s %s", fp.fieldName, op, value)
}

func (fp *filterPredicate) match(o interface{}) bool {
	v, err := extractValue(fp.fieldName, o)
	klog.V(6).Infof("extractValue(%q, %#v) = %v, %v", fp.fieldName, o, v, err)
	if err != nil {
		return false
	}

	var match bool
	switch x := v.(type) {
	case string:
		if fp.s == nil {
			return false
		}
		re, err := regexp.Compile(*fp.s)
		if err != nil {
			klog.Errorf("Match regexp %q is invalid: %v", *fp.s, err)
			return false
		}
		match = re.Match([]byte(x))
	case int:
		if fp.i == nil {
			return false
		}
		match = x == *fp.i
	case bool:
		if fp.b == nil {
			return false
		}
		match = x == *fp.b
	}

	switch fp.op {
	case equals:
		return match
	case notEquals:
		return !match
	}

	return false
}

// snakeToCamelCase converts from "names_like_this" to "NamesLikeThis" to
// interoperate between proto and Golang naming conventions.
func snakeToCamelCase(s string) string {
	parts := strings.Split(s, "_")
	var ret string
	for _, x := range parts {
		ret += strings.Title(x)
	}
	return ret
}

// extractValue returns the value of the field named by path in object o if it exists.
func extractValue(path string, o interface{}) (interface{}, error) {
	parts := strings.Split(path, ".")
	for _, f := range parts {
		v := reflect.ValueOf(o)
		// Dereference Ptr to handle *struct.
		if v.Kind() == reflect.Ptr {
			if v.IsNil() {
				return nil, errors.New("field is nil")
			}
			v = v.Elem()
		}
		if v.Kind() != reflect.Struct {
			return nil, fmt.Errorf("cannot get field from non-struct (%T)", o)
		}
		v = v.FieldByName(snakeToCamelCase(f))
		if !v.IsValid() {
			return nil, fmt.Errorf("cannot get field %q as it is not a valid field in %T", f, o)
		}
		if !v.CanInterface() {
			return nil, fmt.Errorf("cannot get field %q in obj of type %T", f, o)
		}
		o = v.Interface()
	}
	switch o.(type) {
	case string, int, bool:
		return o, nil
	}
	return nil, fmt.Errorf("unhandled object of type %T", o)
}
