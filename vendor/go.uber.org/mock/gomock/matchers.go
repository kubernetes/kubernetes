// Copyright 2010 Google Inc.
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

package gomock

import (
	"fmt"
	"reflect"
	"regexp"
	"strings"
)

// A Matcher is a representation of a class of values.
// It is used to represent the valid or expected arguments to a mocked method.
type Matcher interface {
	// Matches returns whether x is a match.
	Matches(x any) bool

	// String describes what the matcher matches.
	String() string
}

// WantFormatter modifies the given Matcher's String() method to the given
// Stringer. This allows for control on how the "Want" is formatted when
// printing .
func WantFormatter(s fmt.Stringer, m Matcher) Matcher {
	type matcher interface {
		Matches(x any) bool
	}

	return struct {
		matcher
		fmt.Stringer
	}{
		matcher:  m,
		Stringer: s,
	}
}

// StringerFunc type is an adapter to allow the use of ordinary functions as
// a Stringer. If f is a function with the appropriate signature,
// StringerFunc(f) is a Stringer that calls f.
type StringerFunc func() string

// String implements fmt.Stringer.
func (f StringerFunc) String() string {
	return f()
}

// GotFormatter is used to better print failure messages. If a matcher
// implements GotFormatter, it will use the result from Got when printing
// the failure message.
type GotFormatter interface {
	// Got is invoked with the received value. The result is used when
	// printing the failure message.
	Got(got any) string
}

// GotFormatterFunc type is an adapter to allow the use of ordinary
// functions as a GotFormatter. If f is a function with the appropriate
// signature, GotFormatterFunc(f) is a GotFormatter that calls f.
type GotFormatterFunc func(got any) string

// Got implements GotFormatter.
func (f GotFormatterFunc) Got(got any) string {
	return f(got)
}

// GotFormatterAdapter attaches a GotFormatter to a Matcher.
func GotFormatterAdapter(s GotFormatter, m Matcher) Matcher {
	return struct {
		GotFormatter
		Matcher
	}{
		GotFormatter: s,
		Matcher:      m,
	}
}

type anyMatcher struct{}

func (anyMatcher) Matches(any) bool {
	return true
}

func (anyMatcher) String() string {
	return "is anything"
}

type condMatcher struct {
	fn func(x any) bool
}

func (c condMatcher) Matches(x any) bool {
	return c.fn(x)
}

func (condMatcher) String() string {
	return "adheres to a custom condition"
}

type eqMatcher struct {
	x any
}

func (e eqMatcher) Matches(x any) bool {
	// In case, some value is nil
	if e.x == nil || x == nil {
		return reflect.DeepEqual(e.x, x)
	}

	// Check if types assignable and convert them to common type
	x1Val := reflect.ValueOf(e.x)
	x2Val := reflect.ValueOf(x)

	if x1Val.Type().AssignableTo(x2Val.Type()) {
		x1ValConverted := x1Val.Convert(x2Val.Type())
		return reflect.DeepEqual(x1ValConverted.Interface(), x2Val.Interface())
	}

	return false
}

func (e eqMatcher) String() string {
	return fmt.Sprintf("is equal to %v (%T)", e.x, e.x)
}

type nilMatcher struct{}

func (nilMatcher) Matches(x any) bool {
	if x == nil {
		return true
	}

	v := reflect.ValueOf(x)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map,
		reflect.Ptr, reflect.Slice:
		return v.IsNil()
	}

	return false
}

func (nilMatcher) String() string {
	return "is nil"
}

type notMatcher struct {
	m Matcher
}

func (n notMatcher) Matches(x any) bool {
	return !n.m.Matches(x)
}

func (n notMatcher) String() string {
	return "not(" + n.m.String() + ")"
}

type regexMatcher struct {
	regex *regexp.Regexp
}

func (m regexMatcher) Matches(x any) bool {
	switch t := x.(type) {
	case string:
		return m.regex.MatchString(t)
	case []byte:
		return m.regex.Match(t)
	default:
		return false
	}
}

func (m regexMatcher) String() string {
	return "matches regex " + m.regex.String()
}

type assignableToTypeOfMatcher struct {
	targetType reflect.Type
}

func (m assignableToTypeOfMatcher) Matches(x any) bool {
	return reflect.TypeOf(x).AssignableTo(m.targetType)
}

func (m assignableToTypeOfMatcher) String() string {
	return "is assignable to " + m.targetType.Name()
}

type anyOfMatcher struct {
	matchers []Matcher
}

func (am anyOfMatcher) Matches(x any) bool {
	for _, m := range am.matchers {
		if m.Matches(x) {
			return true
		}
	}
	return false
}

func (am anyOfMatcher) String() string {
	ss := make([]string, 0, len(am.matchers))
	for _, matcher := range am.matchers {
		ss = append(ss, matcher.String())
	}
	return strings.Join(ss, " | ")
}

type allMatcher struct {
	matchers []Matcher
}

func (am allMatcher) Matches(x any) bool {
	for _, m := range am.matchers {
		if !m.Matches(x) {
			return false
		}
	}
	return true
}

func (am allMatcher) String() string {
	ss := make([]string, 0, len(am.matchers))
	for _, matcher := range am.matchers {
		ss = append(ss, matcher.String())
	}
	return strings.Join(ss, "; ")
}

type lenMatcher struct {
	i int
}

func (m lenMatcher) Matches(x any) bool {
	v := reflect.ValueOf(x)
	switch v.Kind() {
	case reflect.Array, reflect.Chan, reflect.Map, reflect.Slice, reflect.String:
		return v.Len() == m.i
	default:
		return false
	}
}

func (m lenMatcher) String() string {
	return fmt.Sprintf("has length %d", m.i)
}

type inAnyOrderMatcher struct {
	x any
}

func (m inAnyOrderMatcher) Matches(x any) bool {
	given, ok := m.prepareValue(x)
	if !ok {
		return false
	}
	wanted, ok := m.prepareValue(m.x)
	if !ok {
		return false
	}

	if given.Len() != wanted.Len() {
		return false
	}

	usedFromGiven := make([]bool, given.Len())
	foundFromWanted := make([]bool, wanted.Len())
	for i := 0; i < wanted.Len(); i++ {
		wantedMatcher := Eq(wanted.Index(i).Interface())
		for j := 0; j < given.Len(); j++ {
			if usedFromGiven[j] {
				continue
			}
			if wantedMatcher.Matches(given.Index(j).Interface()) {
				foundFromWanted[i] = true
				usedFromGiven[j] = true
				break
			}
		}
	}

	missingFromWanted := 0
	for _, found := range foundFromWanted {
		if !found {
			missingFromWanted++
		}
	}
	extraInGiven := 0
	for _, used := range usedFromGiven {
		if !used {
			extraInGiven++
		}
	}

	return extraInGiven == 0 && missingFromWanted == 0
}

func (m inAnyOrderMatcher) prepareValue(x any) (reflect.Value, bool) {
	xValue := reflect.ValueOf(x)
	switch xValue.Kind() {
	case reflect.Slice, reflect.Array:
		return xValue, true
	default:
		return reflect.Value{}, false
	}
}

func (m inAnyOrderMatcher) String() string {
	return fmt.Sprintf("has the same elements as %v", m.x)
}

// Constructors

// All returns a composite Matcher that returns true if and only all of the
// matchers return true.
func All(ms ...Matcher) Matcher { return allMatcher{ms} }

// Any returns a matcher that always matches.
func Any() Matcher { return anyMatcher{} }

// Cond returns a matcher that matches when the given function returns true
// after passing it the parameter to the mock function.
// This is particularly useful in case you want to match over a field of a custom struct, or dynamic logic.
//
// Example usage:
//
//	Cond(func(x any){return x.(int) == 1}).Matches(1) // returns true
//	Cond(func(x any){return x.(int) == 2}).Matches(1) // returns false
func Cond(fn func(x any) bool) Matcher { return condMatcher{fn} }

// AnyOf returns a composite Matcher that returns true if at least one of the
// matchers returns true.
//
// Example usage:
//
//	AnyOf(1, 2, 3).Matches(2) // returns true
//	AnyOf(1, 2, 3).Matches(10) // returns false
//	AnyOf(Nil(), Len(2)).Matches(nil) // returns true
//	AnyOf(Nil(), Len(2)).Matches("hi") // returns true
//	AnyOf(Nil(), Len(2)).Matches("hello") // returns false
func AnyOf(xs ...any) Matcher {
	ms := make([]Matcher, 0, len(xs))
	for _, x := range xs {
		if m, ok := x.(Matcher); ok {
			ms = append(ms, m)
		} else {
			ms = append(ms, Eq(x))
		}
	}
	return anyOfMatcher{ms}
}

// Eq returns a matcher that matches on equality.
//
// Example usage:
//
//	Eq(5).Matches(5) // returns true
//	Eq(5).Matches(4) // returns false
func Eq(x any) Matcher { return eqMatcher{x} }

// Len returns a matcher that matches on length. This matcher returns false if
// is compared to a type that is not an array, chan, map, slice, or string.
func Len(i int) Matcher {
	return lenMatcher{i}
}

// Nil returns a matcher that matches if the received value is nil.
//
// Example usage:
//
//	var x *bytes.Buffer
//	Nil().Matches(x) // returns true
//	x = &bytes.Buffer{}
//	Nil().Matches(x) // returns false
func Nil() Matcher { return nilMatcher{} }

// Not reverses the results of its given child matcher.
//
// Example usage:
//
//	Not(Eq(5)).Matches(4) // returns true
//	Not(Eq(5)).Matches(5) // returns false
func Not(x any) Matcher {
	if m, ok := x.(Matcher); ok {
		return notMatcher{m}
	}
	return notMatcher{Eq(x)}
}

// Regex checks whether parameter matches the associated regex.
//
// Example usage:
//
//	Regex("[0-9]{2}:[0-9]{2}").Matches("23:02") // returns true
//	Regex("[0-9]{2}:[0-9]{2}").Matches([]byte{'2', '3', ':', '0', '2'}) // returns true
//	Regex("[0-9]{2}:[0-9]{2}").Matches("hello world") // returns false
//	Regex("[0-9]{2}").Matches(21) // returns false as it's not a valid type
func Regex(regexStr string) Matcher {
	return regexMatcher{regex: regexp.MustCompile(regexStr)}
}

// AssignableToTypeOf is a Matcher that matches if the parameter to the mock
// function is assignable to the type of the parameter to this function.
//
// Example usage:
//
//	var s fmt.Stringer = &bytes.Buffer{}
//	AssignableToTypeOf(s).Matches(time.Second) // returns true
//	AssignableToTypeOf(s).Matches(99) // returns false
//
//	var ctx = reflect.TypeOf((*context.Context)(nil)).Elem()
//	AssignableToTypeOf(ctx).Matches(context.Background()) // returns true
func AssignableToTypeOf(x any) Matcher {
	if xt, ok := x.(reflect.Type); ok {
		return assignableToTypeOfMatcher{xt}
	}
	return assignableToTypeOfMatcher{reflect.TypeOf(x)}
}

// InAnyOrder is a Matcher that returns true for collections of the same elements ignoring the order.
//
// Example usage:
//
//	InAnyOrder([]int{1, 2, 3}).Matches([]int{1, 3, 2}) // returns true
//	InAnyOrder([]int{1, 2, 3}).Matches([]int{1, 2}) // returns false
func InAnyOrder(x any) Matcher {
	return inAnyOrderMatcher{x}
}
