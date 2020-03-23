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
)

// A Matcher is a representation of a class of values.
// It is used to represent the valid or expected arguments to a mocked method.
type Matcher interface {
	// Matches returns whether x is a match.
	Matches(x interface{}) bool

	// String describes what the matcher matches.
	String() string
}

type anyMatcher struct{}

func (anyMatcher) Matches(x interface{}) bool {
	return true
}

func (anyMatcher) String() string {
	return "is anything"
}

type eqMatcher struct {
	x interface{}
}

func (e eqMatcher) Matches(x interface{}) bool {
	return reflect.DeepEqual(e.x, x)
}

func (e eqMatcher) String() string {
	return fmt.Sprintf("is equal to %v", e.x)
}

type nilMatcher struct{}

func (nilMatcher) Matches(x interface{}) bool {
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

func (n notMatcher) Matches(x interface{}) bool {
	return !n.m.Matches(x)
}

func (n notMatcher) String() string {
	// TODO: Improve this if we add a NotString method to the Matcher interface.
	return "not(" + n.m.String() + ")"
}

type assignableToTypeOfMatcher struct {
	targetType reflect.Type
}

func (m assignableToTypeOfMatcher) Matches(x interface{}) bool {
	return reflect.TypeOf(x).AssignableTo(m.targetType)
}

func (m assignableToTypeOfMatcher) String() string {
	return "is assignable to " + m.targetType.Name()
}

// Constructors
// Any returns a matcher that always matches.
func Any() Matcher { return anyMatcher{} }

// Eq returns a matcher that matches on equality.
//
// Example usage:
//   Eq(5).Matches(5) // returns true
//   Eq(5).Matches(4) // returns false
func Eq(x interface{}) Matcher { return eqMatcher{x} }

// Nil returns a matcher that matches if the received value is nil.
//
// Example usage:
//   var x *bytes.Buffer
//   Nil().Matches(x) // returns true
//   x = &bytes.Buffer{}
//   Nil().Matches(x) // returns false
func Nil() Matcher { return nilMatcher{} }

// Not reverses the results of its given child matcher.
//
// Example usage:
//   Not(Eq(5)).Matches(4) // returns true
//   Not(Eq(5)).Matches(5) // returns false
func Not(x interface{}) Matcher {
	if m, ok := x.(Matcher); ok {
		return notMatcher{m}
	}
	return notMatcher{Eq(x)}
}

// AssignableToTypeOf is a Matcher that matches if the parameter to the mock
// function is assignable to the type of the parameter to this function.
//
// Example usage:
//   var s fmt.Stringer = &bytes.Buffer{}
//   AssignableToTypeOf(s).Matches(time.Second) // returns true
//   AssignableToTypeOf(s).Matches(99) // returns false
func AssignableToTypeOf(x interface{}) Matcher {
	return assignableToTypeOfMatcher{reflect.TypeOf(x)}
}
