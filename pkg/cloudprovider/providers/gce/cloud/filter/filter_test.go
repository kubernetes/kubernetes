/*
Copyright 2017 The Kubernetes Authors.

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

package filter

import (
	"reflect"
	"testing"
)

func TestFilterToString(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		f    *F
		want string
	}{
		{Regexp("field1", "abc"), `field1 eq abc`},
		{NotRegexp("field1", "abc"), `field1 ne abc`},
		{EqualInt("field1", 13), "field1 eq 13"},
		{NotEqualInt("field1", 13), "field1 ne 13"},
		{EqualBool("field1", true), "field1 eq true"},
		{NotEqualBool("field1", true), "field1 ne true"},
		{Regexp("field1", "abc").AndRegexp("field2", "def"), `(field1 eq abc) (field2 eq def)`},
		{Regexp("field1", "abc").AndNotEqualInt("field2", 17), `(field1 eq abc) (field2 ne 17)`},
		{Regexp("field1", "abc").And(EqualInt("field2", 17)), `(field1 eq abc) (field2 eq 17)`},
	} {
		if tc.f.String() != tc.want {
			t.Errorf("filter %#v String() = %q, want %q", tc.f, tc.f.String(), tc.want)
		}
	}
}

func TestFilterMatch(t *testing.T) {
	t.Parallel()

	type inner struct {
		X string
	}
	type S struct {
		S           string
		I           int
		B           bool
		Unhandled   struct{}
		NestedField *inner
	}

	for _, tc := range []struct {
		f    *F
		o    interface{}
		want bool
	}{
		{f: None, o: &S{}, want: true},
		{f: Regexp("s", "abc"), o: &S{}},
		{f: EqualInt("i", 10), o: &S{}},
		{f: EqualBool("b", true), o: &S{}},
		{f: NotRegexp("s", "abc"), o: &S{}, want: true},
		{f: NotEqualInt("i", 10), o: &S{}, want: true},
		{f: NotEqualBool("b", true), o: &S{}, want: true},
		{f: Regexp("s", "abc").AndEqualBool("b", true), o: &S{}},
		{f: Regexp("s", "abc"), o: &S{S: "abc"}, want: true},
		{f: Regexp("s", "a.*"), o: &S{S: "abc"}, want: true},
		{f: Regexp("s", "a((("), o: &S{S: "abc"}},
		{f: NotRegexp("s", "abc"), o: &S{S: "abc"}},
		{f: EqualInt("i", 10), o: &S{I: 11}},
		{f: EqualInt("i", 10), o: &S{I: 10}, want: true},
		{f: Regexp("s", "abc").AndEqualBool("b", true), o: &S{S: "abc"}},
		{f: Regexp("s", "abcd").AndEqualBool("b", true), o: &S{S: "abc"}},
		{f: Regexp("s", "abc").AndEqualBool("b", true), o: &S{S: "abc", B: true}, want: true},
		{f: Regexp("s", "abc").And(EqualBool("b", true)), o: &S{S: "abc", B: true}, want: true},
		{f: Regexp("unhandled", "xyz"), o: &S{}},
		{f: Regexp("nested_field.x", "xyz"), o: &S{}},
		{f: Regexp("nested_field.x", "xyz"), o: &S{NestedField: &inner{"xyz"}}, want: true},
		{f: NotRegexp("nested_field.x", "xyz"), o: &S{NestedField: &inner{"xyz"}}},
		{f: Regexp("nested_field.y", "xyz"), o: &S{NestedField: &inner{"xyz"}}},
		{f: Regexp("nested_field", "xyz"), o: &S{NestedField: &inner{"xyz"}}},
	} {
		got := tc.f.Match(tc.o)
		if got != tc.want {
			t.Errorf("%v: Match(%+v) = %v, want %v", tc.f, tc.o, got, tc.want)
		}
	}
}

func TestFilterSnakeToCamelCase(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		s    string
		want string
	}{
		{"", ""},
		{"abc", "Abc"},
		{"_foo", "Foo"},
		{"a_b_c", "ABC"},
		{"a_BC_def", "ABCDef"},
		{"a_Bc_def", "ABcDef"},
	} {
		got := snakeToCamelCase(tc.s)
		if got != tc.want {
			t.Errorf("snakeToCamelCase(%q) = %q, want %q", tc.s, got, tc.want)
		}
	}
}

func TestFilterExtractValue(t *testing.T) {
	t.Parallel()

	type nest2 struct {
		Y string
	}
	type nest struct {
		X     string
		Nest2 nest2
	}
	st := &struct {
		S       string
		I       int
		F       bool
		Nest    nest
		NestPtr *nest

		Unhandled float64
	}{
		"abc",
		13,
		true,
		nest{"xyz", nest2{"zzz"}},
		&nest{"yyy", nest2{}},
		0.0,
	}

	for _, tc := range []struct {
		path    string
		o       interface{}
		want    interface{}
		wantErr bool
	}{
		{path: "s", o: st, want: "abc"},
		{path: "i", o: st, want: 13},
		{path: "f", o: st, want: true},
		{path: "nest.x", o: st, want: "xyz"},
		{path: "nest_ptr.x", o: st, want: "yyy"},
		// Error cases.
		{path: "", o: st, wantErr: true},
		{path: "no_such_field", o: st, wantErr: true},
		{path: "s.invalid_type", o: st, wantErr: true},
		{path: "unhandled", o: st, wantErr: true},
		{path: "nest.x", o: &struct{ Nest *nest }{}, wantErr: true},
	} {
		o, err := extractValue(tc.path, tc.o)
		gotErr := err != nil
		if gotErr != tc.wantErr {
			t.Errorf("extractValue(%v, %+v) = %v, %v; gotErr = %v, tc.wantErr = %v", tc.path, tc.o, o, err, gotErr, tc.wantErr)
		}
		if err != nil {
			continue
		}
		if !reflect.DeepEqual(o, tc.want) {
			t.Errorf("extractValue(%v, %+v) = %v, nil; want %v, nil", tc.path, tc.o, o, tc.want)
		}
	}
}
