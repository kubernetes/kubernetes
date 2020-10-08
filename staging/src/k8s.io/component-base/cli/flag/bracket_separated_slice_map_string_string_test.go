/*
Copyright 2020 The Kubernetes Authors.

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

package flag

import (
	"reflect"
	"testing"
)

func TestStringBracketSeparatedSliceMapStringString(t *testing.T) {
	var nilSliceMap []map[string]string
	testCases := []struct {
		desc   string
		m      *BracketSeparatedSliceMapStringString
		expect string
	}{
		{"nill", NewBracketSeparatedSliceMapStringString(&nilSliceMap), ""},
		{"empty", NewBracketSeparatedSliceMapStringString(&[]map[string]string{}), ""},
		{"one key", NewBracketSeparatedSliceMapStringString(&[]map[string]string{{"a": "string"}}), "{a=string}"},
		{"two keys", NewBracketSeparatedSliceMapStringString(&[]map[string]string{{"a": "string", "b": "string"}}), "{a=string,b=string}"},
	}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			str := tc.m.String()
			if tc.expect != str {
				t.Fatalf("expect %q but got %q", tc.expect, str)
			}
		})
	}
}

func TestSetBracketSeparatedSliceMapStringString(t *testing.T) {
	var nilMap []map[string]string
	testCases := []struct {
		desc   string
		vals   []string
		start  *BracketSeparatedSliceMapStringString
		expect *BracketSeparatedSliceMapStringString
		err    string
	}{
		// we initialize the map with a default key that should be cleared by Set
		{"clears defaults", []string{""},
			NewBracketSeparatedSliceMapStringString(&[]map[string]string{{"default": ""}}),
			&BracketSeparatedSliceMapStringString{
				initialized: true,
				Value:       &[]map[string]string{},
			}, ""},
		// make sure we still allocate for "initialized" multimaps where Multimap was initially set to a nil map
		{"allocates map if currently nil", []string{""},
			&BracketSeparatedSliceMapStringString{initialized: true, Value: &nilMap},
			&BracketSeparatedSliceMapStringString{
				initialized: true,
				Value:       &[]map[string]string{},
			}, ""},
		// for most cases, we just reuse nilMap, which should be allocated by Set, and is reset before each test case
		{"empty", []string{""},
			NewBracketSeparatedSliceMapStringString(&nilMap),
			&BracketSeparatedSliceMapStringString{
				initialized: true,
				Value:       &[]map[string]string{},
			}, ""},
		{"empty braket", []string{"{}"},
			NewBracketSeparatedSliceMapStringString(&nilMap),
			&BracketSeparatedSliceMapStringString{
				initialized: true,
				Value:       &[]map[string]string{},
			}, ""},
		{"missing braket", []string{"a=string, b=string"},
			NewBracketSeparatedSliceMapStringString(&nilMap),
			&BracketSeparatedSliceMapStringString{
				initialized: true,
				Value:       &[]map[string]string{{"a": "string", "b": "string"}},
			}, ""},
		{"empty key", []string{"{=string}"},
			NewBracketSeparatedSliceMapStringString(&nilMap),
			&BracketSeparatedSliceMapStringString{
				initialized: true,
				Value:       &[]map[string]string{{"": "string"}},
			}, ""},
		{"one key", []string{"{a=string}"},
			NewBracketSeparatedSliceMapStringString(&nilMap),
			&BracketSeparatedSliceMapStringString{
				initialized: true,
				Value:       &[]map[string]string{{"a": "string"}},
			}, ""},
		{"two keys", []string{"{a=string,b=string}"},
			NewBracketSeparatedSliceMapStringString(&nilMap),
			&BracketSeparatedSliceMapStringString{
				initialized: true,
				Value:       &[]map[string]string{{"a": "string", "b": "string"}},
			}, ""},
		{"two duplecated keys", []string{"{a=string,a=string}"},
			NewBracketSeparatedSliceMapStringString(&nilMap),
			&BracketSeparatedSliceMapStringString{
				initialized: true,
				Value:       &[]map[string]string{{"a": "string"}},
			}, ""},
		{"two keys with space", []string{"{a = string, b = string}"},
			NewBracketSeparatedSliceMapStringString(&nilMap),
			&BracketSeparatedSliceMapStringString{
				initialized: true,
				Value:       &[]map[string]string{{"a": "string", "b": "string"}},
			}, ""},
		{"two keys, multiple Set invocations", []string{"{a=string, b=string}", "{a=string, b=string}"},
			NewBracketSeparatedSliceMapStringString(&nilMap),
			&BracketSeparatedSliceMapStringString{
				initialized: true,
				Value:       &[]map[string]string{{"a": "string", "b": "string"}, {"a": "string", "b": "string"}},
			}, ""},
		{"no target", []string{""},
			NewBracketSeparatedSliceMapStringString(nil),
			nil,
			"no target (nil pointer to []map[string]string)"},
	}
	for _, tc := range testCases {
		nilMap = nil
		t.Run(tc.desc, func(t *testing.T) {
			var err error
			for _, val := range tc.vals {
				err = tc.start.Set(val)
				if err != nil {
					break
				}
			}
			if tc.err != "" {
				if err == nil || err.Error() != tc.err {
					t.Fatalf("expect error %s but got %v", tc.err, err)
				}
				return
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(tc.expect, tc.start) {
				t.Fatalf("expect %#v but got %#v", tc.expect, tc.start)
			}
		})
	}
}

func TestEmptyBracketSeparatedSliceMapStringString(t *testing.T) {
	var nilSliceMap []map[string]string
	notEmpty := &BracketSeparatedSliceMapStringString{
		Value:       &[]map[string]string{{"a": "int", "b": "string", "c": "string"}},
		initialized: true,
	}

	testCases := []struct {
		desc   string
		m      *BracketSeparatedSliceMapStringString
		expect bool
	}{
		{"nil", NewBracketSeparatedSliceMapStringString(&nilSliceMap), true},
		{"empty", NewBracketSeparatedSliceMapStringString(&[]map[string]string{}), true},
		{"populated", notEmpty, false},
	}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			ret := tc.m.Empty()
			if ret != tc.expect {
				t.Fatalf("expect %t but got %t", tc.expect, ret)
			}
		})
	}
}
