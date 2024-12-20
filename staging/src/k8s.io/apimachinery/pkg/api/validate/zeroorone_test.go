/*
Copyright 2024 The Kubernetes Authors.

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

package validate

import (
	"context"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestZeroOrOneOfUnion(t *testing.T) {
	testCases := []struct {
		name        string
		fields      [][2]string
		fieldValues []bool
		expected    field.ErrorList
	}{
		{
			name:        "one member set",
			fields:      [][2]string{{"a", "A"}, {"b", "B"}, {"c", "C"}, {"d", "D"}},
			fieldValues: []bool{false, false, false, true},
			expected:    nil,
		},
		{
			name:        "two members set",
			fields:      [][2]string{{"a", "A"}, {"b", "B"}, {"c", "C"}, {"d", "D"}},
			fieldValues: []bool{false, true, false, true},
			expected:    field.ErrorList{field.Invalid(nil, "{b, d}", "must specify at most one of: `a`, `b`, `c`, `d`").WithOrigin("zeroOrOneOf")},
		},
		{
			name:        "all members set",
			fields:      [][2]string{{"a", "A"}, {"b", "B"}, {"c", "C"}, {"d", "D"}},
			fieldValues: []bool{true, true, true, true},
			expected:    field.ErrorList{field.Invalid(nil, "{a, b, c, d}", "must specify at most one of: `a`, `b`, `c`, `d`").WithOrigin("zeroOrOneOf")},
		},
		{
			name:        "no member set - allowed for ZeroOrOneOf",
			fields:      [][2]string{{"a", "A"}, {"b", "B"}, {"c", "C"}, {"d", "D"}},
			fieldValues: []bool{false, false, false, false},
			expected:    nil, // This is the key difference from Union
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create mock extractors that return predefined values instead of
			// actually extracting from the object.
			extractors := make([]ExtractorFn[*testMember, bool], len(tc.fieldValues))
			for i, val := range tc.fieldValues {
				extractors[i] = func(_ *testMember) bool { return val }
			}

			got := ZeroOrOneOfUnion(context.Background(), operation.Operation{}, nil, &testMember{}, nil, NewUnionMembership(tc.fields...), extractors...)
			if !reflect.DeepEqual(got, tc.expected) {
				t.Errorf("got %v want %v", got, tc.expected)
			}
		})
	}
}

func TestZeroOrOneOfUnionRatcheting(t *testing.T) {
	testCases := []struct {
		name      string
		oldStruct *testStruct
		newStruct *testStruct
		expected  field.ErrorList
	}{
		{
			name:      "both nil",
			oldStruct: nil,
			newStruct: nil,
		},
		{
			name:      "both empty struct - allowed for ZeroOrOneOf",
			oldStruct: &testStruct{},
			newStruct: &testStruct{},
		},
		{
			name: "both have more than one member",
			oldStruct: &testStruct{
				M1: &m1{},
				M2: &m2{},
			},
			newStruct: &testStruct{
				M1: &m1{},
				M2: &m2{},
			},
		},
		{
			name: "change to invalid",
			oldStruct: &testStruct{
				M1: &m1{},
			},
			newStruct: &testStruct{
				M1: &m1{},
				M2: &m2{},
			},
			expected: field.ErrorList{
				field.Invalid(nil, "{m1, m2}", "must specify at most one of: `m1`, `m2`").WithOrigin("zeroOrOneOf"),
			},
		},
		{
			name:      "change from empty to one member - allowed",
			oldStruct: &testStruct{},
			newStruct: &testStruct{
				M1: &m1{},
			},
			expected: nil,
		},
		{
			name: "change from one member to empty - allowed",
			oldStruct: &testStruct{
				M1: &m1{},
			},
			newStruct: &testStruct{},
			expected:  nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := ZeroOrOneOfUnion(context.Background(), operation.Operation{Type: operation.Update}, nil, tc.newStruct, tc.oldStruct, NewUnionMembership([][2]string{{"m1", "m1"}, {"m2", "m2"}}...), extractors...)
			if !reflect.DeepEqual(got, tc.expected) {
				t.Errorf("got %v want %v", got, tc.expected)
			}
		})
	}
}
