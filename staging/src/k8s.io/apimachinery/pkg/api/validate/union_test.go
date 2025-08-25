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

type testMember struct{}

func TestUnion(t *testing.T) {
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
			expected:    field.ErrorList{field.Invalid(nil, "{b, d}", "must specify exactly one of: `a`, `b`, `c`, `d`")},
		},
		{
			name:        "all members set",
			fields:      [][2]string{{"a", "A"}, {"b", "B"}, {"c", "C"}, {"d", "D"}},
			fieldValues: []bool{true, true, true, true},
			expected:    field.ErrorList{field.Invalid(nil, "{a, b, c, d}", "must specify exactly one of: `a`, `b`, `c`, `d`")},
		},
		{
			name:        "no member set",
			fields:      [][2]string{{"a", "A"}, {"b", "B"}, {"c", "C"}, {"d", "D"}},
			fieldValues: []bool{false, false, false, false},
			expected:    field.ErrorList{field.Invalid(nil, "", "must specify one of: `a`, `b`, `c`, `d`")},
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

			got := Union(context.Background(), operation.Operation{}, nil, &testMember{}, nil, NewUnionMembership(tc.fields...), extractors...)
			if !reflect.DeepEqual(got, tc.expected) {
				t.Errorf("got %v want %v", got, tc.expected)
			}
		})
	}
}

func TestDiscriminatedUnion(t *testing.T) {
	testCases := []struct {
		name               string
		discriminatorField string
		fields             [][2]string
		discriminatorValue string
		fieldValues        []bool
		expected           field.ErrorList
	}{
		{
			name:               "valid discriminated union A",
			discriminatorField: "d",
			fields:             [][2]string{{"a", "A"}, {"b", "B"}, {"c", "C"}, {"d", "D"}},
			discriminatorValue: "A",
			fieldValues:        []bool{true, false, false, false},
		},
		{
			name:               "valid discriminated union C",
			discriminatorField: "d",
			fields:             [][2]string{{"a", "A"}, {"b", "B"}, {"c", "C"}, {"d", "D"}},
			discriminatorValue: "C",
			fieldValues:        []bool{false, false, true, false},
		},
		{
			name:               "invalid, discriminator not set to member that is specified",
			discriminatorField: "type",
			fields:             [][2]string{{"a", "A"}, {"b", "B"}, {"c", "C"}, {"d", "D"}},
			discriminatorValue: "C",
			fieldValues:        []bool{false, true, false, false},
			expected: field.ErrorList{
				field.Invalid(field.NewPath("b"), "", "may only be specified when `type` is \"B\""),
				field.Invalid(field.NewPath("c"), "", "must be specified when `type` is \"C\""),
			},
		},
		{
			name:               "invalid, discriminator correct, multiple members set",
			discriminatorField: "type",
			fields:             [][2]string{{"a", "A"}, {"b", "B"}, {"c", "C"}, {"d", "D"}},
			discriminatorValue: "C",
			fieldValues:        []bool{false, true, true, true},
			expected: field.ErrorList{
				field.Invalid(field.NewPath("b"), "", "may only be specified when `type` is \"B\""),
				field.Invalid(field.NewPath("d"), "", "may only be specified when `type` is \"D\""),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			discriminatorExtractor := func(_ *testMember) string { return tc.discriminatorValue }

			// Create mock extractors that return predefined values instead of
			// actually extracting from the object.
			extractors := make([]ExtractorFn[*testMember, bool], len(tc.fieldValues))
			for i, val := range tc.fieldValues {
				extractors[i] = func(_ *testMember) bool { return val }
			}

			got := DiscriminatedUnion(context.Background(), operation.Operation{}, nil, &testMember{}, nil, NewDiscriminatedUnionMembership(tc.discriminatorField, tc.fields...), discriminatorExtractor, extractors...)
			if !reflect.DeepEqual(got, tc.expected) {
				t.Errorf("got %v want %v", got.ToAggregate(), tc.expected.ToAggregate())
			}
		})
	}
}

type testStruct struct {
	M1 *m1 `json:"m1"`
	M2 *m2 `json:"m2"`
}

type m1 struct{}
type m2 struct{}

var extractors = []ExtractorFn[*testStruct, bool]{
	func(s *testStruct) bool {
		if s == nil {
			return false
		}
		return s.M1 != nil
	},
	func(s *testStruct) bool {
		if s == nil {
			return false
		}
		return s.M2 != nil
	},
}

func TestUnionRatcheting(t *testing.T) {
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
			name:      "both empty struct",
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
				field.Invalid(nil, "{m1, m2}", "must specify exactly one of: `m1`, `m2`"),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := Union(context.Background(), operation.Operation{Type: operation.Update}, nil, tc.newStruct, tc.oldStruct, NewUnionMembership([][2]string{{"m1", "m1"}, {"m2", "m2"}}...), extractors...)
			if !reflect.DeepEqual(got, tc.expected) {
				t.Errorf("got %v want %v", got, tc.expected)
			}
		})
	}
}

type testDiscriminatedStruct struct {
	D  string `json:"d"`
	M1 *m1    `json:"m1"`
	M2 *m2    `json:"m2"`
}

var testDiscriminatorExtractor = func(s *testDiscriminatedStruct) string {
	if s != nil {
		return s.D
	}
	return ""
}
var testDiscriminatedExtractors = []ExtractorFn[*testDiscriminatedStruct, bool]{
	func(s *testDiscriminatedStruct) bool {
		if s == nil {
			return false
		}
		return s.M1 != nil
	},
	func(s *testDiscriminatedStruct) bool {
		if s == nil {
			return false
		}
		return s.M2 != nil
	},
}

func TestDiscriminatedUnionRatcheting(t *testing.T) {
	testCases := []struct {
		name      string
		oldStruct *testDiscriminatedStruct
		newStruct *testDiscriminatedStruct
		expected  field.ErrorList
	}{
		{
			name: "pass with both nil",
		},
		{
			name:      "pass with both empty struct",
			oldStruct: &testDiscriminatedStruct{},
			newStruct: &testDiscriminatedStruct{},
		},
		{
			name: "pass with both not set to member that is specified",
			oldStruct: &testDiscriminatedStruct{
				D:  "m1",
				M2: &m2{},
			},
			newStruct: &testDiscriminatedStruct{
				D:  "m1",
				M2: &m2{},
			},
		},
		{
			name: "pass with both set to more than one member",
			oldStruct: &testDiscriminatedStruct{
				D:  "m1",
				M1: &m1{},
				M2: &m2{},
			},
			newStruct: &testDiscriminatedStruct{
				D:  "m1",
				M1: &m1{},
				M2: &m2{},
			},
		},
		{
			name: "fail on changing to invalid with both set",
			oldStruct: &testDiscriminatedStruct{
				D:  "m1",
				M1: &m1{},
			},
			newStruct: &testDiscriminatedStruct{
				D:  "m1",
				M1: &m1{},
				M2: &m2{},
			},
			expected: field.ErrorList{
				field.Invalid(field.NewPath("m2"), "", "may only be specified when `d` is \"m2\""),
			},
		},
		{
			name: "fail on changing the discriminator",
			oldStruct: &testDiscriminatedStruct{
				D:  "m1",
				M1: &m1{},
			},
			newStruct: &testDiscriminatedStruct{
				D:  "m2",
				M1: &m1{},
			},
			expected: field.ErrorList{
				field.Invalid(field.NewPath("m1"), "", "may only be specified when `d` is \"m1\""),
				field.Invalid(field.NewPath("m2"), "", "must be specified when `d` is \"m2\""),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := DiscriminatedUnion(context.Background(), operation.Operation{Type: operation.Update}, nil, tc.newStruct, tc.oldStruct, NewDiscriminatedUnionMembership("d", [][2]string{{"m1", "m1"}, {"m2", "m2"}}...), testDiscriminatorExtractor, testDiscriminatedExtractors...)
			if !reflect.DeepEqual(got, tc.expected) {
				t.Errorf("got %v want %v", got, tc.expected)
			}
		})
	}
}
