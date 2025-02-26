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

type testUnion struct{}
type testMember struct{}

func TestUnion(t *testing.T) {
	testCases := []struct {
		name        string
		fields      [][2]string
		fieldValues []any
		expected    field.ErrorList
	}{
		{
			name:        "valid pointers one of",
			fields:      [][2]string{{"a", "A"}, {"b", "B"}, {"c", "C"}, {"d", "D"}},
			fieldValues: []any{nil, nil, nil, &testMember{}},
			expected:    nil,
		},
		{
			name:        "invalid pointers one of",
			fields:      [][2]string{{"a", "A"}, {"b", "B"}, {"c", "C"}, {"d", "D"}},
			fieldValues: []any{nil, &testMember{}, nil, &testMember{}},
			expected:    field.ErrorList{field.Invalid(nil, "{b, d}", "must specify exactly one of: `a`, `b`, `c`, `d`")},
		},
		{
			name:        "valid string one of",
			fields:      [][2]string{{"a", "A"}, {"b", "B"}, {"c", "C"}, {"d", "D"}},
			fieldValues: []any{"", "", "", "x"},
			expected:    nil,
		},
		{
			name:        "invalid string one of",
			fields:      [][2]string{{"a", "A"}, {"b", "B"}, {"c", "C"}, {"d", "D"}},
			fieldValues: []any{"", "x", "", "x"},
			expected:    field.ErrorList{field.Invalid(nil, "{b, d}", "must specify exactly one of: `a`, `b`, `c`, `d`")},
		},
		{
			name:        "valid mixed type one of",
			fields:      [][2]string{{"a", "A"}, {"b", "B"}, {"c", "C"}, {"d", "D"}},
			fieldValues: []any{0, "", nil, "x"},
			expected:    nil,
		},
		{
			name:        "invalid mixed type one of",
			fields:      [][2]string{{"a", "A"}, {"b", "B"}, {"c", "C"}, {"d", "D"}},
			fieldValues: []any{0, "x", nil, &testMember{}},
			expected:    field.ErrorList{field.Invalid(nil, "{b, d}", "must specify exactly one of: `a`, `b`, `c`, `d`")},
		},
		{
			name:        "invalid no member set",
			fields:      [][2]string{{"a", "A"}, {"b", "B"}, {"c", "C"}, {"d", "D"}},
			fieldValues: []any{nil, nil, nil, nil},
			expected:    field.ErrorList{field.Invalid(nil, "", "must specify exactly one of: `a`, `b`, `c`, `d`")},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := Union(context.Background(), operation.Operation{Code: operation.Update}, nil, nil, nil, NewUnionMembership(tc.fields...), tc.fieldValues...)
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
		fieldValues        []any
		expected           field.ErrorList
	}{
		{
			name:               "valid discriminated union",
			discriminatorField: "d",
			fields:             [][2]string{{"a", "A"}, {"b", "B"}, {"c", "C"}, {"d", "D"}},
			discriminatorValue: "A",
			fieldValues:        []any{1, nil, nil, nil},
		},
		{
			name:               "invalid, discriminator not set to member that is specified",
			discriminatorField: "type",
			fields:             [][2]string{{"a", "A"}, {"b", "B"}, {"c", "C"}, {"d", "D"}},
			discriminatorValue: "C",
			fieldValues:        []any{nil, 1, nil, nil},
			expected: field.ErrorList{
				field.Invalid(field.NewPath("b"), "", "may only be specified when `type` is \"B\""),
				field.Invalid(field.NewPath("c"), "", "must be specified when `type` is \"C\""),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := DiscriminatedUnion(context.Background(), operation.Operation{Code: operation.Update}, nil, nil, nil, NewDiscriminatedUnionMembership(tc.discriminatorField, tc.fields...), tc.discriminatorValue, tc.fieldValues...)
			if !reflect.DeepEqual(got, tc.expected) {
				t.Errorf("got %v want %v", got.ToAggregate(), tc.expected.ToAggregate())
			}
		})
	}
}
