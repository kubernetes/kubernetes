/*
Copyright 2025 The Kubernetes Authors.

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

func TestModal(t *testing.T) {
	errMatch := field.ErrorList{field.Invalid(field.NewPath("foo"), "bar", "match error")}
	errDefault := field.ErrorList{field.Invalid(field.NewPath("foo"), "bar", "default error")}

	mockValid := func(_ context.Context, _ operation.Operation, _ *field.Path, _, _ *string) field.ErrorList {
		return nil
	}
	mockErrorMatch := func(_ context.Context, _ operation.Operation, _ *field.Path, _, _ *string) field.ErrorList {
		return errMatch
	}
	mockErrorDefault := func(_ context.Context, _ operation.Operation, _ *field.Path, _, _ *string) field.ErrorList {
		return errDefault
	}

	mockEqual := func(a, b *string) bool {
		return a == b
	}

	testCases := []struct {
		name              string
		opType            operation.Type
		discriminator     any
		oldDiscriminator  any
		value             *string
		oldValue          *string
		rules             []ModalRule[*string]
		defaultValidation ValidateFunc[*string]
		expected          field.ErrorList
	}{
		{
			name:             "matches rule, returns valid",
			opType:           operation.Create,
			discriminator:    "A",
			oldDiscriminator: "A",
			rules: []ModalRule[*string]{
				{Value: "A", Validation: mockValid},
				{Value: "B", Validation: mockErrorMatch},
			},
			defaultValidation: mockErrorDefault,
			expected:          nil,
		},
		{
			name:             "matches rule, returns error",
			opType:           operation.Create,
			discriminator:    "B",
			oldDiscriminator: "B",
			rules: []ModalRule[*string]{
				{Value: "A", Validation: mockValid},
				{Value: "B", Validation: mockErrorMatch},
			},
			defaultValidation: mockErrorDefault,
			expected:          errMatch,
		},
		{
			name:             "ratcheting: update, unchanged, skips validation",
			opType:           operation.Update,
			discriminator:    "B",
			oldDiscriminator: "B", // unchanged
			value:            nil,
			oldValue:         nil, // unchanged
			rules: []ModalRule[*string]{
				{Value: "B", Validation: mockErrorMatch}, // would fail if run
			},
			defaultValidation: mockErrorDefault,
			expected:          nil,
		},
		{
			name:             "ratcheting: update, discriminator changed, runs validation",
			opType:           operation.Update,
			discriminator:    "B",
			oldDiscriminator: "A", // changed
			value:            nil,
			oldValue:         nil,
			rules: []ModalRule[*string]{
				{Value: "B", Validation: mockErrorMatch},
			},
			defaultValidation: mockErrorDefault,
			expected:          errMatch,
		},
		{
			name:          "matches rule with nil validation, returns valid",
			opType:        operation.Create,
			discriminator: "A",
			rules: []ModalRule[*string]{
				{Value: "A", Validation: nil},
			},
			defaultValidation: mockErrorDefault,
			expected:          nil,
		},
		{
			name:          "no match, runs default",
			opType:        operation.Create,
			discriminator: "C",
			rules: []ModalRule[*string]{
				{Value: "A", Validation: mockValid},
				{Value: "B", Validation: mockErrorMatch},
			},
			defaultValidation: mockErrorDefault,
			expected:          errDefault,
		},
		{
			name:          "no match, nil default, returns valid",
			opType:        operation.Create,
			discriminator: "C",
			rules: []ModalRule[*string]{
				{Value: "A", Validation: mockValid},
			},
			defaultValidation: nil,
			expected:          nil,
		},
		{
			name:          "int discriminator matches",
			opType:        operation.Create,
			discriminator: 1,
			rules: []ModalRule[*string]{
				{Value: "1", Validation: mockErrorMatch},
			},
			defaultValidation: mockErrorDefault,
			expected:          errMatch,
		},
		{
			name:          "bool discriminator matches",
			opType:        operation.Create,
			discriminator: true,
			rules: []ModalRule[*string]{
				{Value: "true", Validation: mockErrorMatch},
			},
			defaultValidation: mockErrorDefault,
			expected:          errMatch,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Handle nil oldDiscriminator for tests that don't specify it
			oldDisc := tc.oldDiscriminator
			if oldDisc == nil {
				oldDisc = tc.discriminator
			}

			// We need separate Modal calls because D is different per test case
			var got field.ErrorList

			switch d := tc.discriminator.(type) {
			case int:
				od, _ := oldDisc.(int)
				type IntP struct {
					Val  *string
					Disc int
				}
				newObj := &IntP{Val: tc.value, Disc: d}
				var oldObj *IntP
				if tc.opType == operation.Update {
					oldObj = &IntP{Val: tc.oldValue, Disc: od}
				}
				getVal := func(p *IntP) *string { return p.Val }
				getDisc := func(p *IntP) int { return p.Disc }

				got = Modal[*string, int, IntP](context.Background(), operation.Operation{Type: tc.opType}, field.NewPath("root"), newObj, oldObj, "field", getVal, getDisc, mockEqual, tc.defaultValidation, tc.rules)

			case bool:
				od, _ := oldDisc.(bool)
				type BoolP struct {
					Val  *string
					Disc bool
				}
				newObj := &BoolP{Val: tc.value, Disc: d}
				var oldObj *BoolP
				if tc.opType == operation.Update {
					oldObj = &BoolP{Val: tc.oldValue, Disc: od}
				}
				getVal := func(p *BoolP) *string { return p.Val }
				getDisc := func(p *BoolP) bool { return p.Disc }
				got = Modal[*string, bool, BoolP](context.Background(), operation.Operation{Type: tc.opType}, field.NewPath("root"), newObj, oldObj, "field", getVal, getDisc, mockEqual, tc.defaultValidation, tc.rules)

			case string:
				od, _ := oldDisc.(string)
				type StringP struct {
					Val  *string
					Disc string
				}
				newObj := &StringP{Val: tc.value, Disc: d}
				var oldObj *StringP
				if tc.opType == operation.Update {
					oldObj = &StringP{Val: tc.oldValue, Disc: od}
				}
				getVal := func(p *StringP) *string { return p.Val }
				getDisc := func(p *StringP) string { return p.Disc }
				got = Modal[*string, string, StringP](context.Background(), operation.Operation{Type: tc.opType}, field.NewPath("root"), newObj, oldObj, "field", getVal, getDisc, mockEqual, tc.defaultValidation, tc.rules)
			}

			if !reflect.DeepEqual(got, tc.expected) {
				t.Errorf("got %v want %v", got, tc.expected)
			}
		})
	}
}
