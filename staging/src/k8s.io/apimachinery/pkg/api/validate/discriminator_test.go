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

func TestDiscriminated(t *testing.T) {
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

	// mockEqual compares pointer values by dereferencing, not by pointer identity.
	mockEqual := func(a, b *string) bool {
		if a == nil && b == nil {
			return true
		}
		if a == nil || b == nil {
			return false
		}
		return *a == *b
	}

	testCases := []struct {
		name              string
		opType            operation.Type
		discriminator     string
		oldDiscriminator  string
		value             *string
		oldValue          *string
		rules             []DiscriminatedRule[*string, string]
		defaultValidation ValidateFunc[*string]
		expected          field.ErrorList
	}{
		{
			name:             "matches rule, returns valid",
			opType:           operation.Create,
			discriminator:    "A",
			oldDiscriminator: "A",
			rules: []DiscriminatedRule[*string, string]{
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
			rules: []DiscriminatedRule[*string, string]{
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
			rules: []DiscriminatedRule[*string, string]{
				{Value: "B", Validation: mockErrorMatch}, // would fail if run
			},
			defaultValidation: mockErrorDefault,
			expected:          nil,
		},
		{
			name:             "ratcheting: update, same value different pointers, skips validation",
			opType:           operation.Update,
			discriminator:    "B",
			oldDiscriminator: "B",
			value:            strPtr("same"),
			oldValue:         strPtr("same"), // different pointer, same value
			rules: []DiscriminatedRule[*string, string]{
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
			rules: []DiscriminatedRule[*string, string]{
				{Value: "B", Validation: mockErrorMatch},
			},
			defaultValidation: mockErrorDefault,
			expected:          errMatch,
		},
		{
			name:             "ratcheting: update, value changed, discriminator unchanged, runs validation",
			opType:           operation.Update,
			discriminator:    "B",
			oldDiscriminator: "B", // unchanged
			value:            strPtr("new"),
			oldValue:         strPtr("old"), // changed
			rules: []DiscriminatedRule[*string, string]{
				{Value: "B", Validation: mockErrorMatch},
			},
			defaultValidation: mockErrorDefault,
			expected:          errMatch,
		},
		{
			name:          "matches rule with nil validation, returns valid",
			opType:        operation.Create,
			discriminator: "A",
			rules: []DiscriminatedRule[*string, string]{
				{Value: "A", Validation: nil},
			},
			defaultValidation: mockErrorDefault,
			expected:          nil,
		},
		{
			name:          "no match, runs default",
			opType:        operation.Create,
			discriminator: "C",
			rules: []DiscriminatedRule[*string, string]{
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
			rules: []DiscriminatedRule[*string, string]{
				{Value: "A", Validation: mockValid},
			},
			defaultValidation: nil,
			expected:          nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			oldDisc := tc.oldDiscriminator
			if oldDisc == "" {
				oldDisc = tc.discriminator
			}

			type StringP struct {
				Val  *string
				Disc string
			}
			newObj := &StringP{Val: tc.value, Disc: tc.discriminator}
			var oldObj *StringP
			if tc.opType == operation.Update {
				oldObj = &StringP{Val: tc.oldValue, Disc: oldDisc}
			}
			getVal := func(p *StringP) *string { return p.Val }
			getDisc := func(p *StringP) string { return p.Disc }
			got := Discriminated[*string, string, StringP](context.Background(), operation.Operation{Type: tc.opType}, field.NewPath("root"), newObj, oldObj, "field", getVal, getDisc, mockEqual, tc.defaultValidation, tc.rules)

			if !reflect.DeepEqual(got, tc.expected) {
				t.Errorf("got %v want %v", got, tc.expected)
			}
		})
	}
}

func TestDiscriminatedIntDiscriminator(t *testing.T) {
	errMatch := field.ErrorList{field.Invalid(field.NewPath("foo"), "bar", "match error")}
	errDefault := field.ErrorList{field.Invalid(field.NewPath("foo"), "bar", "default error")}

	mockErrorMatch := func(_ context.Context, _ operation.Operation, _ *field.Path, _, _ *string) field.ErrorList {
		return errMatch
	}
	mockErrorDefault := func(_ context.Context, _ operation.Operation, _ *field.Path, _, _ *string) field.ErrorList {
		return errDefault
	}

	mockEqual := func(a, b *string) bool {
		return a == b
	}

	type IntP struct {
		Val  *string
		Disc int
	}
	newObj := &IntP{Val: nil, Disc: 1}
	getVal := func(p *IntP) *string { return p.Val }
	getDisc := func(p *IntP) int { return p.Disc }

	got := Discriminated[*string, int, IntP](context.Background(), operation.Operation{Type: operation.Create}, field.NewPath("root"), newObj, nil, "field", getVal, getDisc, mockEqual, mockErrorDefault, []DiscriminatedRule[*string, int]{
		{Value: 1, Validation: mockErrorMatch},
	})
	if !reflect.DeepEqual(got, errMatch) {
		t.Errorf("int discriminator: got %v want %v", got, errMatch)
	}
}

func TestDiscriminatedBoolDiscriminator(t *testing.T) {
	errMatch := field.ErrorList{field.Invalid(field.NewPath("foo"), "bar", "match error")}
	errDefault := field.ErrorList{field.Invalid(field.NewPath("foo"), "bar", "default error")}

	mockErrorMatch := func(_ context.Context, _ operation.Operation, _ *field.Path, _, _ *string) field.ErrorList {
		return errMatch
	}
	mockErrorDefault := func(_ context.Context, _ operation.Operation, _ *field.Path, _, _ *string) field.ErrorList {
		return errDefault
	}

	mockEqual := func(a, b *string) bool {
		return a == b
	}

	type BoolP struct {
		Val  *string
		Disc bool
	}
	newObj := &BoolP{Val: nil, Disc: true}
	getVal := func(p *BoolP) *string { return p.Val }
	getDisc := func(p *BoolP) bool { return p.Disc }

	got := Discriminated[*string, bool, BoolP](context.Background(), operation.Operation{Type: operation.Create}, field.NewPath("root"), newObj, nil, "field", getVal, getDisc, mockEqual, mockErrorDefault, []DiscriminatedRule[*string, bool]{
		{Value: true, Validation: mockErrorMatch},
	})
	if !reflect.DeepEqual(got, errMatch) {
		t.Errorf("bool discriminator: got %v want %v", got, errMatch)
	}
}

// strPtr returns a new pointer to a copy of s, guaranteeing a distinct allocation.
func strPtr(s string) *string {
	return &s
}
