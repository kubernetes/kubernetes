/*
Copyright The Kubernetes Authors.

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

package shortcircuit

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestRequiredShortCircuit(t *testing.T) {
	tests := []struct {
		name                string
		value               any
		expectValidateFalse map[string][]string
		expectErrors        field.ErrorList
	}{
		{
			name: "required field is nil, short circuits",
			value: &ParentWithRequired{
				Field: TargetWithRequired{
					Value: nil,
				},
			},
			expectErrors: field.ErrorList{
				field.Required(field.NewPath("field", "value"), ""),
			},
		},
		{
			name: "required field is provided, subfield validation runs",
			value: &ParentWithRequired{
				Field: TargetWithRequired{
					Value: new(""),
				},
			},
			expectValidateFalse: map[string][]string{
				"field.value": {"subfield ParentWithRequired.Field.Value"},
			},
		},
		{
			name: "opaqueType on field, required check is not inherited, subfield validation runs",
			value: &ParentWithOpaqueField{
				Field: TargetWithRequired{
					Value: nil,
				},
			},
			expectValidateFalse: map[string][]string{
				"field.value": {"subfield ParentWithOpaqueField.Field.Value"},
			},
		},
		{
			name: "alpha opaqueType on field, required check is not inherited, subfield validation runs",
			value: &ParentWithAlphaOpaqueField{
				Field: TargetWithRequired{
					Value: nil,
				},
			},
			expectValidateFalse: map[string][]string{
				"field.value": {"subfield ParentWithAlphaOpaqueField.Field.Value"},
			},
		},

		{
			name: "opaqueType on alias, required check is not inherited, subfield validation runs",
			value: &ParentWithOpaqueAlias{
				Field: AliasOpaqueTargetWithRequired{
					Value: nil,
				},
			},
			expectValidateFalse: map[string][]string{
				"field.value": {"subfield ParentWithOpaqueAlias.Field.Value"},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			st := localSchemeBuilder.Test(t)
			tester := st.Value(tc.value)
			if len(tc.expectErrors) > 0 {
				tester.ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), tc.expectErrors)
			} else {
				tester.ExpectValidateFalseByPath(tc.expectValidateFalse)
			}
		})
	}
}

func TestImmutableShortCircuit(t *testing.T) {
	tests := []struct {
		name                string
		value               any
		oldValue            any
		expectValidateFalse map[string][]string
		expectErrors        field.ErrorList
	}{
		{
			name: "immutable field changed on update, short circuits",
			value: &ParentWithImmutable{
				Field: TargetWithImmutable{
					Value: "new",
				},
			},
			oldValue: &ParentWithImmutable{
				Field: TargetWithImmutable{
					Value: "old",
				},
			},
			expectErrors: field.ErrorList{
				field.Invalid(field.NewPath("field", "value"), "new", "").WithOrigin("immutable"),
			},
		},
		{
			name: "immutable field not validated on create, subfield validation runs",
			value: &ParentWithImmutable{
				Field: TargetWithImmutable{
					Value: "new",
				},
			},
			expectValidateFalse: map[string][]string{
				"field.value": {"subfield ParentWithImmutable.Field.Value"},
			},
		},
		{
			name: "opaqueType on field, immutable check is not inherited, subfield validation runs on update",
			value: &ParentWithOpaqueImmutableField{
				Field: TargetWithImmutable{
					Value: "new",
				},
			},
			oldValue: &ParentWithOpaqueImmutableField{
				Field: TargetWithImmutable{
					Value: "old",
				},
			},
			expectValidateFalse: map[string][]string{
				"field.value": {"subfield ParentWithOpaqueImmutableField.Field.Value"},
			},
		},
		{
			name: "alpha opaqueType on field, immutable check is not inherited, subfield validation runs on update",
			value: &ParentWithAlphaOpaqueImmutableField{
				Field: TargetWithImmutable{
					Value: "new",
				},
			},
			oldValue: &ParentWithAlphaOpaqueImmutableField{
				Field: TargetWithImmutable{
					Value: "old",
				},
			},
			expectValidateFalse: map[string][]string{
				"field.value": {"subfield ParentWithAlphaOpaqueImmutableField.Field.Value"},
			},
		},

		{
			name: "opaqueType on alias, immutable check is not inherited, subfield validation runs on update",
			value: &ParentWithOpaqueImmutableAlias{
				Field: AliasOpaqueTargetWithImmutable{
					Value: "new",
				},
			},
			oldValue: &ParentWithOpaqueImmutableAlias{
				Field: AliasOpaqueTargetWithImmutable{
					Value: "old",
				},
			},
			expectValidateFalse: map[string][]string{
				"field.value": {"subfield ParentWithOpaqueImmutableAlias.Field.Value"},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			st := localSchemeBuilder.Test(t)
			tester := st.Value(tc.value)
			if tc.oldValue != nil {
				tester.OldValue(tc.oldValue)
			}
			if len(tc.expectErrors) > 0 {
				tester.ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().MatchShortCircuit(), tc.expectErrors.MarkShortCircuit())
			} else {
				tester.ExpectValidateFalseByPath(tc.expectValidateFalse)
			}
		})
	}
}

func TestMultipleShortCircuit(t *testing.T) {
	tests := []struct {
		name                string
		value               any
		oldValue            any
		expectValidateFalse map[string][]string
		expectErrors        field.ErrorList
	}{
		{
			name: "required field is nil, short circuits at field level",
			value: &ParentWithMultipleShortCircuit{
				Field: nil,
			},
			expectErrors: field.ErrorList{
				field.Required(field.NewPath("field"), ""),
			},
		},
		{
			name: "immutable subfield changed, short circuits at subfield level on update",
			value: &ParentWithMultipleShortCircuit{
				Field: &TargetWithRequired{
					Value: new("new"),
				},
			},
			oldValue: &ParentWithMultipleShortCircuit{
				Field: &TargetWithRequired{
					Value: new("old"),
				},
			},
			expectErrors: field.ErrorList{
				field.Invalid(field.NewPath("field", "value"), "new", "").WithOrigin("immutable"),
			},
		},
		{
			name: "field is not nil, subfield validation runs on create",
			value: &ParentWithMultipleShortCircuit{
				Field: &TargetWithRequired{
					Value: new("val"),
				},
			},
			expectValidateFalse: map[string][]string{
				"field.value": {"subfield ParentWithMultipleShortCircuit.Field.Value"},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			st := localSchemeBuilder.Test(t)
			tester := st.Value(tc.value)
			if tc.oldValue != nil {
				tester.OldValue(tc.oldValue)
			}
			if len(tc.expectErrors) > 0 {
				tester.ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().MatchShortCircuit(), tc.expectErrors.MarkShortCircuit())
			} else {
				tester.ExpectValidateFalseByPath(tc.expectValidateFalse)
			}
		})
	}
}

func TestOtherShortCircuits(t *testing.T) {
	tests := []struct {
		name                string
		value               any
		oldValue            any
		expectValidateFalse map[string][]string
		expectErrors        field.ErrorList
	}{
		// +k8s:optional
		{
			name: "optional field is nil, short circuits",
			value: &ParentWithOptional{
				Field: TargetWithOptional{
					Value: nil,
				},
			},
			expectValidateFalse: map[string][]string{},
		},
		{
			name: "optional field is provided, subfield validation runs",
			value: &ParentWithOptional{
				Field: TargetWithOptional{
					Value: new(""),
				},
			},
			expectValidateFalse: map[string][]string{
				"field.value": {"subfield ParentWithOptional.Field.Value"},
			},
		},

		// +k8s:forbidden
		{
			name: "forbidden field is nil, short circuits",
			value: &ParentWithForbidden{
				Field: TargetWithForbidden{
					Value: nil,
				},
			},
			expectValidateFalse: map[string][]string{},
		},
		{
			name: "forbidden field is provided, fails forbidden validation and short circuits",
			value: &ParentWithForbidden{
				Field: TargetWithForbidden{
					Value: new(""),
				},
			},
			expectErrors: field.ErrorList{
				field.Forbidden(field.NewPath("field", "value"), ""),
			},
		},

		// +k8s:update
		{
			name: "update field on create, subfield validation runs",
			value: &ParentWithUpdate{
				Field: TargetWithUpdate{
					Value: "any",
				},
			},
			expectValidateFalse: map[string][]string{
				"field.value": {"subfield ParentWithUpdate.Field.Value"},
			},
		},
		{
			name: "update field modified on update, fails update validation and short circuits",
			value: &ParentWithUpdate{
				Field: TargetWithUpdate{
					Value: "new",
				},
			},
			oldValue: &ParentWithUpdate{
				Field: TargetWithUpdate{
					Value: "old",
				},
			},
			expectErrors: field.ErrorList{
				field.Invalid(field.NewPath("field", "value"), "new", "").WithOrigin("update"),
			},
		},

		// +k8s:maxItems
		{
			name: "maxItems field within limit, subfield validation runs",
			value: &ParentWithMaxItems{
				Field: TargetWithMaxItems{
					Value: []string{"a", "b"},
				},
			},
			expectValidateFalse: map[string][]string{
				"field.value": {"subfield ParentWithMaxItems.Field.Value"},
			},
		},
		{
			name: "maxItems field exceeds limit, fails maxItems validation and short circuits",
			value: &ParentWithMaxItems{
				Field: TargetWithMaxItems{
					Value: []string{"a", "b", "c"},
				},
			},
			expectErrors: field.ErrorList{
				field.TooMany(field.NewPath("field", "value"), 3, 2).WithOrigin("maxItems"),
			},
		},

		// +k8s:maxProperties
		{
			name: "maxProperties field within limit, subfield validation runs",
			value: &ParentWithMaxProperties{
				Field: TargetWithMaxProperties{
					Value: map[string]string{"a": "1", "b": "2"},
				},
			},
			expectValidateFalse: map[string][]string{
				"field.value": {"subfield ParentWithMaxProperties.Field.Value"},
			},
		},
		{
			name: "maxProperties field exceeds limit, fails maxProperties validation and short circuits",
			value: &ParentWithMaxProperties{
				Field: TargetWithMaxProperties{
					Value: map[string]string{"a": "1", "b": "2", "c": "3"},
				},
			},
			expectErrors: field.ErrorList{
				field.TooMany(field.NewPath("field", "value"), 3, 2).WithOrigin("maxProperties"),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			st := localSchemeBuilder.Test(t)
			tester := st.Value(tc.value)
			if tc.oldValue != nil {
				tester.OldValue(tc.oldValue)
			}
			if len(tc.expectErrors) > 0 {
				tester.ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().MatchShortCircuit(), tc.expectErrors.MarkShortCircuit())
			} else {
				tester.ExpectValidateFalseByPath(tc.expectValidateFalse)
			}
		})
	}
}
