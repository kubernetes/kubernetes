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
		name         string
		value        any
		expectErrors field.ErrorList
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
			}.MarkShortCircuit(),
		},
		{
			name: "required field is provided, subfield validation runs",
			value: &ParentWithRequired{
				Field: TargetWithRequired{
					Value: new(""),
				},
			},
			expectErrors: field.ErrorList{
				field.Invalid(field.NewPath("field", "value"), nil, "forced failure: subfield ParentWithRequired.Field.Value").WithOrigin("validateFalse"),
			},
		},
		{
			name: "opaqueType on field, required check is not inherited, subfield validation runs",
			value: &ParentWithOpaqueField{
				Field: TargetWithRequired{
					Value: nil,
				},
			},
			expectErrors: field.ErrorList{
				field.Invalid(field.NewPath("field", "value"), nil, "forced failure: subfield ParentWithOpaqueField.Field.Value").WithOrigin("validateFalse"),
			},
		},
		{
			name: "alpha opaqueType on field, required check is not inherited, subfield validation runs",
			value: &ParentWithAlphaOpaqueField{
				Field: TargetWithRequired{
					Value: nil,
				},
			},
			expectErrors: field.ErrorList{
				field.Invalid(field.NewPath("field", "value"), nil, "forced failure: subfield ParentWithAlphaOpaqueField.Field.Value").WithOrigin("validateFalse"),
			},
		},
		{
			name: "opaqueType on alias, required check is not inherited, subfield validation runs",
			value: &ParentWithOpaqueAlias{
				Field: AliasOpaqueTargetWithRequired{
					Value: nil,
				},
			},
			expectErrors: field.ErrorList{
				field.Invalid(field.NewPath("field", "value"), nil, "forced failure: subfield ParentWithOpaqueAlias.Field.Value").WithOrigin("validateFalse"),
			},
		},
		{
			name: "opaqueType on pointer alias, required check is not inherited, subfield validation runs",
			value: &ParentWithPointerOpaqueAlias{
				Field: &AliasOpaqueTargetWithRequired{
					Value: nil,
				},
			},
			expectErrors: field.ErrorList{
				field.Invalid(field.NewPath("field", "value"), nil, "forced failure: subfield ParentWithPointerOpaqueAlias.Field.Value").WithOrigin("validateFalse"),
			},
		},
	}

	matcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByDetailExact().MatchShortCircuit()

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			st := localSchemeBuilder.Test(t)
			st.Value(tc.value).ExpectMatches(matcher, tc.expectErrors)
		})
	}
}

func TestImmutableShortCircuit(t *testing.T) {
	tests := []struct {
		name         string
		value        any
		oldValue     any
		expectErrors field.ErrorList
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
				field.Invalid(field.NewPath("field", "value"), "new", "field is immutable").WithOrigin("immutable"),
			}.MarkShortCircuit(),
		},
		{
			name: "immutable field not validated on create, subfield validation runs",
			value: &ParentWithImmutable{
				Field: TargetWithImmutable{
					Value: "new",
				},
			},
			expectErrors: field.ErrorList{
				field.Invalid(field.NewPath("field", "value"), nil, "forced failure: subfield ParentWithImmutable.Field.Value").WithOrigin("validateFalse"),
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
			expectErrors: field.ErrorList{
				field.Invalid(field.NewPath("field", "value"), nil, "forced failure: subfield ParentWithOpaqueImmutableField.Field.Value").WithOrigin("validateFalse"),
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
			expectErrors: field.ErrorList{
				field.Invalid(field.NewPath("field", "value"), nil, "forced failure: subfield ParentWithAlphaOpaqueImmutableField.Field.Value").WithOrigin("validateFalse"),
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
			expectErrors: field.ErrorList{
				field.Invalid(field.NewPath("field", "value"), nil, "forced failure: subfield ParentWithOpaqueImmutableAlias.Field.Value").WithOrigin("validateFalse"),
			},
		},
	}

	matcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByDetailExact().MatchShortCircuit()

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			st := localSchemeBuilder.Test(t)
			tester := st.Value(tc.value)
			if tc.oldValue != nil {
				tester.OldValue(tc.oldValue)
			}
			tester.ExpectMatches(matcher, tc.expectErrors)
		})
	}
}

func TestMultipleShortCircuit(t *testing.T) {
	tests := []struct {
		name         string
		value        any
		oldValue     any
		expectErrors field.ErrorList
	}{
		{
			name: "required field is nil, short circuits at field level",
			value: &ParentWithMultipleShortCircuit{
				Field: nil,
			},
			expectErrors: field.ErrorList{
				field.Required(field.NewPath("field"), ""),
			}.MarkShortCircuit(),
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
				field.Invalid(field.NewPath("field", "value"), "new", "field is immutable").WithOrigin("immutable"),
			}.MarkShortCircuit(),
		},
		{
			name: "immutable subfield changed to nil on update, subfield immutable runs before inherited required",
			value: &ParentWithMultipleShortCircuit{
				Field: &TargetWithRequired{
					Value: nil,
				},
			},
			oldValue: &ParentWithMultipleShortCircuit{
				Field: &TargetWithRequired{
					Value: new("old"),
				},
			},
			expectErrors: field.ErrorList{
				field.Invalid(field.NewPath("field", "value"), nil, "field is immutable").WithOrigin("immutable"),
				field.Required(field.NewPath("field", "value"), ""),
			}.MarkShortCircuit(),
		},
		{
			name: "field is not nil, subfield validation runs on create",
			value: &ParentWithMultipleShortCircuit{
				Field: &TargetWithRequired{
					Value: new("val"),
				},
			},
			expectErrors: field.ErrorList{
				field.Invalid(field.NewPath("field", "value"), nil, "forced failure: subfield ParentWithMultipleShortCircuit.Field.Value").WithOrigin("validateFalse"),
			},
		},
	}

	matcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByDetailExact().MatchShortCircuit()

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			st := localSchemeBuilder.Test(t)
			tester := st.Value(tc.value)
			if tc.oldValue != nil {
				tester.OldValue(tc.oldValue)
			}
			tester.ExpectMatches(matcher, tc.expectErrors)
		})
	}
}

func TestOtherShortCircuits(t *testing.T) {
	tests := []struct {
		name         string
		value        any
		oldValue     any
		expectErrors field.ErrorList
	}{
		// +k8s:optional
		{
			name: "optional field is nil, short circuits",
			value: &ParentWithOptional{
				Field: TargetWithOptional{
					Value: nil,
				},
			},
			expectErrors: nil,
		},
		{
			name: "optional field is provided, subfield validation runs",
			value: &ParentWithOptional{
				Field: TargetWithOptional{
					Value: new(""),
				},
			},
			expectErrors: field.ErrorList{
				field.Invalid(field.NewPath("field", "value"), nil, "forced failure: subfield ParentWithOptional.Field.Value").WithOrigin("validateFalse"),
			},
		},

		// +k8s:required on subfield, +k8s:optional on child
		{
			name: "subfield required but value is nil, fails required validation and short circuits (optional child does not prevent it)",
			value: &ParentWithSubfieldRequiredAndChildOptional{
				Field: TargetWithOptional{
					Value: nil,
				},
			},
			expectErrors: field.ErrorList{
				field.Required(field.NewPath("field", "value"), ""),
			}.MarkShortCircuit(),
		},
		{
			name: "subfield required and value is provided, runs non-short-circuit validation",
			value: &ParentWithSubfieldRequiredAndChildOptional{
				Field: TargetWithOptional{
					Value: new("val"),
				},
			},
			expectErrors: field.ErrorList{
				field.Invalid(field.NewPath("field", "value"), nil, "forced failure: subfield ParentWithSubfieldRequiredAndChildOptional.Field.Value").WithOrigin("validateFalse"),
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
			expectErrors: nil,
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
			}.MarkShortCircuit(),
		},

		// +k8s:update
		{
			name: "update field on create, subfield validation runs",
			value: &ParentWithUpdate{
				Field: TargetWithUpdate{
					Value: "any",
				},
			},
			expectErrors: field.ErrorList{
				field.Invalid(field.NewPath("field", "value"), nil, "forced failure: subfield ParentWithUpdate.Field.Value").WithOrigin("validateFalse"),
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
				field.Invalid(field.NewPath("field", "value"), "new", "field cannot be modified once set").WithOrigin("update"),
			}.MarkShortCircuit(),
		},

		// +k8s:maxItems
		{
			name: "maxItems field within limit, subfield validation runs",
			value: &ParentWithMaxItems{
				Field: TargetWithMaxItems{
					Value: []string{"a", "b"},
				},
			},
			expectErrors: field.ErrorList{
				field.Invalid(field.NewPath("field", "value"), nil, "forced failure: subfield ParentWithMaxItems.Field.Value").WithOrigin("validateFalse"),
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
			}.MarkShortCircuit(),
		},

		// +k8s:maxProperties
		{
			name: "maxProperties field within limit, subfield validation runs",
			value: &ParentWithMaxProperties{
				Field: TargetWithMaxProperties{
					Value: map[string]string{"a": "1", "b": "2"},
				},
			},
			expectErrors: field.ErrorList{
				field.Invalid(field.NewPath("field", "value"), nil, "forced failure: subfield ParentWithMaxProperties.Field.Value").WithOrigin("validateFalse"),
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
			}.MarkShortCircuit(),
		},
	}

	matcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByDetailExact().MatchShortCircuit()

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			st := localSchemeBuilder.Test(t)
			tester := st.Value(tc.value)
			if tc.oldValue != nil {
				tester.OldValue(tc.oldValue)
			}
			tester.ExpectMatches(matcher, tc.expectErrors)
		})
	}
}
