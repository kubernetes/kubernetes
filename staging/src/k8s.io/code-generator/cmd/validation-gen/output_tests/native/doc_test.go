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

package native

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestMyObjectValidation(t *testing.T) {
	strPtr := func(s string) *string {
		return &s
	}
	validObj := MyObject{
		UUIDField:                       "a0a2a2d2-0b87-4964-a123-78d00a8787a6",
		UUIDFieldWithoutDV:              "a0a2a2d2-0b87-4964-a123-78d00a8787a6",
		UUIDPtrField:                    strPtr("a0a2a2d2-0b87-4964-a123-78d00a8787a6"),
		UUIDPtrFieldWithoutDV:           strPtr("a0a2a2d2-0b87-4964-a123-78d00a8787a6"),
		UUIDTypedefField:                "a0a2a2d2-0b87-4964-a123-78d00a8787a6",
		UUIDTypedefFieldWithoutDV:       "a0a2a2d2-0b87-4964-a123-78d00a8787a6",
		FieldForLength:                  strings.Repeat("a", 55),
		FieldForLengthWithoutDV:         strings.Repeat("a", 55),
		StableTypeField:                 StableType{InnerField: "abc"},
		StableTypeFieldWithoutDV:        StableType{InnerField: "abc"},
		StableTypeFieldPointer:          &StableType{InnerField: "abc"},
		StableTypeFieldPointerWithoutDV: &StableType{InnerField: "abc"},
		StableTypeSlice:                 []StableType{{InnerField: "abc"}},
		StableTypeSliceWithoutDV:        []StableType{{InnerField: "abc"}},
		SetList:                         []string{"a", "b"},
		SetListWithoutDV:                []string{"a", "b"},
		NestedStable:                    NestedStableType{NestedField: StableType{InnerField: "abc"}, NestedFieldWithoutDV: StableType{InnerField: "abc"}},
		NestedStableWithoutDV:           NestedStableType{NestedField: StableType{InnerField: "abc"}, NestedFieldWithoutDV: StableType{InnerField: "abc"}},
		IPAddress:                       "1.2.3.4",
		IPAddressWithoutDV:              "1.2.3.4",
	}

	tests := []struct {
		name string
		obj  MyObject
		errs field.ErrorList
	}{
		{
			name: "valid",
			obj:  *validObj.DeepCopy(),
		},
		{
			name: "uuid",
			obj: func() MyObject {
				obj := validObj.DeepCopy()
				obj.UUIDField = "invalid"
				obj.UUIDFieldWithoutDV = "invalid"
				obj.UUIDPtrField = strPtr("invalid")
				obj.UUIDPtrFieldWithoutDV = strPtr("invalid")
				return *obj
			}(),
			errs: field.ErrorList{
				field.Invalid(field.NewPath("uuidField"), "invalid", "is not a valid UUID").MarkDeclarativeOnly(),
				field.Invalid(field.NewPath("uuidFieldWithoutDV"), "invalid", "is not a valid UUID"),
				field.Invalid(field.NewPath("uuidPtrField"), "invalid", "is not a valid UUID").MarkDeclarativeOnly(),
				field.Invalid(field.NewPath("uuidPtrFieldWithoutDV"), "invalid", "is not a valid UUID"),
			},
		},
		{
			name: "typedef",
			obj: func() MyObject {
				obj := validObj.DeepCopy()
				obj.UUIDTypedefField = "invalid"
				obj.UUIDTypedefFieldWithoutDV = "invalid"
				return *obj
			}(),
			errs: field.ErrorList{
				field.Invalid(field.NewPath("uuidTypedefField"), UUIDString("invalid"), "is not a valid UUID").MarkDeclarativeOnly(),
				field.Invalid(field.NewPath("uuidTypedefFieldWithoutDV"), UUIDString("invalid"), "is not a valid UUID"),
			},
		},
		{
			name: "required",
			obj: func() MyObject {
				obj := validObj.DeepCopy()
				obj.StableTypeField.InnerField = ""
				obj.StableTypeFieldWithoutDV.InnerField = ""
				return *obj
			}(),
			errs: field.ErrorList{
				field.Required(field.NewPath("stableTypeField", "innerField"), "").MarkDeclarativeOnly(),
				field.Required(field.NewPath("stableTypeFieldWithoutDV", "innerField"), ""),
			},
		},
		{
			name: "slice",
			obj: func() MyObject {
				obj := validObj.DeepCopy()
				obj.StableTypeSlice = []StableType{{InnerField: "a"}, {InnerField: "a"}, {InnerField: "a"}, {InnerField: "a"}, {InnerField: "a"}, {InnerField: "a"}}
				obj.StableTypeSliceWithoutDV = []StableType{{InnerField: "a"}, {InnerField: "a"}, {InnerField: "a"}, {InnerField: "a"}, {InnerField: "a"}, {InnerField: "a"}}
				return *obj
			}(),
			errs: field.ErrorList{
				field.TooMany(field.NewPath("stableTypeSlice"), 6, 5).MarkDeclarativeOnly(),
				field.TooMany(field.NewPath("stableTypeSliceWithoutDV"), 6, 5),
			},
		},
		{
			name: "set",
			obj: func() MyObject {
				obj := validObj.DeepCopy()
				obj.SetList = []string{"a", "a"}
				obj.SetListWithoutDV = []string{"a", "a"}
				return *obj
			}(),
			errs: field.ErrorList{
				field.Duplicate(field.NewPath("setList").Index(1), "a").MarkDeclarativeOnly(),
				field.Duplicate(field.NewPath("setListWithoutDV").Index(1), "a"),
			},
		},
		{
			name: "nested",
			obj: func() MyObject {
				obj := validObj.DeepCopy()
				obj.NestedStable.NestedField.InnerField = ""
				obj.NestedStable.NestedFieldWithoutDV.InnerField = ""
				obj.NestedStableWithoutDV.NestedFieldWithoutDV.InnerField = ""
				obj.NestedStableWithoutDV.NestedField.InnerField = ""
				return *obj
			}(),
			errs: field.ErrorList{
				field.Required(field.NewPath("nestedStable", "nestedField", "innerField"), "").MarkDeclarativeOnly(),
				field.Required(field.NewPath("nestedStable", "nestedFieldWithoutDV", "innerField"), "").MarkDeclarativeOnly(),
				field.Required(field.NewPath("nestedStableWithoutDV", "nestedFieldWithoutDV", "innerField"), ""),
				field.Required(field.NewPath("nestedStableWithoutDV", "nestedField", "innerField"), "").MarkDeclarativeOnly(),
			},
		},
		{
			name: "ip",
			obj: func() MyObject {
				obj := validObj.DeepCopy()
				obj.IPAddress = "invalid"
				obj.IPAddressWithoutDV = "invalid"
				return *obj
			}(),
			errs: field.ErrorList{
				field.Invalid(field.NewPath("ipAddress"), "invalid", "is not a valid IP address").MarkDeclarativeOnly(),
				field.Invalid(field.NewPath("ipAddressWithoutDV"), "invalid", "is not a valid IP address"),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			st := localSchemeBuilder.Test(t)
			st.Value(&tc.obj).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDeclarativeOnly(), tc.errs)
		})
	}
}
