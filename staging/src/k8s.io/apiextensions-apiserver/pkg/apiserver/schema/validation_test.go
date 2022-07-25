/*
Copyright 2019 The Kubernetes Authors.

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

package schema

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	fuzz "github.com/google/gofuzz"
)

func TestValidateStructuralMetadataInvariants(t *testing.T) {
	fuzzer := fuzz.New()
	fuzzer.Funcs(
		func(s *JSON, c fuzz.Continue) {
			if c.RandBool() {
				s.Object = float64(42.0)
			}
		},
		func(s **StructuralOrBool, c fuzz.Continue) {
			if c.RandBool() {
				*s = &StructuralOrBool{}
			}
		},
		func(s **Structural, c fuzz.Continue) {
			if c.RandBool() {
				*s = &Structural{}
			}
		},
		func(s *Structural, c fuzz.Continue) {
			if c.RandBool() {
				*s = Structural{}
			}
		},
		func(vv **NestedValueValidation, c fuzz.Continue) {
			if c.RandBool() {
				*vv = &NestedValueValidation{}
			}
		},
		func(vv *NestedValueValidation, c fuzz.Continue) {
			if c.RandBool() {
				*vv = NestedValueValidation{}
			}
		},
	)
	fuzzer.NilChance(0)

	// check that type must be object
	typeNames := []string{"object", "array", "number", "integer", "boolean", "string"}
	for _, typeName := range typeNames {
		s := Structural{
			Generic: Generic{
				Type: typeName,
			},
		}

		errs := validateStructuralMetadataInvariants(&s, true, rootLevel, nil)
		if len(errs) != 0 {
			t.Logf("errors returned: %v", errs)
		}
		if len(errs) != 0 && typeName == "object" {
			t.Errorf("unexpected forbidden field validation errors for: %#v", s)
		}
		if len(errs) == 0 && typeName != "object" {
			t.Errorf("expected forbidden field validation errors for: %#v", s)
		}
	}

	// check that anything other than name and generateName of ObjectMeta in metadata properties is forbidden
	tt := reflect.TypeOf(metav1.ObjectMeta{})
	for i := 0; i < tt.NumField(); i++ {
		property := tt.Field(i).Name
		s := &Structural{
			Generic: Generic{
				Type: "object",
			},
			Properties: map[string]Structural{
				property: {},
			},
		}

		errs := validateStructuralMetadataInvariants(s, true, rootLevel, nil)
		if len(errs) != 0 {
			t.Logf("errors returned: %v", errs)
		}
		if len(errs) != 0 && (property == "name" || property == "generateName") {
			t.Errorf("unexpected forbidden field validation errors for: %#v", s)
		}
		if len(errs) == 0 && property != "name" && property != "generateName" {
			t.Errorf("expected forbidden field validation errors for: %#v", s)
		}
	}

	// check that anything other than type and properties in metadata is forbidden
	tt = reflect.TypeOf(Structural{})
	for i := 0; i < tt.NumField(); i++ {
		s := Structural{}
		x := reflect.ValueOf(&s).Elem()
		fuzzer.Fuzz(x.Field(i).Addr().Interface())
		s.Type = "object"
		s.Properties = map[string]Structural{
			"name":         {},
			"generateName": {},
		}
		s.Default.Object = nil // this is checked in API validation, we don't need to test it here

		valid := reflect.DeepEqual(s, Structural{
			Generic: Generic{
				Type: "object",
				Default: JSON{
					Object: nil,
				},
			},
			Properties: map[string]Structural{
				"name":         {},
				"generateName": {},
			},
		})

		errs := validateStructuralMetadataInvariants(s.DeepCopy(), true, rootLevel, nil)
		if len(errs) != 0 {
			t.Logf("errors returned: %v", errs)
		}
		if len(errs) != 0 && valid {
			t.Errorf("unexpected forbidden field validation errors for: %#v", s)
		}
		if len(errs) == 0 && !valid {
			t.Errorf("expected forbidden field validation errors for: %#v", s)
		}
	}
}

func TestValidateNestedValueValidationComplete(t *testing.T) {
	fuzzer := fuzz.New()
	fuzzer.Funcs(
		func(s *JSON, c fuzz.Continue) {
			if c.RandBool() {
				s.Object = float64(42.0)
			}
		},
		func(s **StructuralOrBool, c fuzz.Continue) {
			if c.RandBool() {
				*s = &StructuralOrBool{}
			}
		},
	)
	fuzzer.NilChance(0)

	// check that we didn't forget to check any forbidden generic field
	tt := reflect.TypeOf(Generic{})
	for i := 0; i < tt.NumField(); i++ {
		vv := &NestedValueValidation{}
		x := reflect.ValueOf(&vv.ForbiddenGenerics).Elem()
		fuzzer.Fuzz(x.Field(i).Addr().Interface())

		errs := validateNestedValueValidation(vv, false, false, fieldLevel, nil)
		if len(errs) == 0 && !reflect.DeepEqual(vv.ForbiddenGenerics, Generic{}) {
			t.Errorf("expected ForbiddenGenerics validation errors for: %#v", vv)
		}
	}

	// check that we didn't forget to check any forbidden extension field
	tt = reflect.TypeOf(Extensions{})
	for i := 0; i < tt.NumField(); i++ {
		vv := &NestedValueValidation{}
		x := reflect.ValueOf(&vv.ForbiddenExtensions).Elem()
		fuzzer.Fuzz(x.Field(i).Addr().Interface())

		errs := validateNestedValueValidation(vv, false, false, fieldLevel, nil)
		if len(errs) == 0 && !reflect.DeepEqual(vv.ForbiddenExtensions, Extensions{}) {
			t.Errorf("expected ForbiddenExtensions validation errors for: %#v", vv)
		}
	}
}
