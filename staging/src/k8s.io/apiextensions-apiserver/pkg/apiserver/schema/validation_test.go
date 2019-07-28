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

	fuzz "github.com/google/gofuzz"

	"k8s.io/apimachinery/pkg/util/rand"
)

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
		i := rand.Intn(x.NumField())
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
		i := rand.Intn(x.NumField())
		fuzzer.Fuzz(x.Field(i).Addr().Interface())

		errs := validateNestedValueValidation(vv, false, false, fieldLevel, nil)
		if len(errs) == 0 && !reflect.DeepEqual(vv.ForbiddenExtensions, Extensions{}) {
			t.Errorf("expected ForbiddenExtensions validation errors for: %#v", vv)
		}
	}
}
