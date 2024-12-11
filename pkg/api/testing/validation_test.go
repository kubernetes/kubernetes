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

package testing

import (
	"math/rand"
	"testing"

	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	"k8s.io/apimachinery/pkg/api/apitesting/roundtrip"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

// FIXME: Automatically finds all group/versions supporting declarative validation, or add
// a reflexive test that verifies that they are all registered.
func TestVersionedValidationByFuzzing(t *testing.T) {
	typesWithDeclarativeValidation := []schema.GroupVersion{
		// Registered group versions for versioned validation fuzz testing:
	}

	for _, gv := range typesWithDeclarativeValidation {
		t.Run(gv.String(), func(t *testing.T) {
			for i := 0; i < *roundtrip.FuzzIters; i++ {
				f := fuzzer.FuzzerFor(FuzzerFuncs, rand.NewSource(rand.Int63()), legacyscheme.Codecs)
				for kind := range legacyscheme.Scheme.KnownTypes(gv) {
					obj, err := legacyscheme.Scheme.New(gv.WithKind(kind))
					if err != nil {
						t.Fatalf("could not create a %v: %s", kind, err)
					}
					f.Fill(obj)
					VerifyVersionedValidationEquivalence(t, obj, nil)

					old, err := legacyscheme.Scheme.New(gv.WithKind(kind))
					if err != nil {
						t.Fatalf("could not create a %v: %s", kind, err)
					}
					f.Fill(old)
					VerifyVersionedValidationEquivalence(t, obj, old)
				}
			}
		})
	}
}
