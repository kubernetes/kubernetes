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
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	nodevalidation "k8s.io/kubernetes/pkg/apis/node/validation"
	resourcevalidation "k8s.io/kubernetes/pkg/apis/resource/validation"
)

// FIXME: Automatically finds all group/versions supporting declarative validation, or add
// a reflexive test that verifies that they are all registered.
func TestVersionedValidationByFuzzing(t *testing.T) {
	typesWithDeclarativeValidation := []schema.GroupVersion{
		// Registered group versions for versioned validation fuzz testing:
		{Group: "", Version: "v1"},
		{Group: "certificates.k8s.io", Version: "v1"},
		{Group: "certificates.k8s.io", Version: "v1alpha1"},
		{Group: "certificates.k8s.io", Version: "v1beta1"},
		{Group: "resource.k8s.io", Version: "v1beta1"},
		{Group: "resource.k8s.io", Version: "v1beta2"},
		{Group: "resource.k8s.io", Version: "v1"},
		{Group: "storage.k8s.io", Version: "v1"},
		{Group: "storage.k8s.io", Version: "v1beta1"},
		{Group: "storage.k8s.io", Version: "v1alpha1"},
		{Group: "node.k8s.io", Version: "v1beta1"},
		{Group: "node.k8s.io", Version: "v1"},
		{Group: "node.k8s.io", Version: "v1alpha1"},
	}

	fuzzIters := *roundtrip.FuzzIters / 10 // TODO: Find a better way to manage test running time
	f := fuzzer.FuzzerFor(FuzzerFuncs, rand.NewSource(rand.Int63()), legacyscheme.Codecs)

	for _, gv := range typesWithDeclarativeValidation {
		for kind := range legacyscheme.Scheme.KnownTypes(gv) {
			gvk := gv.WithKind(kind)
			t.Run(gvk.String(), func(t *testing.T) {
				for i := 0; i < fuzzIters; i++ {
					obj, err := legacyscheme.Scheme.New(gvk)
					if err != nil {
						t.Fatalf("could not create a %v: %s", kind, err)
					}
					f.Fill(obj)

					var opts []ValidationTestConfig
					// TODO(API group level configuration): Consider configuring normalization rules at the
					// API group level to avoid potential collisions when multiple rule sets are combined.
					// This would allow each API group to register its own normalization rules independently.
					allRules := append([]field.NormalizationRule{}, resourcevalidation.ResourceNormalizationRules...)
					allRules = append(allRules, nodevalidation.NodeNormalizationRules...)
					opts = append(opts, WithNormalizationRules(allRules...))

					VerifyVersionedValidationEquivalence(t, obj, nil, opts...)

					old, err := legacyscheme.Scheme.New(gv.WithKind(kind))
					if err != nil {
						t.Fatalf("could not create a %v: %s", kind, err)
					}
					f.Fill(old)
					VerifyVersionedValidationEquivalence(t, obj, old, opts...)
				}
			})
		}
	}
}
