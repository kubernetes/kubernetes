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
	"sort"
	"testing"

	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	"k8s.io/apimachinery/pkg/api/apitesting/roundtrip"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	nodevalidation "k8s.io/kubernetes/pkg/apis/node/validation"
	resourcevalidation "k8s.io/kubernetes/pkg/apis/resource/validation"
)

func TestVersionedValidationByFuzzing(t *testing.T) {
	// Discover every served GroupVersionKind that has declarative validation, so
	// new group/versions are covered automatically without hand maintenance.
	var gvks []schema.GroupVersionKind
	for gvk := range legacyscheme.Scheme.AllKnownTypes() {
		if gvk.Version == runtime.APIVersionInternal {
			continue
		}
		obj, err := legacyscheme.Scheme.New(gvk)
		if err != nil {
			continue
		}
		if !legacyscheme.Scheme.HasValidationFunc(obj) {
			continue
		}
		gvks = append(gvks, gvk)
	}
	sort.Slice(gvks, func(i, j int) bool { return gvks[i].String() < gvks[j].String() })

	// subresourceOnlyKinds maps a Kind to the subresource it must be validated as
	// (e.g. Scale), since it has no root-level validation. Keyed by Kind because
	// the same Kind is registered under several groups (autoscaling, apps). Other
	// kinds default to the root resource (""), whose validation covers their
	// subresources too.
	subresourceOnlyKinds := map[string]string{
		"Scale": "scale",
	}

	fuzzIters := *roundtrip.FuzzIters / 10 // TODO: Find a better way to manage test running time
	f := fuzzer.FuzzerFor(FuzzerFuncs, rand.NewSource(rand.Int63()), legacyscheme.Codecs)

	for _, gvk := range gvks {
		t.Run(gvk.String(), func(t *testing.T) {
			for range fuzzIters {
				obj, err := legacyscheme.Scheme.New(gvk)
				if err != nil {
					t.Fatalf("could not create a %v: %s", gvk.Kind, err)
				}

				subresource := ""
				if specific, ok := subresourceOnlyKinds[gvk.Kind]; ok {
					subresource = specific
				}

				var opts []ValidationTestConfig
				// TODO(API group level configuration): Consider configuring normalization rules at the
				// API group level to avoid potential collisions when multiple rule sets are combined.
				// This would allow each API group to register its own normalization rules independently.
				allRules := append([]field.NormalizationRule{}, resourcevalidation.ResourceNormalizationRules...)
				allRules = append(allRules, nodevalidation.NodeNormalizationRules...)
				opts = append(opts, WithNormalizationRules(allRules...), WithFuzzer(f))

				if subresource != "" {
					opts = append(opts, WithSubResources(subresource))
				}

				VerifyVersionedValidationEquivalence(t, obj, nil, opts...)

				old, err := legacyscheme.Scheme.New(gvk)
				if err != nil {
					t.Fatalf("could not create a %v: %s", gvk.Kind, err)
				}

				VerifyVersionedValidationEquivalence(t, obj, old, opts...)
			}
		})
	}
}
