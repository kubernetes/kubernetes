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
	"reflect"
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

// getGVsWithDeclarativeValidation gets all group/versions supporting declarative validation.
func getGVsWithDeclarativeValidation() []schema.GroupVersion {
	var groupVersions []schema.GroupVersion

	for gvk, rtype := range legacyscheme.Scheme.AllKnownTypes() {
		obj, ok := reflect.New(rtype).Interface().(runtime.Object)
		if !ok || !legacyscheme.Scheme.HasValidationFunc(obj) {
			continue
		}
		groupVersions = append(groupVersions, gvk.GroupVersion())
	}

	return groupVersions
}

func TestVersionedValidationByFuzzing(t *testing.T) {
	typesWithDeclarativeValidation := getGVsWithDeclarativeValidation()

	// subresourceOnly specifies the subresource path for types that can only be validated
	// as subresources (e.g. autoscaling/Scale) and do not support root-level validation.
	// For GVKs not in this map, the test defaults to fuzzing the root resource ("").
	// Other resources with subresources (e.g. Pod status, exec) share validation logic with
	// the root resource, so fuzzing the root is sufficient to verify validation equivalence.
	subresourceOnly := map[schema.GroupVersionKind]string{
		{Group: "autoscaling", Version: "v1", Kind: "Scale"}: "scale",
		{Group: "autoscaling", Version: "v2", Kind: "Scale"}: "scale",
		{Group: "apps", Version: "v1beta1", Kind: "Scale"}:   "scale",
		{Group: "apps", Version: "v1beta2", Kind: "Scale"}:   "scale",
	}

	fuzzIters := *roundtrip.FuzzIters / 10 // TODO: Find a better way to manage test running time
	f := fuzzer.FuzzerFor(FuzzerFuncs, rand.NewSource(rand.Int63()), legacyscheme.Codecs)

	for _, gv := range typesWithDeclarativeValidation {
		for kind := range legacyscheme.Scheme.KnownTypes(gv) {
			gvk := gv.WithKind(kind)
			t.Run(gvk.String(), func(t *testing.T) {
				for range fuzzIters {
					obj, err := legacyscheme.Scheme.New(gvk)
					if err != nil {
						t.Fatalf("could not create a %v: %s", kind, err)
					}

					subresource := ""
					if specific, ok := subresourceOnly[gvk]; ok {
						subresource = specific
					}

					var opts []ValidationTestConfig
					// TODO(API group level configuration): Consider configuring normalization rules at the
					// API group level to avoid potential collisions when multiple rule sets are combined.
					// This would allow each API group to register its own normalization rules independently.
					allRules := append([]field.NormalizationRule{}, resourcevalidation.ResourceNormalizationRules...)
					allRules = append(allRules, nodevalidation.NodeNormalizationRules...)
					opts = append(opts, WithNormalizationRules(allRules...), WithFuzzer(f), WithSkipGroupVersions("extensions/v1beta1"))

					if subresource != "" {
						opts = append(opts, WithSubResources(subresource))
					}

					VerifyVersionedValidationEquivalence(t, obj, nil, opts...)

					old, err := legacyscheme.Scheme.New(gv.WithKind(kind))
					if err != nil {
						t.Fatalf("could not create a %v: %s", kind, err)
					}

					VerifyVersionedValidationEquivalence(t, obj, old, opts...)
				}
			})
		}
	}
}
