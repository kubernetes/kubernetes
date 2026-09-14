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

// Package test provides test utilities for DRA API conversion testing.
package test

import (
	"math/rand"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	genericfuzzer "k8s.io/apimachinery/pkg/apis/meta/fuzzer"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"sigs.k8s.io/randfill"
)

// FuzzIterations is the number of fuzz iterations to run for each type.
const FuzzIterations = 100

// NewFiller creates a new randfill.Filler for fuzzing objects.
func NewFiller(t *testing.T, scheme *runtime.Scheme, customFuncs fuzzer.FuzzerFuncs) *randfill.Filler {
	seed := rand.Int63()
	t.Logf("Using seed: %d", seed)
	codecs := serializer.NewCodecFactory(scheme)
	return fuzzer.FuzzerFor(
		fuzzer.MergeFuzzerFuncs(genericfuzzer.Funcs, customFuncs),
		rand.NewSource(seed),
		codecs,
	)
}

// ConversionRoundTrip tests that all non-list types in srcGV can be converted
// to dstGV and back without loss of information.
func ConversionRoundTrip(t *testing.T, scheme *runtime.Scheme, filler *randfill.Filler, srcGV, dstGV schema.GroupVersion) {
	t.Helper()

	tested := 0
	for kind := range scheme.KnownTypes(srcGV) {
		if strings.HasSuffix(kind, "List") {
			continue
		}

		srcGVK := srcGV.WithKind(kind)
		dstGVK := dstGV.WithKind(kind)
		if _, err := scheme.New(dstGVK); err != nil {
			// Kind does not exist in the destination version.
			continue
		}

		tested++
		t.Run(kind, func(t *testing.T) {
			for i := range FuzzIterations {
				// Create and fuzz source object.
				srcObj, err := scheme.New(srcGVK)
				if err != nil {
					t.Fatal(err)
				}
				filler.Fill(srcObj)

				// Convert source -> destination.
				dstObj, err := scheme.New(dstGVK)
				if err != nil {
					t.Fatal(err)
				}
				if err := scheme.Convert(srcObj, dstObj, nil); err != nil {
					t.Fatalf("iteration %d: %v -> %v: %v", i, srcGVK, dstGVK, err)
				}

				// Convert destination -> source (round-trip).
				roundTripped, err := scheme.New(srcGVK)
				if err != nil {
					t.Fatal(err)
				}
				if err := scheme.Convert(dstObj, roundTripped, nil); err != nil {
					t.Fatalf("iteration %d: %v -> %v: %v", i, dstGVK, srcGVK, err)
				}

				// The round-tripped object must be equal to the original.
				// Use Semantic.DeepEqual which treats nil and empty
				// slices/maps as equivalent.
				if !apiequality.Semantic.DeepEqual(srcObj, roundTripped) {
					t.Errorf("iteration %d: round-trip %v -> %v -> %v failed\ndiff (-want +got):\n%s",
						i, srcGVK, dstGVK, srcGVK,
						cmp.Diff(srcObj, roundTripped))
				}
			}
		})
	}

	if tested == 0 {
		t.Fatal("no types were tested")
	}
	t.Logf("tested %d types for %v <-> %v", tested, srcGV, dstGV)
}
