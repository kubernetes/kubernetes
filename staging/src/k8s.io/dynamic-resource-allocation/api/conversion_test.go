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

package api

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	apitest "k8s.io/dynamic-resource-allocation/api/internal/test"
	"sigs.k8s.io/randfill"
)

// v1FillFuncs returns custom fill functions needed for v1 types where
// the default random filling would create objects that cannot round-trip
// due to pointer-to-value conversions in the internal api types.
func v1FillFuncs(codecs serializer.CodecFactory) []interface{} {
	return []interface{}{
		// The internal api types use bool instead of *bool for AllNodes.
		// Converting nil -> false -> &false means nil doesn't round-trip.
		// Ensure AllNodes is always non-nil to avoid this.
		func(s *resourceapi.ResourceSliceSpec, c randfill.Continue) {
			c.FillNoCustom(s)
			if s.AllNodes == nil {
				s.AllNodes = new(bool)
			}
		},
		// The internal api types use bool instead of *bool for BindsToNode.
		// Converting nil -> false -> &false means nil doesn't round-trip.
		// Ensure BindsToNode is always non-nil to avoid this.
		func(d *resourceapi.Device, c randfill.Continue) {
			c.FillNoCustom(d)
			if d.BindsToNode == nil {
				d.BindsToNode = new(bool)
			}
		},
	}
}

// TestConversionRoundTrip verifies that the conversion code for the internal
// DRA API correctly round-trips between v1 and the internal api types.
// For v1 -> api -> v1, the conversion should be lossless.
//
// api -> v1 -> api is not lossless because of
// additional fields in the internal representation.
// Those would get lost during conversion.
// But this doesn't matter because in practice,
// the internal representation is never converted back,
// therefore this direction is not tested.
func TestConversionRoundTrip(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := resourceapi.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	filler := apitest.NewFiller(t, scheme, fuzzer.FuzzerFuncs(v1FillFuncs))

	testCases := []struct {
		name   string
		v1Type func() runtime.Object
		apiNew func() interface{}
	}{
		{
			name:   "ResourceSlice",
			v1Type: func() runtime.Object { return &resourceapi.ResourceSlice{} },
			apiNew: func() interface{} { return &ResourceSlice{} },
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			for i := range apitest.FuzzIterations {
				// Create and fuzz v1 object.
				v1Obj := tc.v1Type()
				filler.Fill(v1Obj)

				// Convert v1 -> api.
				apiObj := tc.apiNew()
				if err := scheme.Convert(v1Obj, apiObj, nil); err != nil {
					t.Fatalf("iteration %d: v1 -> api: %v", i, err)
				}

				// Convert api -> v1 (round-trip).
				roundTripped := tc.v1Type()
				if err := scheme.Convert(apiObj, roundTripped, nil); err != nil {
					t.Fatalf("iteration %d: api -> v1: %v", i, err)
				}

				// The round-tripped object must be equal to the original.
				if !apiequality.Semantic.DeepEqual(v1Obj, roundTripped) {
					t.Errorf("iteration %d: round-trip v1 -> api -> v1 failed\ndiff (-want +got):\n%s",
						i, cmp.Diff(v1Obj, roundTripped))
				}
			}
		})
	}
}
