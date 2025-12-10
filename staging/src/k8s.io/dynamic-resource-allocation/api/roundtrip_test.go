/*
Copyright 2024 The Kubernetes Authors.

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

package api_test

import (
    "math/rand"
    "testing"
    "time"

    "k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
    "k8s.io/apimachinery/pkg/api/apitesting/roundtrip"
    metafuzzer "k8s.io/apimachinery/pkg/apis/meta/fuzzer"
    "k8s.io/apimachinery/pkg/runtime"
    "k8s.io/apimachinery/pkg/runtime/schema"
    "k8s.io/apimachinery/pkg/runtime/serializer"

    // DRA conversion packages
    drav1beta1 "k8s.io/dynamic-resource-allocation/api/v1beta1"
    drav1beta2 "k8s.io/dynamic-resource-allocation/api/v1beta2"
    
    // External API packages
    resourcev1 "k8s.io/api/resource/v1"
    resourcev1beta1 "k8s.io/api/resource/v1beta1"
    resourcev1beta2 "k8s.io/api/resource/v1beta2"
)

func TestDynamicResourceAllocationRoundTripFuzz(t *testing.T) {
    scheme := runtime.NewScheme()

    // Add external types to scheme (all versions)
    resourcev1.AddToScheme(scheme)
    resourcev1beta1.AddToScheme(scheme)
    resourcev1beta2.AddToScheme(scheme)
    
    // Add DRA conversion functions to scheme
    drav1beta1.AddToScheme(scheme)
    drav1beta2.AddToScheme(scheme)

    codecs := serializer.NewCodecFactory(scheme)

    seed := time.Now().UnixNano()
    t.Logf("Fuzz seed: %d", seed)

    f := fuzzer.FuzzerFor(metafuzzer.Funcs, rand.NewSource(seed), codecs)

    // Test conversion between all versions via internal types
    kinds := map[schema.GroupVersionKind]bool{
        // v1 types (the hub version)
        resourcev1.SchemeGroupVersion.WithKind("ResourceClaim"):        true,
        resourcev1.SchemeGroupVersion.WithKind("ResourceClaimTemplate"): true,
        resourcev1.SchemeGroupVersion.WithKind("ResourceClass"):         true,
        resourcev1.SchemeGroupVersion.WithKind("ResourceSlice"):         true,
        
        // v1beta1 types
        resourcev1beta1.SchemeGroupVersion.WithKind("ResourceClaim"):        true,
        resourcev1beta1.SchemeGroupVersion.WithKind("ResourceClaimTemplate"): true,
        resourcev1beta1.SchemeGroupVersion.WithKind("ResourceClass"):         true,
        resourcev1beta1.SchemeGroupVersion.WithKind("ResourceSlice"):         true,
        
        // v1beta2 types
        resourcev1beta2.SchemeGroupVersion.WithKind("ResourceClaim"):        true,
        resourcev1beta2.SchemeGroupVersion.WithKind("ResourceClaimTemplate"): true,
        resourcev1beta2.SchemeGroupVersion.WithKind("ResourceClass"):         true,
        resourcev1beta2.SchemeGroupVersion.WithKind("ResourceSlice"):         true,
    }

    roundtrip.RoundTripTypes(t, scheme, codecs, f, kinds)
}