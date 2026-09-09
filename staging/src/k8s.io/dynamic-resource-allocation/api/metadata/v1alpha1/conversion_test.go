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

package v1alpha1

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	apitest "k8s.io/dynamic-resource-allocation/api/internal/test"
	"k8s.io/dynamic-resource-allocation/api/metadata"
)

// TestConversionRoundTrip verifies that the automatically generated
// conversion code for the metadata API correctly round-trips between
// v1alpha1 and internal. For each non-list type registered in v1alpha1,
// a fuzzed object is converted to the internal version and back, and the
// result must be equal to the original.
//
// Note that this only covers conversion code which is called
// while converting the top-level API types. Types embedded
// inside those have their own conversion functions, but those
// are not necessarily called.
func TestConversionRoundTrip(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := metadata.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	filler := apitest.NewFiller(t, scheme, nil)

	apitest.ConversionRoundTrip(t, scheme, filler,
		SchemeGroupVersion,
		schema.GroupVersion{Group: GroupName, Version: runtime.APIVersionInternal},
	)
}
