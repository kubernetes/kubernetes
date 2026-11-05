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
	"context"
	"strconv"
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	apiresource "k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/internal/storagecompression"
)

func compressionConfig(entries ...CompressionResourceConfiguration) *StorageCompressionConfiguration {
	return &StorageCompressionConfiguration{CompressionResources: entries}
}

func deflateEntry(resources ...string) CompressionResourceConfiguration {
	return CompressionResourceConfiguration{Resources: resources, Algorithm: CompressionAlgorithmDeflate}
}

func minSizeEntry(resource, minSize string) CompressionResourceConfiguration {
	e := deflateEntry(resource)
	e.MinSize = new(apiresource.MustParse(minSize))
	return e
}

// TestValidateStorageCompressionConfiguration covers the +k8s: tags. The
// selector checks run in the loader's validateConfig, and storagecompression
// tests them. The floor cases pin +k8s:minimum=64: DEFLATE almost never shrinks
// a smaller value, which then costs a 9-byte frame and the CPU.
func TestValidateStorageCompressionConfiguration(t *testing.T) {
	root := field.NewPath("compressionResources")
	resources := root.Index(0).Child("resources")
	minSize := func(i int) *field.Path { return root.Index(i).Child("minSize") }
	matcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByDetailSubstring().RequireOriginWhenInvalid()

	for _, tc := range []struct {
		desc string
		in   *StorageCompressionConfiguration
		want field.ErrorList
	}{{
		desc: "no entries compress nothing",
		in:   compressionConfig(),
	}, {
		desc: "an entry with no resources",
		in:   compressionConfig(deflateEntry()),
		want: field.ErrorList{field.Required(resources, "")},
	}, {
		desc: "every selector form",
		in: compressionConfig(
			CompressionResourceConfiguration{Resources: []string{"events", "*.coordination.k8s.io"}, Algorithm: CompressionAlgorithmNone},
			CompressionResourceConfiguration{Resources: []string{"widgets.example.com"}, Algorithm: CompressionAlgorithmDeflate, MinSize: new(apiresource.MustParse("4Ki"))},
			deflateEntry("*.apps", "secrets"),
			deflateEntry("*."),
			deflateEntry("*.*"),
		),
	}, {
		desc: "a missing algorithm",
		in:   compressionConfig(CompressionResourceConfiguration{Resources: []string{"configmaps"}}),
		want: field.ErrorList{field.Required(root.Index(0).Child("algorithm"), "")},
	}, {
		desc: "an unknown algorithm, including a known one in another case",
		in: compressionConfig(
			CompressionResourceConfiguration{Resources: []string{"configmaps"}, Algorithm: "Zstd"},
			CompressionResourceConfiguration{Resources: []string{"pods"}, Algorithm: "deflate"},
		),
		want: field.ErrorList{
			field.NotSupported[CompressionAlgorithm](root.Index(0).Child("algorithm"), nil, nil),
			field.NotSupported[CompressionAlgorithm](root.Index(1).Child("algorithm"), nil, nil),
		},
	}, {
		desc: "minSize anywhere in its range, fractions included",
		in: compressionConfig(
			minSizeEntry("configmaps", "64"),
			minSizeEntry("secrets", "64.0"),
			minSizeEntry("pods", "1471Ki"),
			minSizeEntry("widgets.example.com", "100.5"),
			minSizeEntry("gadgets.example.com", "0.1Ki"),
			// +k8s:maximum takes a literal, 1507327, not MaxPlaintextBytes. This row
			// and the next case fail if the two drift apart.
			minSizeEntry("services", strconv.Itoa(storagecompression.MaxPlaintextBytes-1)),
		),
	}, {
		desc: "minSize at the write ceiling",
		in:   compressionConfig(minSizeEntry("configmaps", strconv.Itoa(storagecompression.MaxPlaintextBytes))),
		want: field.ErrorList{field.Invalid(minSize(0), nil, "").WithOrigin("maximum")},
	}, {
		desc: "minSize outside its range",
		in: compressionConfig(
			minSizeEntry("configmaps", "63"),
			minSizeEntry("pods", "1472Ki"),
			minSizeEntry("services", "512m"),
			minSizeEntry("nodes", "-1"),
			minSizeEntry("leases.coordination.k8s.io", "1e30"),
			minSizeEntry("endpoints", "63.999"),
			minSizeEntry("events", "-1e30"),
			minSizeEntry("widgets.example.com", "1507327.5"),
		),
		want: field.ErrorList{
			field.Invalid(minSize(0), nil, "").WithOrigin("minimum"),
			field.Invalid(minSize(1), nil, "").WithOrigin("maximum"),
			field.Invalid(minSize(2), nil, "").WithOrigin("minimum"),
			field.Invalid(minSize(3), nil, "").WithOrigin("minimum"),
			field.Invalid(minSize(4), nil, "").WithOrigin("maximum"),
			field.Invalid(minSize(5), nil, "").WithOrigin("minimum"),
			field.Invalid(minSize(6), nil, "").WithOrigin("minimum"),
			field.Invalid(minSize(7), nil, "").WithOrigin("maximum"),
		},
	}} {
		t.Run(tc.desc, func(t *testing.T) {
			got := Validate_StorageCompressionConfiguration(context.Background(), operation.Operation{Type: operation.Create}, nil, tc.in, nil)
			matcher.Test(t, tc.want, got)
			// A want with an origin can absorb several errors, so count them too.
			if len(got) != len(tc.want) {
				t.Errorf("got %d errors, want %d: %v", len(got), len(tc.want), got)
			}
		})
	}
}
