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

package validation

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	apiresource "k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/apis/apiserver"
	"k8s.io/apiserver/pkg/storage/value/compression"
)

func compressionConfig(entries ...apiserver.CompressionResourceConfiguration) *apiserver.StorageCompressionConfiguration {
	return &apiserver.StorageCompressionConfiguration{Resources: entries}
}

func deflateEntry(resources ...string) apiserver.CompressionResourceConfiguration {
	return apiserver.CompressionResourceConfiguration{Resources: resources, Algorithm: apiserver.CompressionAlgorithmDeflate}
}

func minSizeEntry(resource, minSize string) apiserver.CompressionResourceConfiguration {
	e := deflateEntry(resource)
	e.MinSize = new(apiresource.MustParse(minSize))
	return e
}

func TestValidateStorageCompressionConfiguration(t *testing.T) {
	root := field.NewPath("resources")
	resources := root.Index(0).Child("resources")

	for _, tc := range []struct {
		desc string
		in   *apiserver.StorageCompressionConfiguration
		want field.ErrorList
	}{{
		desc: "nil config",
		want: field.ErrorList{field.Required(root, compressionConfigNilErr)},
	}, {
		desc: "no entries compress nothing",
		in:   compressionConfig(),
	}, {
		desc: "an entry with no resources",
		in:   compressionConfig(deflateEntry()),
		want: field.ErrorList{field.Required(resources, fmt.Sprintf(atLeastOneRequiredErrFmt, resources))},
	}, {
		desc: "every selector form",
		in: compressionConfig(
			apiserver.CompressionResourceConfiguration{Resources: []string{"events", "*.coordination.k8s.io"}, Algorithm: apiserver.CompressionAlgorithmNone},
			apiserver.CompressionResourceConfiguration{Resources: []string{"widgets.example.com"}, Algorithm: apiserver.CompressionAlgorithmDeflate, MinSize: new(apiresource.MustParse("4Ki"))},
			deflateEntry("*.apps", "secrets"),
			deflateEntry("*."),
			deflateEntry("*.*"),
		),
	}, {
		desc: "selectors that can't match a stored resource",
		in: compressionConfig(
			deflateEntry("ConfigMaps"), deflateEntry("*"), deflateEntry("serviceipallocations"),
			deflateEntry("events.events.k8s.io"), deflateEntry("*.events.k8s.io"),
			deflateEntry("deployments.extensions"), deflateEntry("configmaps.*"),
		),
		want: field.ErrorList{
			field.Invalid(root.Index(0).Child("resources").Index(0), "ConfigMaps", resourceNameErr),
			field.Invalid(root.Index(1).Child("resources").Index(0), "*", compressionStarResourceErr),
			field.Invalid(root.Index(2).Child("resources").Index(0), "serviceipallocations", compressionNonRESTAPIErr),
			field.Invalid(root.Index(3).Child("resources").Index(0), "events.events.k8s.io", eventsGroupErr),
			field.Invalid(root.Index(4).Child("resources").Index(0), "*.events.k8s.io", eventsGroupErr),
			field.Invalid(root.Index(5).Child("resources").Index(0), "deployments.extensions", compressionExtensionsGroupErr),
			field.Invalid(root.Index(6).Child("resources").Index(0), "configmaps.*", compressionAcrossGroupErr),
		},
	}, {
		desc: "a resource beside its group wildcard",
		in:   compressionConfig(deflateEntry("configmaps", "*.")),
		want: field.ErrorList{field.Invalid(resources, []string{"configmaps", "*."}, compressionOverlapErr)},
	}, {
		desc: "a group wildcard beside the everything wildcard",
		in:   compressionConfig(deflateEntry("*.*", "*.apps")),
		want: field.ErrorList{field.Invalid(resources, []string{"*.*", "*.apps"}, compressionOverlapErr)},
	}, {
		desc: "secrets beside wildcards, which never select it",
		in:   compressionConfig(deflateEntry("secrets", "*.", "*.apps")),
	}, {
		desc: "secrets beside the everything wildcard",
		in:   compressionConfig(deflateEntry("secrets", "*.*")),
	}, {
		desc: "a repeated resource",
		in:   compressionConfig(deflateEntry("secrets", "secrets")),
		want: field.ErrorList{field.Invalid(resources, []string{"secrets", "secrets"}, duplicateResourceErr)},
	}, {
		desc: "a repeated wildcard is a duplicate and an overlap, as in encryption",
		in:   compressionConfig(deflateEntry("*.apps", "*.apps")),
		want: field.ErrorList{
			field.Invalid(resources, []string{"*.apps", "*.apps"}, duplicateResourceErr),
			field.Invalid(resources, []string{"*.apps", "*.apps"}, compressionOverlapErr),
		},
	}, {
		desc: "a missing algorithm",
		in:   compressionConfig(apiserver.CompressionResourceConfiguration{Resources: []string{"configmaps"}}),
		want: field.ErrorList{field.Required(root.Index(0).Child("algorithm"), "")},
	}, {
		desc: "an unknown algorithm",
		in:   compressionConfig(apiserver.CompressionResourceConfiguration{Resources: []string{"configmaps"}, Algorithm: "Zstd"}),
		want: field.ErrorList{field.NotSupported(root.Index(0).Child("algorithm"), apiserver.CompressionAlgorithm("Zstd"), supportedCompressionAlgorithms)},
	}, {
		desc: "minSize at both ends of its range",
		in: compressionConfig(
			minSizeEntry("configmaps", "64"),
			minSizeEntry("pods", "1471Ki"),
			minSizeEntry("services", fmt.Sprint(compression.MaxPlaintextBytes-1)),
		),
	}, {
		desc: "minSize outside its range or not a whole number of bytes",
		in: compressionConfig(
			minSizeEntry("configmaps", "63"),
			minSizeEntry("pods", "1472Ki"),
			minSizeEntry("services", "512m"),
			minSizeEntry("nodes", "-1"),
			minSizeEntry("leases.coordination.k8s.io", "1e30"),
			minSizeEntry("widgets.example.com", "100.5"),
			minSizeEntry("gadgets.example.com", "0.1Ki"),
		),
		want: field.ErrorList{
			field.Invalid(root.Index(0).Child("minSize"), "63", compressionMinSizeRangeErr),
			field.Invalid(root.Index(1).Child("minSize"), "1472Ki", compressionMinSizeRangeErr),
			field.Invalid(root.Index(2).Child("minSize"), "512m", compressionMinSizeRangeErr),
			field.Invalid(root.Index(3).Child("minSize"), "-1", compressionMinSizeRangeErr),
			field.Invalid(root.Index(4).Child("minSize"), "1e30", compressionMinSizeRangeErr),
			field.Invalid(root.Index(5).Child("minSize"), "100500m", compressionMinSizeFractionErr),
			field.Invalid(root.Index(6).Child("minSize"), "102400m", compressionMinSizeFractionErr),
		},
	}, {
		desc: "every selector an earlier entry masks, reported together",
		in: compressionConfig(
			deflateEntry("pods", "*.example.com"),
			deflateEntry("*.*"),
			deflateEntry("configmaps", "*.coordination.k8s.io"),
			deflateEntry("*."),
			deflateEntry("widgets.example.com", "pods"),
		),
		want: field.ErrorList{
			field.Invalid(root.Index(2).Child("resources").Index(0), "configmaps", fmt.Sprintf(compressionMaskedErrFmt, "*.*")),
			field.Invalid(root.Index(2).Child("resources").Index(1), "*.coordination.k8s.io", fmt.Sprintf(compressionMaskedErrFmt, "*.*")),
			field.Invalid(root.Index(3).Child("resources").Index(0), "*.", fmt.Sprintf(compressionMaskedErrFmt, "*.*")),
			field.Invalid(root.Index(4).Child("resources").Index(0), "widgets.example.com", fmt.Sprintf(compressionMaskedErrFmt, "*.example.com")),
			field.Invalid(root.Index(4).Child("resources").Index(1), "pods", fmt.Sprintf(compressionMaskedErrFmt, "pods")),
		},
	}, {
		desc: "a rejected selector neither masks nor is masked",
		in:   compressionConfig(deflateEntry("*"), deflateEntry("configmaps"), deflateEntry("*.*"), deflateEntry("*")),
		want: field.ErrorList{
			field.Invalid(root.Index(0).Child("resources").Index(0), "*", compressionStarResourceErr),
			field.Invalid(root.Index(3).Child("resources").Index(0), "*", compressionStarResourceErr),
		},
	}, {
		desc: "earlier selectors that leave a later one reachable",
		in: compressionConfig(
			apiserver.CompressionResourceConfiguration{Resources: []string{"events", "*.coordination.k8s.io"}, Algorithm: apiserver.CompressionAlgorithmNone},
			deflateEntry("*.example.com"),
			deflateEntry("*.other.com"),
			deflateEntry("*."),
			deflateEntry("*.*"),
			deflateEntry("secrets"),
		),
	}} {
		t.Run(tc.desc, func(t *testing.T) {
			got := ValidateStorageCompressionConfiguration(tc.in)
			if diff := cmp.Diff(tc.want, got, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("ValidateStorageCompressionConfiguration() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
