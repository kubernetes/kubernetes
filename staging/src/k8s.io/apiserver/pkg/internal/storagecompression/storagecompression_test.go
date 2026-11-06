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

package storagecompression

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

var matcher = field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByDetailSubstring().RequireOriginWhenInvalid()

// testErrors checks that got matches want by type, field and origin, plus the
// path a masking error names. It compares the counts too, since one want with
// an origin can match several errors.
func testErrors(t *testing.T, want, got field.ErrorList) {
	t.Helper()
	matcher.Test(t, want, got)
	if len(got) != len(want) {
		t.Errorf("got %d errors, want %d: %v", len(got), len(want), got)
	}
}

// TestValidateMasking checks only what ValidateMasking reports. Errors within
// one entry are ValidateSelectors' to report, so several cases expect nothing
// for a selector its own entry already flags.
func TestValidateMasking(t *testing.T) {
	selector := func(i, j int) *field.Path {
		return field.NewPath("compressionResources").Index(i).Child("resources").Index(j)
	}
	for _, tc := range []struct {
		desc    string
		entries [][]string
		want    field.ErrorList
	}{{
		desc:    "earlier selectors that leave a later one reachable",
		entries: [][]string{{"events", "*.coordination.k8s.io"}, {"*.example.com"}, {"*.other.com"}, {"*."}, {"*.*"}, {"secrets"}},
	}, {
		desc: "every selector an earlier entry masks, reported together",
		entries: [][]string{
			{"pods", "*.example.com"}, {"*.*"}, {"configmaps", "*.coordination.k8s.io"}, {"*."},
			{"widgets.example.com", "pods"}, {"*.example.com"}, {"gadgets.example.com"},
		},
		want: field.ErrorList{
			field.Invalid(selector(2, 0), nil, "compressionResources[1].resources[0]").WithOrigin("masked"),
			field.Invalid(selector(2, 1), nil, "compressionResources[1].resources[0]").WithOrigin("masked"),
			field.Invalid(selector(3, 0), nil, "compressionResources[1].resources[0]").WithOrigin("masked"),
			field.Invalid(selector(4, 0), nil, "compressionResources[0].resources[1]").WithOrigin("masked"),
			field.Duplicate(selector(4, 1), nil),
			field.Duplicate(selector(5, 0), nil),
			// selector(0, 1) and selector(5, 0) are both *.example.com, which covers
			// gadgets.example.com. The error names the first; the second is a repeat.
			field.Invalid(selector(6, 0), nil, "compressionResources[0].resources[1]").WithOrigin("masked"),
		},
	}, {
		desc:    "a chain of masks names the selector at its root",
		entries: [][]string{{"*.*"}, {"*.", "*.apps"}, {"configmaps", "deployments.apps"}},
		want: field.ErrorList{
			field.Invalid(selector(1, 0), nil, "compressionResources[0].resources[0]").WithOrigin("masked"),
			field.Invalid(selector(1, 1), nil, "compressionResources[0].resources[0]").WithOrigin("masked"),
			field.Invalid(selector(2, 0), nil, "compressionResources[0].resources[0]").WithOrigin("masked"),
			field.Invalid(selector(2, 1), nil, "compressionResources[0].resources[0]").WithOrigin("masked"),
		},
	}, {
		desc:    "a selector its own entry repeats is reported once",
		entries: [][]string{{"configmaps"}, {"configmaps", "configmaps"}},
		want:    field.ErrorList{field.Duplicate(selector(1, 0), nil)},
	}, {
		desc:    "a selector reported in its own entry still masks later entries",
		entries: [][]string{{"configmaps", "*."}, {"pods"}},
		want:    field.ErrorList{field.Invalid(selector(1, 0), nil, "compressionResources[0].resources[1]").WithOrigin("masked")},
	}, {
		desc:    "a masked selector names the wider selector its entry already holds",
		entries: [][]string{{"*.*", "*."}, {"pods"}},
		want:    field.ErrorList{field.Invalid(selector(1, 0), nil, "compressionResources[0].resources[0]").WithOrigin("masked")},
	}, {
		desc:    "a selector reported in its own entry passes on the earlier selector that covers it",
		entries: [][]string{{"*.*"}, {"configmaps", "*."}, {"pods"}},
		want: field.ErrorList{
			field.Invalid(selector(1, 0), nil, "compressionResources[0].resources[0]").WithOrigin("masked"),
			field.Invalid(selector(2, 0), nil, "compressionResources[0].resources[0]").WithOrigin("masked"),
		},
	}, {
		desc:    "a masked selector names the first of two identical selectors in an earlier entry",
		entries: [][]string{{"*.apps", "*.apps"}, {"deployments.apps"}},
		want:    field.ErrorList{field.Invalid(selector(1, 0), nil, "compressionResources[0].resources[0]").WithOrigin("masked")},
	}, {
		// '*' parses like '*.', so if bad names took part, the first '*' would mask
		// configmaps and the last '*.' would count as a repeat of it.
		desc:    "masking ignores selectors with a bad name",
		entries: [][]string{{"*"}, {"configmaps"}, {"*.*", "ConfigMaps"}, {"*"}, {"*", "*."}},
		want:    field.ErrorList{field.Invalid(selector(4, 1), nil, "compressionResources[2].resources[0]").WithOrigin("masked")},
	}} {
		t.Run(tc.desc, func(t *testing.T) {
			testErrors(t, tc.want, ValidateMasking(tc.entries, field.NewPath("compressionResources")))
		})
	}
}
