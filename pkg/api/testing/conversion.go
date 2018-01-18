/*
Copyright 2015 The Kubernetes Authors.

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
	"testing"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

// TestSelectableFieldLabelConversions verifies that given resource have field
// label conversion defined for each its selectable field.
// fields contains selectable fields of the resource.
// labelMap maps deprecated labels to their canonical names.
func TestSelectableFieldLabelConversionsOfKind(t *testing.T, apiVersion string, kind string, fields fields.Set, labelMap map[string]string) {
	badFieldLabels := []string{
		"name",
		".name",
		"bad",
		"metadata",
		"foo.bar",
	}

	value := "value"

	if len(fields) == 0 {
		t.Logf("no selectable fields for kind %q, skipping", kind)
	}
	for label := range fields {
		if label == "name" {
			t.Logf("FIXME: \"name\" is deprecated by \"metadata.name\", it should be removed from selectable fields of kind=%s", kind)
			continue
		}
		newLabel, newValue, err := legacyscheme.Scheme.ConvertFieldLabel(apiVersion, kind, label, value)
		if err != nil {
			t.Errorf("kind=%s label=%s: got unexpected error: %v", kind, label, err)
		} else {
			expectedLabel := label
			if l, exists := labelMap[label]; exists {
				expectedLabel = l
			}
			if newLabel != expectedLabel {
				t.Errorf("kind=%s label=%s: got unexpected label name (%q != %q)", kind, label, newLabel, expectedLabel)
			}
			if newValue != value {
				t.Errorf("kind=%s label=%s: got unexpected new value (%q != %q)", kind, label, newValue, value)
			}
		}
	}

	for _, label := range badFieldLabels {
		_, _, err := legacyscheme.Scheme.ConvertFieldLabel(apiVersion, kind, label, "value")
		if err == nil {
			t.Errorf("kind=%s label=%s: got unexpected non-error", kind, label)
		}
	}
}
