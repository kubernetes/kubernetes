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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// TestSelectableFieldLabelConversionsOfKind verifies that given resource have field
// label conversion defined for each its selectable field.
// fields contains selectable fields of the resource.
// conversions maps deprecated labels to their canonical names.
func TestSelectableFieldLabelConversionsOfKind(t *testing.T, scheme *runtime.Scheme, apiVersion string, kind string, versionedFields, internalFields fields.Set, conversions map[string]string) {
	badFieldLabels := []string{
		"name",
		".name",
		"bad",
		"metadata",
		"foo.bar",
	}

	value := "value"

	gv, err := schema.ParseGroupVersion(apiVersion)
	if err != nil {
		t.Errorf("kind=%s: got unexpected error: %v", kind, err)
		return
	}
	gvk := gv.WithKind(kind)

	if len(versionedFields) == 0 {
		t.Logf("no selectable fields for kind %q, skipping", kind)
	}
	for versionedFieldLabel := range versionedFields {
		if versionedFieldLabel == "name" {
			t.Logf("FIXME: \"name\" is deprecated by \"metadata.name\", it should be removed from selectable fields of kind=%s", kind)
			continue
		}
		convertedFieldLabel, convertedValue, err := scheme.ConvertFieldLabel(gvk, versionedFieldLabel, value)
		if err != nil {
			t.Errorf("kind=%s label=%s: got unexpected error: %v", kind, versionedFieldLabel, err)
			continue
		}
		if _, exists := internalFields[convertedFieldLabel]; !exists {
			t.Errorf("kind=%s label=%s: converted field label not found in internal labels: %q", kind, versionedFieldLabel, convertedFieldLabel)
		}
		expectedLabel := versionedFieldLabel
		if l, exists := conversions[versionedFieldLabel]; exists {
			expectedLabel = l
		}
		if convertedFieldLabel != expectedLabel {
			t.Errorf("kind=%s label=%s: got unexpected label name (%q != %q)", kind, versionedFieldLabel, convertedFieldLabel, expectedLabel)
		}
		if convertedValue != value {
			t.Errorf("kind=%s label=%s: got unexpected new value (%q != %q)", kind, versionedFieldLabel, convertedValue, value)
		}
	}

	for _, label := range badFieldLabels {
		_, _, err := scheme.ConvertFieldLabel(gvk, label, "value")
		if err == nil {
			t.Errorf("kind=%s label=%s: got unexpected non-error", kind, label)
		}
	}
}
