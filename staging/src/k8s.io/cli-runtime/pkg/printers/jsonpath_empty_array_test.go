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

package printers

import (
	"bytes"
	"encoding/json"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func TestJSONPathPrinterRetainsUnstructuredValuesAfterEmptyArray(t *testing.T) {
	// Decode a JSON array instead of depending on typed API empty-slice serialization.
	raw := []byte(`{"apiVersion":"example.test/v1","kind":"WidgetList","items":[{"spec":{"values":[]}},{"spec":{"values":["second"]}}]}`)
	object := &unstructured.Unstructured{}
	if err := json.Unmarshal(raw, &object.Object); err != nil {
		t.Fatal(err)
	}
	printer, err := NewJSONPathPrinter("{.items[*].spec.values[*]}")
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := printer.PrintObj(object, &out); err != nil {
		t.Fatal(err)
	}
	if out.String() != "second" {
		t.Fatalf("got %q, want second from the later input", out.String())
	}
}
