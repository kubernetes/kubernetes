/*
Copyright 2017 The Kubernetes Authors.

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
	"regexp"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func TestPrintUnstructuredObject(t *testing.T) {
	tests := []struct {
		expected string
		options  PrintOptions
	}{
		{
			expected: "NAME\\s+KIND\\s+DUMMY 1\\s+DUMMY 2\\s+ITEMS\nMyName\\s+Test\\.v1\\.\\s+present\\s+present\\s+1 item\\(s\\)",
		},
		{
			options: PrintOptions{
				WithNamespace: true,
			},
			expected: "NAMESPACE\\s+NAME\\s+KIND\\s+DUMMY 1\\s+DUMMY 2\nMyNamespace\\s+MyName\\s+Test\\.v1\\.\\s+present\\s+present",
		},
		{
			options: PrintOptions{
				ShowLabels:    true,
				WithNamespace: true,
			},
			expected: "NAMESPACE\\s+NAME\\s+KIND\\s+DUMMY 1\\s+LABELS\nMyNamespace\\s+MyName\\s+Test\\.v1\\.\\s+present\\s+<none>",
		},
	}
	out := bytes.NewBuffer([]byte{})

	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Test",
			"dummy1":     "present",
			"dummy2":     "present",
			"metadata": map[string]interface{}{
				"name":              "MyName",
				"namespace":         "MyNamespace",
				"creationTimestamp": "2017-04-01T00:00:00Z",
				"resourceVersion":   123,
				"uid":               "00000000-0000-0000-0000-000000000001",
				"dummy3":            "present",
			},
			"items": []interface{}{
				map[string]interface{}{
					"itemBool": true,
					"itemInt":  42,
				},
			},
			"url":    "http://localhost",
			"status": "ok",
		},
	}

	for _, test := range tests {
		printer := &HumanReadablePrinter{
			options: test.options,
		}
		printer.PrintObj(obj, out)

		matches, err := regexp.MatchString(test.expected, out.String())
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !matches {
			t.Errorf("wanted %s, got %s", test.expected, out)
		}
	}
}
