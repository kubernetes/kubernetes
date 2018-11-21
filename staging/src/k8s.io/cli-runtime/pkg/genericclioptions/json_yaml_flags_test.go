/*
Copyright 2018 The Kubernetes Authors.

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

package genericclioptions

import (
	"bytes"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestPrinterSupportsExpectedJSONYamlFormats(t *testing.T) {
	testObject := &v1.Pod{
		TypeMeta:   metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"},
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
	}

	testCases := []struct {
		name           string
		outputFormat   string
		expectedOutput string
		expectNoMatch  bool
	}{
		{
			name:           "json output format matches a json printer",
			outputFormat:   "json",
			expectedOutput: "\"name\": \"foo\"",
		},
		{
			name:           "yaml output format matches a yaml printer",
			outputFormat:   "yaml",
			expectedOutput: "name: foo",
		},
		{
			name:          "output format for another printer does not match a json/yaml printer",
			outputFormat:  "jsonpath",
			expectNoMatch: true,
		},
		{
			name:          "invalid output format results in no match",
			outputFormat:  "invalid",
			expectNoMatch: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			printFlags := JSONYamlPrintFlags{}

			p, err := printFlags.ToPrinter(tc.outputFormat)
			if tc.expectNoMatch {
				if !IsNoCompatiblePrinterError(err) {
					t.Fatalf("expected no printer matches for output format %q", tc.outputFormat)
				}
				return
			}
			if IsNoCompatiblePrinterError(err) {
				t.Fatalf("expected to match template printer for output format %q", tc.outputFormat)
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			out := bytes.NewBuffer([]byte{})
			err = p.PrintObj(testObject, out)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			if !strings.Contains(out.String(), tc.expectedOutput) {
				t.Errorf("unexpected output: expecting %q, got %q", tc.expectedOutput, out.String())
			}
		})
	}
}
