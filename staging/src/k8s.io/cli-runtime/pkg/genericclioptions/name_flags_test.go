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

func TestNamePrinterSupportsExpectedFormats(t *testing.T) {
	testObject := &v1.Pod{
		TypeMeta:   metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"},
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
	}

	testCases := []struct {
		name           string
		outputFormat   string
		operation      string
		dryRun         bool
		expectedError  string
		expectedOutput string
		expectNoMatch  bool
	}{
		{
			name:           "valid \"name\" output format with no operation prints resource name",
			outputFormat:   "name",
			expectedOutput: "pod/foo",
		},
		{
			name:           "valid \"name\" output format and an operation results in a short-output (non success printer) message",
			outputFormat:   "name",
			operation:      "patched",
			expectedOutput: "pod/foo",
		},
		{
			name:          "operation and no valid \"name\" output does not match a printer",
			operation:     "patched",
			outputFormat:  "invalid",
			dryRun:        true,
			expectNoMatch: true,
		},
		{
			name:           "operation and empty output still matches name printer",
			expectedOutput: "pod/foo patched",
			operation:      "patched",
		},
		{
			name:          "no printer is matched on an invalid outputFormat",
			outputFormat:  "invalid",
			expectNoMatch: true,
		},
		{
			name:          "printer should not match on any other format supported by another printer",
			outputFormat:  "go-template",
			expectNoMatch: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			printFlags := NamePrintFlags{
				Operation: tc.operation,
			}

			p, err := printFlags.ToPrinter(tc.outputFormat)
			if tc.expectNoMatch {
				if !IsNoCompatiblePrinterError(err) {
					t.Fatalf("expected no printer matches for output format %q", tc.outputFormat)
				}
				return
			}
			if IsNoCompatiblePrinterError(err) {
				t.Fatalf("expected to match name printer for output format %q", tc.outputFormat)
			}

			if len(tc.expectedError) > 0 {
				if err == nil || !strings.Contains(err.Error(), tc.expectedError) {
					t.Errorf("expecting error %q, got %v", tc.expectedError, err)
				}
				return
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
