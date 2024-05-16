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

package get

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
)

func TestPrinterSupportsExpectedCustomColumnFormats(t *testing.T) {
	testObject := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}

	customColumnsFile, err := os.CreateTemp(os.TempDir(), "printers_jsonpath_flags")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer func(tempFile *os.File) {
		tempFile.Close()
		os.Remove(tempFile.Name())
	}(customColumnsFile)

	fmt.Fprintf(customColumnsFile, "NAME\n.metadata.name")

	testCases := []struct {
		name               string
		outputFormat       string
		templateArg        string
		expectedError      string
		expectedParseError string
		expectedOutput     string
		expectNoMatch      bool
	}{
		{
			name:           "valid output format also containing the custom-columns argument succeeds",
			outputFormat:   "custom-columns=NAME:.metadata.name",
			expectedOutput: "foo",
		},
		{
			name:          "valid output format and no --template argument results in an error",
			outputFormat:  "custom-columns",
			expectedError: "custom-columns format specified but no custom columns given",
		},
		{
			name:           "valid output format and --template argument succeeds",
			outputFormat:   "custom-columns",
			templateArg:    "NAME:.metadata.name",
			expectedOutput: "foo",
		},
		{
			name:           "custom-columns template file should match, and successfully return correct value",
			outputFormat:   "custom-columns-file",
			templateArg:    customColumnsFile.Name(),
			expectedOutput: "foo",
		},
		{
			name:          "valid output format and invalid --template argument results in a parsing error from the printer",
			outputFormat:  "custom-columns",
			templateArg:   "invalid",
			expectedError: "unexpected custom-columns spec: invalid, expected <header>:<json-path-expr>",
		},
		{
			name:          "no printer is matched on an invalid outputFormat",
			outputFormat:  "invalid",
			expectNoMatch: true,
		},
		{
			name:          "custom-columns printer should not match on any other format supported by another printer",
			outputFormat:  "go-template",
			expectNoMatch: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			printFlags := CustomColumnsPrintFlags{
				TemplateArgument: tc.templateArg,
			}

			p, err := printFlags.ToPrinter(tc.outputFormat)
			if tc.expectNoMatch {
				if !genericclioptions.IsNoCompatiblePrinterError(err) {
					t.Fatalf("expected no printer matches for output format %q", tc.outputFormat)
				}
				return
			}
			if genericclioptions.IsNoCompatiblePrinterError(err) {
				t.Fatalf("expected to match template printer for output format %q", tc.outputFormat)
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
			if len(tc.expectedParseError) > 0 {
				if err == nil || !strings.Contains(err.Error(), tc.expectedParseError) {
					t.Errorf("expecting error %q, got %v", tc.expectedError, err)
				}
				return
			}
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			if !strings.Contains(out.String(), tc.expectedOutput) {
				t.Errorf("unexpected output: expecting %q, got %q", tc.expectedOutput, out.String())
			}
		})
	}
}
