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
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestPrinterSupportsExpectedTemplateFormats(t *testing.T) {
	testObject := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}

	templateFile, err := os.CreateTemp("", "printers_jsonpath_flags")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer func(tempFile *os.File) {
		tempFile.Close()
		os.Remove(tempFile.Name())
	}(templateFile)

	fmt.Fprintf(templateFile, "{{ .metadata.name }}")

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
			name:           "valid output format also containing the template argument succeeds",
			outputFormat:   "go-template={{ .metadata.name }}",
			expectedOutput: "foo",
		},
		{
			name:          "valid output format and no template argument results in an error",
			outputFormat:  "template",
			expectedError: "template format specified but no template given",
		},
		{
			name:           "valid output format and template argument succeeds",
			outputFormat:   "go-template",
			templateArg:    "{{ .metadata.name }}",
			expectedOutput: "foo",
		},
		{
			name:           "Go-template file should match, and successfully return correct value",
			outputFormat:   "go-template-file",
			templateArg:    templateFile.Name(),
			expectedOutput: "foo",
		},
		{
			name:           "valid output format and invalid template argument results in the templateArg contents as the output",
			outputFormat:   "go-template",
			templateArg:    "invalid",
			expectedOutput: "invalid",
		},
		{
			name:          "no printer is matched on an invalid outputFormat",
			outputFormat:  "invalid",
			expectNoMatch: true,
		},
		{
			name:          "go-template printer should not match on any other format supported by another printer",
			outputFormat:  "jsonpath",
			expectNoMatch: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			templateArg := &tc.templateArg
			if len(tc.templateArg) == 0 {
				templateArg = nil
			}

			printFlags := GoTemplatePrintFlags{
				TemplateArgument: templateArg,
			}
			if !sort.StringsAreSorted(printFlags.AllowedFormats()) {
				t.Fatalf("allowed formats are not sorted")
			}

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

			if len(out.String()) != len(tc.expectedOutput) {
				t.Errorf("unexpected output: expecting %q, got %q", tc.expectedOutput, out.String())
			}
		})
	}
}

func TestTemplatePrinterDefaultsAllowMissingKeysToTrue(t *testing.T) {
	testObject := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}

	allowMissingKeys := false

	testCases := []struct {
		name             string
		templateArg      string
		expectedOutput   string
		expectedError    string
		allowMissingKeys *bool
	}{
		{
			name:             "existing field does not error and returns expected value",
			templateArg:      "{{ .metadata.name }}",
			expectedOutput:   "foo",
			allowMissingKeys: &allowMissingKeys,
		},
		{
			name:             "missing field does not error and returns no value since missing keys are allowed by default",
			templateArg:      "{{ .metadata.missing }}",
			expectedOutput:   "<no value>",
			allowMissingKeys: nil,
		},
		{
			name:             "missing field returns expected error if field is missing and allowMissingKeys is explicitly set to false",
			templateArg:      "{{ .metadata.missing }}",
			expectedError:    "error executing template",
			allowMissingKeys: &allowMissingKeys,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			printFlags := GoTemplatePrintFlags{
				TemplateArgument: &tc.templateArg,
				AllowMissingKeys: tc.allowMissingKeys,
			}
			if !sort.StringsAreSorted(printFlags.AllowedFormats()) {
				t.Fatalf("allowed formats are not sorted")
			}

			outputFormat := "template"
			p, err := printFlags.ToPrinter(outputFormat)
			if IsNoCompatiblePrinterError(err) {
				t.Fatalf("expected to match template printer for output format %q", outputFormat)
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			out := bytes.NewBuffer([]byte{})
			err = p.PrintObj(testObject, out)

			if len(tc.expectedError) > 0 {
				if err == nil || !strings.Contains(err.Error(), tc.expectedError) {
					t.Errorf("expecting error %q, got %v", tc.expectedError, err)
				}
				return
			}
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			if len(out.String()) != len(tc.expectedOutput) {
				t.Errorf("unexpected output: expecting %q, got %q", tc.expectedOutput, out.String())
			}
		})
	}
}
