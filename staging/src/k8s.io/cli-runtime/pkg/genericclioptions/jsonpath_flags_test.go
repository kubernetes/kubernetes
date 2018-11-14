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
	"io/ioutil"
	"os"
	"sort"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestPrinterSupportsExpectedJSONPathFormats(t *testing.T) {
	testObject := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}

	jsonpathFile, err := ioutil.TempFile("", "printers_jsonpath_flags")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer func(tempFile *os.File) {
		tempFile.Close()
		os.Remove(tempFile.Name())
	}(jsonpathFile)

	fmt.Fprintf(jsonpathFile, "{ .metadata.name }\n")

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
			name:           "valid output format also containing the jsonpath argument succeeds",
			outputFormat:   "jsonpath={ .metadata.name }",
			expectedOutput: "foo",
		},
		{
			name:          "valid output format and no --template argument results in an error",
			outputFormat:  "jsonpath",
			expectedError: "template format specified but no template given",
		},
		{
			name:           "valid output format and --template argument succeeds",
			outputFormat:   "jsonpath",
			templateArg:    "{ .metadata.name }",
			expectedOutput: "foo",
		},
		{
			name:           "jsonpath template file should match, and successfully return correct value",
			outputFormat:   "jsonpath-file",
			templateArg:    jsonpathFile.Name(),
			expectedOutput: "foo",
		},
		{
			name:               "valid output format and invalid --template argument results in a parsing from the printer",
			outputFormat:       "jsonpath",
			templateArg:        "{invalid}",
			expectedParseError: "unrecognized identifier invalid",
		},
		{
			name:          "no printer is matched on an invalid outputFormat",
			outputFormat:  "invalid",
			expectNoMatch: true,
		},
		{
			name:          "jsonpath printer should not match on any other format supported by another printer",
			outputFormat:  "go-template",
			expectNoMatch: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			templateArg := &tc.templateArg
			if len(tc.templateArg) == 0 {
				templateArg = nil
			}

			printFlags := JSONPathPrintFlags{
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

func TestJSONPathPrinterDefaultsAllowMissingKeysToTrue(t *testing.T) {
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
			templateArg:      "{ .metadata.name }",
			expectedOutput:   "foo",
			allowMissingKeys: &allowMissingKeys,
		},
		{
			name:             "missing field does not error and returns an empty string since missing keys are allowed by default",
			templateArg:      "{ .metadata.missing }",
			expectedOutput:   "",
			allowMissingKeys: nil,
		},
		{
			name:             "missing field returns expected error if field is missing and allowMissingKeys is explicitly set to false",
			templateArg:      "{ .metadata.missing }",
			expectedError:    "error executing jsonpath",
			allowMissingKeys: &allowMissingKeys,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			printFlags := JSONPathPrintFlags{
				TemplateArgument: &tc.templateArg,
				AllowMissingKeys: tc.allowMissingKeys,
			}
			if !sort.StringsAreSorted(printFlags.AllowedFormats()) {
				t.Fatalf("allowed formats are not sorted")
			}

			outputFormat := "jsonpath"
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
