/*
Copyright 2016 The Kubernetes Authors.

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

package main

import (
	"fmt"
	"reflect"
	"testing"
)

func TestConformance(t *testing.T) {
	for _, tc := range []struct {
		desc        string
		filename    string
		code        string
		targetFrame frame
		output      *ConformanceData
	}{
		{
			desc:     "Grabs comment above test",
			filename: "test/list/main_test.go",
			code: `package test

	var num = 3
	func Helper(x int) { return x / 0 }
	var _ = Describe("Feature", func() {
	/*
	   Testname: Kubelet-OutputToLogs
	   Description: By default the stdout and stderr from the process
	   being executed in a pod MUST be sent to the pod's logs.
	*/
	 framework.ConformanceIt("validates describe with ConformanceIt", func() {})
	})`,
			output: &ConformanceData{
				URL:         "https://github.com/kubernetes/kubernetes/tree/master/test/list/main_test.go#L11",
				TestName:    "Kubelet-OutputToLogs",
				Description: `By default the stdout and stderr from the process being executed in a pod MUST be sent to the pod's logs.`,
				File:        "test/list/main_test.go",
			},
			targetFrame: frame{File: "test/list/main_test.go", Line: 11},
		}, {
			desc:     "Handles extra spaces",
			filename: "e2e/foo.go",
			code: `package test

	var _ = SIGDescribe("Feature", func() {
		   Context("with context and extra spaces before It block should still pick up Testname", func() {
				   //                                      Testname: Test with spaces
				   //Description: Should pick up testname even if it is not within 3 spaces
				   //even when executed from memory.
				   framework.ConformanceIt("should work", func() {})
		   })
	})`,
			output: &ConformanceData{
				URL:         "https://github.com/kubernetes/kubernetes/tree/master/e2e/foo.go#L8",
				TestName:    "Test with spaces",
				Description: `Should pick up testname even if it is not within 3 spaces even when executed from memory.`,
				File:        "e2e/foo.go",
			},
			targetFrame: frame{File: "e2e/foo.go", Line: 8},
		}, {
			desc:     "Should target the correct comment based on the line numbers (second)",
			filename: "e2e/foo.go",
			code: `package test

	var _ = SIGDescribe("Feature", func() {
		   Context("with context and extra spaces before It block should still pick up Testname", func() {
				   // Testname: First test
				   // Description: Should pick up testname even if it is not within 3 spaces
				   // even when executed from memory.
				   framework.ConformanceIt("should work", func() {})

				   // Testname: Second test
				   // Description: Should target the correct test/comment based on the line numbers
				   framework.ConformanceIt("should work", func() {})
		   })
	})`,
			output: &ConformanceData{
				URL:         "https://github.com/kubernetes/kubernetes/tree/master/e2e/foo.go#L13",
				TestName:    "Second test",
				Description: `Should target the correct test/comment based on the line numbers`,
				File:        "e2e/foo.go",
			},
			targetFrame: frame{File: "e2e/foo.go", Line: 13},
		}, {
			desc:     "Should target the correct comment based on the line numbers (first)",
			filename: "e2e/foo.go",
			code: `package test

	var _ = SIGDescribe("Feature", func() {
		   Context("with context and extra spaces before It block should still pick up Testname", func() {
				   // Testname: First test
				   // Description: Should target the correct test/comment based on the line numbers
				   framework.ConformanceIt("should work", func() {})

				   // Testname: Second test
				   // Description: Should target the correct test/comment based on the line numbers
				   framework.ConformanceIt("should work", func() {})
		   })
	})`,
			output: &ConformanceData{
				URL:         "https://github.com/kubernetes/kubernetes/tree/master/e2e/foo.go#L8",
				TestName:    "First test",
				Description: `Should target the correct test/comment based on the line numbers`,
				File:        "e2e/foo.go",
			},
			targetFrame: frame{File: "e2e/foo.go", Line: 8},
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			*confDoc = true
			cd, err := scanFileForFrame(tc.filename, tc.code, tc.targetFrame)
			if err != nil {
				panic(err)
			}
			if !reflect.DeepEqual(cd, tc.output) {
				t.Errorf("code:\n%s\ngot  %+v\nwant %+v",
					tc.code, cd, tc.output)
			}
		})
	}
}

func TestCommentToConformanceData(t *testing.T) {
	tcs := []struct {
		desc     string
		input    string
		expected *ConformanceData
	}{
		{
			desc: "Empty comment leads to nil",
		}, {
			desc:  "No Release or Testname leads to nil",
			input: "Description: foo",
		}, {
			desc:  "Release but no Testname should result in nil",
			input: "Release: v1.1\nDescription: foo",
		}, {
			desc:     "Testname but no Release does not result in nil",
			input:    "Testname: mytest\nDescription: foo",
			expected: &ConformanceData{TestName: "mytest", Description: "foo"},
		}, {
			desc:     "All fields parsed and newlines and whitespace removed from description",
			input:    "Release: v1.1\n\t\tTestname: mytest\n\t\tDescription: foo\n\t\tbar\ndone",
			expected: &ConformanceData{TestName: "mytest", Release: "v1.1", Description: "foo bar done"},
		},
	}

	for _, tc := range tcs {
		t.Run(tc.desc, func(t *testing.T) {
			out := commentToConformanceData(tc.input)
			if !reflect.DeepEqual(out, tc.expected) {
				t.Errorf("Expected %#v but got %#v", tc.expected, out)
			}
		})
	}
}

func TestValidateTestName(t *testing.T) {
	testCases := []struct {
		testName  string
		tagString string
	}{
		{
			"a test case with no tags",
			"",
		},
		{
			"a test case with valid tags [LinuxOnly] [NodeConformance] [Serial] [Disruptive]",
			"",
		},
		{
			"a flaky test case that is invalid [Flaky]",
			"[Flaky]",
		},
		{
			"a feature test case that is invalid [Feature:Awesome]",
			"[Feature:Awesome]",
		},
		{
			"an alpha test case that is invalid [Alpha]",
			"[Alpha]",
		},
		{
			"a test case with multiple invalid tags [Flaky] [Feature:Awesome] [Alpha]",
			"[Flaky],[Feature:Awesome],[Alpha]",
		},
		{
			"[sig-awesome] [Alpha] [Disruptive] a test case with valid and invalid tags [Serial] [Flaky]",
			"[Alpha],[Flaky]",
		},
	}
	for i, tc := range testCases {
		err := validateTestName(tc.testName)
		if err != nil {
			if tc.tagString == "" {
				t.Errorf("test case[%d]: expected no validate error, got %q", i, err.Error())
			} else {
				expectedMsg := fmt.Sprintf("'%s' cannot have invalid tags %s", tc.testName, tc.tagString)
				actualMsg := err.Error()
				if actualMsg != expectedMsg {
					t.Errorf("test case[%d]: expected error message %q, got %q", i, expectedMsg, actualMsg)
				}
			}
		}
	}
}
