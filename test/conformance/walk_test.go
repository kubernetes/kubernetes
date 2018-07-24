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
	"reflect"
	"testing"
)

var conformanceCases = []struct {
	filename string
	code     string
	output   []conformanceData
}{
	// Go unit test
	{"test/list/main_test.go", `
var num = 3
func Helper(x int) { return x / 0 }
var _ = Describe("Feature", func() {
/*
   Testname: Kubelet-OutputToLogs
   Description: By default the stdout and stderr from the process
   being executed in a pod MUST be sent to the pod's logs.
*/
 framework.ConformanceIt("validates describe with ConformanceIt", func() {})
})`, []conformanceData{{URL: "https://github.com/kubernetes/kubernetes/tree/master/test/list/main_test.go#L11", TestName: "Kubelet-OutputToLogs",
		Description: `By default the stdout and stderr from the process
being executed in a pod MUST be sent to the pod's logs.` + "\n\n"}},
	},
	// Describe + It
	{"e2e/foo.go", `
var _ = Describe("Feature", func() {
	//It should have comment
	framework.ConformanceIt("should work properly", func() {})
})`, []conformanceData{{URL: "https://github.com/kubernetes/kubernetes/tree/master/e2e/foo.go#L5", TestName: "Feature should work properly", Description: "It should have comment\n\n"}},
	},
	// KubeDescribe + It
	{"e2e/foo.go", `
var _ = framework.KubeDescribe("Feature", func() {
	/*It should have comment*/
	framework.ConformanceIt("should work properly", func() {})
})`, []conformanceData{{URL: "https://github.com/kubernetes/kubernetes/tree/master/e2e/foo.go#L5", TestName: "Feature should work properly", Description: "It should have comment\n\n"}},
	},
	// KubeDescribe + Context + It
	{"e2e/foo.go", `
var _ = framework.KubeDescribe("Feature", func() {
	Context("when offline", func() {
		//Testname: Kubelet-OutputToLogs
		//Description: By default the stdout and stderr from the process
		//being executed in a pod MUST be sent to the pod's logs.
		framework.ConformanceIt("should work", func() {})
	})
})`, []conformanceData{{URL: "https://github.com/kubernetes/kubernetes/tree/master/e2e/foo.go#L8", TestName: "Kubelet-OutputToLogs",
		Description: `By default the stdout and stderr from the process
being executed in a pod MUST be sent to the pod's logs.` + "\n\n"}},
	},
	// KubeDescribe + Context + It
	{"e2e/foo.go", `
var _ = framework.KubeDescribe("Feature", func() {
	Context("with context", func() {
		//Description: By default the stdout and stderr from the process
		//being executed in a pod MUST be sent to the pod's logs.
		framework.ConformanceIt("should work", func() {})
	})
})`, []conformanceData{{URL: "https://github.com/kubernetes/kubernetes/tree/master/e2e/foo.go#L7", TestName: "Feature with context should work",
		Description: `By default the stdout and stderr from the process
being executed in a pod MUST be sent to the pod's logs.` + "\n\n"}},
	},
}

func TestConformance(t *testing.T) {
	for _, test := range conformanceCases {
		code := "package test\n" + test.code
		*confDoc = true
		tests := scanfile(test.filename, code)
		if !reflect.DeepEqual(tests, test.output) {
			t.Errorf("code:\n%s\ngot  %v\nwant %v",
				code, tests, test.output)
		}
	}
}

func TestNormalizeTestNames(t *testing.T) {
	testCases := []struct {
		rawName        string
		normalizedName string
	}{
		{
			"should have monotonically increasing restart count  [Slow]",
			"should have monotonically increasing restart count",
		},
		{
			" should check is all data is printed  ",
			"should check is all data is printed",
		},
	}
	for i, tc := range testCases {
		actualName := normalizeTestName(tc.rawName)
		if actualName != tc.normalizedName {
			t.Errorf("test case[%d]: expected normalized name %q, got %q", i, tc.normalizedName, actualName)
		}
	}
}
