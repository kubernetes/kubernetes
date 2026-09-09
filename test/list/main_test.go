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
	"errors"
	"path/filepath"
	"reflect"
	"testing"
)

var collectCases = []struct {
	filename string
	code     string
	output   []Test
}{
	// Empty: no tests
	{"e2e/util_test.go", "", []Test{}},
	// Go unit test
	{"test/list/main_test.go", `
var num = 3
func Helper(x int) { return x / 0 }
func TestStuff(t *Testing.T) {
}`, []Test{{"test/list/main_test.go:5:1", "k8s.io/kubernetes/test/list", "TestStuff"}},
	},
	// Describe + It
	{"e2e/foo.go", `
var _ = Describe("Feature", func() {
	It("should work properly", func() {})
})`, []Test{{"e2e/foo.go:4:2", "[k8s.io] Feature", "should work properly"}},
	},
	// SIGDescribe + It
	{"e2e/foo.go", `
var _ = SIGDescribe("Feature", func() {
	It("should work properly", func() {})
})`, []Test{{"e2e/foo.go:4:2", "[k8s.io] Feature", "should work properly"}},
	},
	// SIGDescribe + Context + It
	{"e2e/foo.go", `
var _ = SIGDescribe("Feature", func() {
	Context("when offline", func() {
		It("should work", func() {})
	})
})`, []Test{{"e2e/foo.go:5:3", "[k8s.io] Feature when offline", "should work"}},
	},
	// SIGDescribe + It(Sprintf)
	{"e2e/foo.go", `
var _ = SIGDescribe("Feature", func() {
	It(fmt.Sprintf("handles %d nodes", num), func() {})
})`, []Test{{"e2e/foo.go:4:2", "[k8s.io] Feature", "handles * nodes"}},
	},
	// SIGDescribe + Sprintf + It(var)
	{"e2e/foo.go", `
var _ = SIGDescribe("Feature", func() {
	arg := fmt.Sprintf("does %s and %v at %d", task, desc, num)
	It(arg, func() {})
})`, []Test{{"e2e/foo.go:5:2", "[k8s.io] Feature", "does * and * at *"}},
	},
	// SIGDescribe + string + It(var)
	{"e2e/foo.go", `
var _ = SIGDescribe("Feature", func() {
	arg := "does stuff"
	It(arg, func() {})
})`, []Test{{"e2e/foo.go:5:2", "[k8s.io] Feature", "does stuff"}},
	},
	// SIGDescribe + It(unknown)
	{"e2e/foo.go", `
var _ = SIGDescribe("Feature", func() {
	It(mysteryFunc(), func() {})
})`, []Test{{"e2e/foo.go:4:2", "[k8s.io] Feature", "*"}},
	},
}

func TestCollect(t *testing.T) {
	for _, test := range collectCases {
		code := "package test\n" + test.code
		tests := collect(test.filename, code)
		if !reflect.DeepEqual(tests, test.output) {
			t.Errorf("code:\n%s\ngot  %v\nwant %v",
				code, tests, test.output)
		}
	}
}

func TestHandlePath(t *testing.T) {
	tl := testList{}
	e := errors.New("ex")
	if tl.handlePath("foo", nil, e) != e {
		t.Error("handlePath not returning errors")
	}
	if tl.handlePath("foo.txt", nil, nil) != nil {
		t.Error("should skip random files")
	}
	if tl.handlePath("third_party/a_test.go", nil, nil) != filepath.SkipDir {
		t.Error("should skip third_party")
	}
}
