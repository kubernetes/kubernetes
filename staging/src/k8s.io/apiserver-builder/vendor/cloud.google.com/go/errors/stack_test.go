// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package errors

import "testing"

func TestChopStack(t *testing.T) {
	for _, test := range []struct {
		name     string
		in       []byte
		expected string
		isPanic  bool
	}{
		{
			name: "Catch",
			in: []byte(`goroutine 20 [running]:
runtime/debug.Stack()
	/gopath/src/runtime/debug/stack.go:24 +0x79
cloud.google.com/go/errors.(*Client).logInternal()
	/gopath/src/cloud.google.com/go/errors/errors.go:259 +0x18b
cloud.google.com/go/errors.(*Client).Catch()
	/gopath/src/cloud.google.com/go/errors/errors.go:219 +0x6ed
panic()
	/gopath/src/runtime/panic.go:458 +0x243
cloud.google.com/go/errors_test.TestCatchPanic()
	/gopath/src/cloud.google.com/go/errors/errors_test.go:93 +0x171
testing.tRunner()
	/gopath/src/testing/testing.go:610 +0x81
created by testing.(*T).Run
	/gopath/src/testing/testing.go:646 +0x2ec
`),
			expected: `goroutine 20 [running]:
cloud.google.com/go/errors_test.TestCatchPanic()
	/gopath/src/cloud.google.com/go/errors/errors_test.go:93 +0x171
testing.tRunner()
	/gopath/src/testing/testing.go:610 +0x81
created by testing.(*T).Run
	/gopath/src/testing/testing.go:646 +0x2ec
`,
			isPanic: true,
		},
		{
			name: "function not found",
			in: []byte(`goroutine 20 [running]:
runtime/debug.Stack()
	/gopath/src/runtime/debug/stack.go:24 +0x79
cloud.google.com/go/errors.(*Client).logInternal()
	/gopath/src/cloud.google.com/go/errors/errors.go:259 +0x18b
cloud.google.com/go/errors.(*Client).Catch()
	/gopath/src/cloud.google.com/go/errors/errors.go:219 +0x6ed
cloud.google.com/go/errors_test.TestCatchPanic()
	/gopath/src/cloud.google.com/go/errors/errors_test.go:93 +0x171
testing.tRunner()
	/gopath/src/testing/testing.go:610 +0x81
created by testing.(*T).Run
	/gopath/src/testing/testing.go:646 +0x2ec
`),
			expected: `goroutine 20 [running]:
runtime/debug.Stack()
	/gopath/src/runtime/debug/stack.go:24 +0x79
cloud.google.com/go/errors.(*Client).logInternal()
	/gopath/src/cloud.google.com/go/errors/errors.go:259 +0x18b
cloud.google.com/go/errors.(*Client).Catch()
	/gopath/src/cloud.google.com/go/errors/errors.go:219 +0x6ed
cloud.google.com/go/errors_test.TestCatchPanic()
	/gopath/src/cloud.google.com/go/errors/errors_test.go:93 +0x171
testing.tRunner()
	/gopath/src/testing/testing.go:610 +0x81
created by testing.(*T).Run
	/gopath/src/testing/testing.go:646 +0x2ec
`,
			isPanic: true,
		},
		{
			name: "Report",
			in: []byte(` goroutine 39 [running]:
runtime/debug.Stack()
	/gopath/runtime/debug/stack.go:24 +0x79
cloud.google.com/go/errors.(*Client).logInternal()
	/gopath/cloud.google.com/go/errors/errors.go:259 +0x18b
cloud.google.com/go/errors.(*Client).Report()
	/gopath/cloud.google.com/go/errors/errors.go:248 +0x4ed
cloud.google.com/go/errors_test.TestReport()
	/gopath/cloud.google.com/go/errors/errors_test.go:137 +0x2a1
testing.tRunner()
	/gopath/testing/testing.go:610 +0x81
created by testing.(*T).Run
	/gopath/testing/testing.go:646 +0x2ec
`),
			expected: ` goroutine 39 [running]:
cloud.google.com/go/errors_test.TestReport()
	/gopath/cloud.google.com/go/errors/errors_test.go:137 +0x2a1
testing.tRunner()
	/gopath/testing/testing.go:610 +0x81
created by testing.(*T).Run
	/gopath/testing/testing.go:646 +0x2ec
`,
			isPanic: false,
		},
	} {
		out := chopStack(test.in, test.isPanic)
		if out != test.expected {
			t.Errorf("case %q: chopStack(%q, %t): got %q want %q", test.name, test.in, test.isPanic, out, test.expected)
		}
	}
}
