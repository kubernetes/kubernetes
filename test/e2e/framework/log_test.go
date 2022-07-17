/*
Copyright 2019 The Kubernetes Authors.

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

package framework_test

import (
	"errors"
	"testing"

	"github.com/onsi/ginkgo/v2"

	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/internal/output"
)

// The line number of the following code is checked in TestFailureOutput below.
// Be careful when moving it around or changing the import statements above.
// Here are some intentionally blank lines that can be removed to compensate
// for future additional import statements.
//
//
//
//
//
//
// This must be line #39.

// This is included in a stack backtrace.
func failHelper(msg string) {
	framework.Fail(msg)
}

var _ = ginkgo.Describe("log", func() {
	ginkgo.BeforeEach(func() {
		framework.Logf("before")
	})
	ginkgo.AfterEach(func() {
		framework.Logf("after")
		framework.ExpectEqual(true, false, "true is never false either")
	})
	ginkgo.It("fails", func() {
		func() {
			framework.Failf("I'm failing.")
		}()
	})
	ginkgo.It("asserts", func() {
		framework.ExpectEqual(false, true, "false is never true")
	})
	ginkgo.It("error", func() {
		err := errors.New("an error with a long, useless description")
		framework.ExpectNoError(err, "hard-coded error")
	})
	ginkgo.It("equal", func() {
		framework.ExpectEqual(0, 1, "of course it's not equal...")
	})
	ginkgo.It("fails with helper", func() {
		failHelper("I'm failing with helper.")
	})
})

func TestFailureOutput(t *testing.T) {
	// output from AfterEach
	commonOutput := `
INFO: after
FAIL: true is never false either
Expected
    <bool>: true
to equal
    <bool>: false

Full Stack Trace
k8s.io/kubernetes/test/e2e/framework_test.glob..func1.2()
	log_test.go:52
`

	// Sorted by name!
	expected := output.SuiteResults{
		output.TestResult{
			Name: "log asserts",
			Output: `INFO: before
FAIL: false is never true
Expected
    <bool>: false
to equal
    <bool>: true

Full Stack Trace
k8s.io/kubernetes/test/e2e/framework_test.glob..func1.4()
	log_test.go:60` + commonOutput,
			Failure: `false is never true
Expected
    <bool>: false
to equal
    <bool>: true`,
			Stack: `k8s.io/kubernetes/test/e2e/framework_test.glob..func1.4()
	log_test.go:60`,
		},
		output.TestResult{
			Name: "log equal",
			Output: `INFO: before
FAIL: of course it's not equal...
Expected
    <int>: 0
to equal
    <int>: 1

Full Stack Trace
k8s.io/kubernetes/test/e2e/framework_test.glob..func1.6()
	log_test.go:67` + commonOutput,
			Failure: `of course it's not equal...
Expected
    <int>: 0
to equal
    <int>: 1`,
			Stack: `k8s.io/kubernetes/test/e2e/framework_test.glob..func1.6()
	log_test.go:67`,
		},
		output.TestResult{
			Name: "log error",
			Output: `INFO: before
INFO: Unexpected error: hard-coded error: 
    <*errors.errorString>: {
        s: "an error with a long, useless description",
    }
FAIL: hard-coded error: an error with a long, useless description

Full Stack Trace
k8s.io/kubernetes/test/e2e/framework_test.glob..func1.5()
	log_test.go:64` + commonOutput,
			Failure: `hard-coded error: an error with a long, useless description`,
			Stack: `k8s.io/kubernetes/test/e2e/framework_test.glob..func1.5()
	log_test.go:64`,
		},
		output.TestResult{
			Name: "log fails",
			Output: `INFO: before
FAIL: I'm failing.

Full Stack Trace
k8s.io/kubernetes/test/e2e/framework_test.glob..func1.3()
	log_test.go:57` + commonOutput,
			Failure: "I'm failing.",
			Stack: `k8s.io/kubernetes/test/e2e/framework_test.glob..func1.3()
	log_test.go:57`,
		},
		output.TestResult{
			Name: "log fails with helper",
			Output: `INFO: before
FAIL: I'm failing with helper.

Full Stack Trace
k8s.io/kubernetes/test/e2e/framework_test.failHelper()
	log_test.go:43
k8s.io/kubernetes/test/e2e/framework_test.glob..func1.7()
	log_test.go:70` + commonOutput,
			Failure: "I'm failing with helper.",
			Stack: `k8s.io/kubernetes/test/e2e/framework_test.failHelper()
	log_test.go:43
k8s.io/kubernetes/test/e2e/framework_test.glob..func1.7()
	log_test.go:70`,
		},
	}

	output.TestGinkgoOutput(t, expected)
}
