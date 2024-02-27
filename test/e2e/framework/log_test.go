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
	"os"
	"path"
	"testing"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/ginkgo/v2/reporters"
	"github.com/onsi/gomega"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/internal/output"
)

// The line number of the following code is checked in TestFailureOutput below.
// Be careful when moving it around or changing the import statements above.
// Here are some intentionally blank lines that can be removed to compensate
// for future additional import statements.
// This must be line #40.

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
		gomega.Expect(true).To(gomega.BeFalse(), "true is never false either")
	})
	ginkgo.It("fails", func() {
		func() {
			framework.Failf("I'm failing.")
		}()
	})
	ginkgo.It("asserts", func() {
		gomega.Expect(false).To(gomega.BeTrue(), "false is never true")
	})
	ginkgo.It("error", func() {
		err := errors.New("an error with a long, useless description")
		framework.ExpectNoError(err, "hard-coded error")
	})
	ginkgo.It("equal", func() {
		gomega.Expect(0).To(gomega.Equal(1), "of course it's not equal...")
	})
	ginkgo.It("fails with helper", func() {
		failHelper("I'm failing with helper.")
	})
	ginkgo.It("redirects klog", func() {
		klog.Info("hello world")
		klog.Error(nil, "not really an error")
	})
})

func TestFailureOutput(t *testing.T) {
	expected := output.TestResult{
		Suite: reporters.JUnitTestSuite{
			Tests:    6,
			Failures: 6,
			Errors:   0,
			Disabled: 0,
			Skipped:  0,

			TestCases: []reporters.JUnitTestCase{
				{
					Name:   "[It] log fails",
					Status: "failed",
					Failure: &reporters.JUnitFailure{
						Type: "failed",
						Description: `[FAILED] I'm failing.
In [It] at: log_test.go:57 <time>

There were additional failures detected after the initial failure. These are visible in the timeline
`,
					},
					SystemErr: `> Enter [BeforeEach] log - log_test.go:48 <time>
<klog> log_test.go:49] before
< Exit [BeforeEach] log - log_test.go:48 <time>
> Enter [It] fails - log_test.go:55 <time>
[FAILED] I'm failing.
In [It] at: log_test.go:57 <time>
< Exit [It] fails - log_test.go:55 <time>
> Enter [AfterEach] log - log_test.go:51 <time>
<klog> log_test.go:52] after
[FAILED] true is never false either
Expected
    <bool>: true
to be false
In [AfterEach] at: log_test.go:53 <time>
< Exit [AfterEach] log - log_test.go:51 <time>
`,
				},
				{
					Name:   "[It] log asserts",
					Status: "failed",
					Failure: &reporters.JUnitFailure{
						Type: "failed",
						Description: `[FAILED] false is never true
Expected
    <bool>: false
to be true
In [It] at: log_test.go:61 <time>

There were additional failures detected after the initial failure. These are visible in the timeline
`,
					},
					SystemErr: `> Enter [BeforeEach] log - log_test.go:48 <time>
<klog> log_test.go:49] before
< Exit [BeforeEach] log - log_test.go:48 <time>
> Enter [It] asserts - log_test.go:60 <time>
[FAILED] false is never true
Expected
    <bool>: false
to be true
In [It] at: log_test.go:61 <time>
< Exit [It] asserts - log_test.go:60 <time>
> Enter [AfterEach] log - log_test.go:51 <time>
<klog> log_test.go:52] after
[FAILED] true is never false either
Expected
    <bool>: true
to be false
In [AfterEach] at: log_test.go:53 <time>
< Exit [AfterEach] log - log_test.go:51 <time>
`,
				},
				{
					Name:   "[It] log error",
					Status: "failed",
					Failure: &reporters.JUnitFailure{
						Type: "failed",
						Description: `[FAILED] hard-coded error: an error with a long, useless description
In [It] at: log_test.go:65 <time>

There were additional failures detected after the initial failure. These are visible in the timeline
`,
					},
					SystemErr: `> Enter [BeforeEach] log - log_test.go:48 <time>
<klog> log_test.go:49] before
< Exit [BeforeEach] log - log_test.go:48 <time>
> Enter [It] error - log_test.go:63 <time>
<klog> log_test.go:65] Unexpected error: hard-coded error: 
    <*errors.errorString>: 
    an error with a long, useless description
    {
        s: "an error with a long, useless description",
    }
[FAILED] hard-coded error: an error with a long, useless description
In [It] at: log_test.go:65 <time>
< Exit [It] error - log_test.go:63 <time>
> Enter [AfterEach] log - log_test.go:51 <time>
<klog> log_test.go:52] after
[FAILED] true is never false either
Expected
    <bool>: true
to be false
In [AfterEach] at: log_test.go:53 <time>
< Exit [AfterEach] log - log_test.go:51 <time>
`,
				},
				{
					Name:   "[It] log equal",
					Status: "failed",
					Failure: &reporters.JUnitFailure{
						Type: "failed",
						Description: `[FAILED] of course it's not equal...
Expected
    <int>: 0
to equal
    <int>: 1
In [It] at: log_test.go:68 <time>

There were additional failures detected after the initial failure. These are visible in the timeline
`,
					},
					SystemErr: `> Enter [BeforeEach] log - log_test.go:48 <time>
<klog> log_test.go:49] before
< Exit [BeforeEach] log - log_test.go:48 <time>
> Enter [It] equal - log_test.go:67 <time>
[FAILED] of course it's not equal...
Expected
    <int>: 0
to equal
    <int>: 1
In [It] at: log_test.go:68 <time>
< Exit [It] equal - log_test.go:67 <time>
> Enter [AfterEach] log - log_test.go:51 <time>
<klog> log_test.go:52] after
[FAILED] true is never false either
Expected
    <bool>: true
to be false
In [AfterEach] at: log_test.go:53 <time>
< Exit [AfterEach] log - log_test.go:51 <time>
`,
				},
				{
					Name:   "[It] log fails with helper",
					Status: "failed",
					Failure: &reporters.JUnitFailure{
						Type: "failed",
						Description: `[FAILED] I'm failing with helper.
In [It] at: log_test.go:44 <time>

There were additional failures detected after the initial failure. These are visible in the timeline
`,
					},
					SystemErr: `> Enter [BeforeEach] log - log_test.go:48 <time>
<klog> log_test.go:49] before
< Exit [BeforeEach] log - log_test.go:48 <time>
> Enter [It] fails with helper - log_test.go:70 <time>
[FAILED] I'm failing with helper.
In [It] at: log_test.go:44 <time>
< Exit [It] fails with helper - log_test.go:70 <time>
> Enter [AfterEach] log - log_test.go:51 <time>
<klog> log_test.go:52] after
[FAILED] true is never false either
Expected
    <bool>: true
to be false
In [AfterEach] at: log_test.go:53 <time>
< Exit [AfterEach] log - log_test.go:51 <time>
`,
				},
				{
					Name:   "[It] log redirects klog",
					Status: "failed",
					Failure: &reporters.JUnitFailure{
						Type: "failed",
						Description: `[FAILED] true is never false either
Expected
    <bool>: true
to be false
In [AfterEach] at: log_test.go:53 <time>
`,
					},
					SystemErr: `> Enter [BeforeEach] log - log_test.go:48 <time>
<klog> log_test.go:49] before
< Exit [BeforeEach] log - log_test.go:48 <time>
> Enter [It] redirects klog - log_test.go:73 <time>
<klog> log_test.go:74] hello world
<klog> log_test.go:75] <nil>not really an error
< Exit [It] redirects klog - log_test.go:73 <time>
> Enter [AfterEach] log - log_test.go:51 <time>
<klog> log_test.go:52] after
[FAILED] true is never false either
Expected
    <bool>: true
to be false
In [AfterEach] at: log_test.go:53 <time>
< Exit [AfterEach] log - log_test.go:51 <time>
`,
				},
			},
		},
	}

	// Simulate the test setup as in a normal e2e test which uses the
	// framework, but remember to restore klog settings when we are done.
	state := klog.CaptureState()
	defer state.Restore()
	var testContext framework.TestContextType
	framework.AfterReadingAllFlags(&testContext)

	oldStderr := os.Stderr
	tmp := t.TempDir()
	filename := path.Join(tmp, "stderr.log")
	f, err := os.Create(filename)
	require.NoError(t, err, "create temporary file")
	os.Stderr = f
	defer func() {
		os.Stderr = oldStderr

		err := f.Close()
		require.NoError(t, err, "close temporary file")
		actual, err := os.ReadFile(filename)
		require.NoError(t, err, "read temporary file")
		assert.Empty(t, string(actual), "no output on stderr")
	}()

	output.TestGinkgoOutput(t, expected)
}
