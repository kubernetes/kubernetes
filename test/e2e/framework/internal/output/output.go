/*
Copyright 2022 The Kubernetes Authors.

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

package output

import (
	"encoding/xml"
	"os"
	"path"
	"regexp"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/ginkgo/v2/reporters"
	"github.com/onsi/gomega"
	"github.com/stretchr/testify/require"

	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/internal/junit"
)

// TestGinkgoOutput runs the current suite and verifies that the generated
// JUnit file matches the expected result.
//
// The Ginkgo output on the console (aka the test suite log) does not get
// checked. It is usually less important for the CI and more relevant when
// using test suite interactively. To see what that Ginkgo output looks like,
// run tests with "go test -v".
func TestGinkgoOutput(t *testing.T, expected TestResult, runSpecsArgs ...interface{}) {
	tmpdir := t.TempDir()
	junitFile := path.Join(tmpdir, "junit.xml")
	gomega.RegisterFailHandler(framework.Fail)
	ginkgo.ReportAfterSuite("write JUnit file", func(report ginkgo.Report) {
		junit.WriteJUnitReport(report, junitFile)
	})
	fakeT := &testing.T{}
	ginkgo.RunSpecs(fakeT, "Logging Suite", runSpecsArgs...)

	var actual reporters.JUnitTestSuites
	data, err := os.ReadFile(junitFile)
	require.NoError(t, err)
	err = xml.Unmarshal(data, &actual)
	require.NoError(t, err)

	if len(actual.TestSuites) != 1 {
		t.Fatalf("expected one test suite, got %d, JUnit content:\n%s", len(actual.TestSuites), string(data))
	}
	diff := cmp.Diff(expected.Suite, actual.TestSuites[0],
		// Time varies.
		// Name and Classname are "Logging Suite".
		// Package includes a varying path, not interesting.
		// Properties also too complicated to compare.
		cmpopts.IgnoreFields(reporters.JUnitTestSuite{}, "Time", "Timestamp", "Name", "Package", "Properties"),
		cmpopts.IgnoreFields(reporters.JUnitTestCase{}, "Time", "Classname"),
		cmpopts.SortSlices(func(tc1, tc2 reporters.JUnitTestCase) bool {
			return tc1.Name < tc2.Name
		}),
		cmpopts.AcyclicTransformer("simplify", func(in string) any {
			out := simplify(in, expected)
			// Sometimes cmp.Diff does not print the full string when it is long.
			// Uncommenting this here may help debug differences.
			// if len(out) > 100 {
			// 	t.Logf("%s\n---------------------------------------\n%s\n", in, out)
			// }

			// Same idea as in
			// https://github.com/google/go-cmp/issues/192#issuecomment-605346277:
			// it forces cmp.Diff to diff strings line-by-line,
			// even when it normally wouldn't.  The downside is
			// that the output is harder to turn back into the
			// expected reference string.
			// if len(out) > 50 {
			// 	return strings.Split(out, "\n")
			// }

			return out
		}),
	)
	if diff != "" {
		t.Fatalf("Simplified JUnit report not as expected (-want, +got):\n%s\n\nFull XML:\n%s", diff, string(data))
	}
}

// TestResult is the expected outcome of the suite, with additional parameters that
// determine equality.
type TestResult struct {
	// Called to normalize all output strings before comparison if non-nil.
	NormalizeOutput func(string) string

	// All test cases and overall suite results.
	Suite reporters.JUnitTestSuite
}

func simplify(in string, expected TestResult) string {
	out := normalizeLocation(in)
	out = stripTimes(out)
	out = stripAddresses(out)
	out = normalizeInitFunctions(out)
	if expected.NormalizeOutput != nil {
		out = expected.NormalizeOutput(out)
	}
	return out
}

// timePrefix matches "Jul 17 08:08:25.950: " at the beginning of each line.
var timePrefix = regexp.MustCompile(`(?m)^[[:alpha:]]{3} +[[:digit:]]{1,2} +[[:digit:]]{2}:[[:digit:]]{2}:[[:digit:]]{2}.[[:digit:]]{3}: `)

// elapsedSuffix matches "Elapsed: 16.189µs"
var elapsedSuffix = regexp.MustCompile(`Elapsed: [[:digit:]]+(\.[[:digit:]]+)?(µs|ns|ms|s|m)`)

// afterSuffix matches "after 5.001s."
var afterSuffix = regexp.MustCompile(`after [[:digit:]]+(\.[[:digit:]]+)?(µs|ns|ms|s|m).`)

// timeSuffix matches "@ 09/06/22 15:36:43.44 (5.001s)" as printed by Ginkgo v2 for log output, with the duration being optional.
var timeSuffix = regexp.MustCompile(`(?m)@[[:space:]][[:digit:]]{2}/[[:digit:]]{2}/[[:digit:]]{2} [[:digit:]]{2}:[[:digit:]]{2}:[[:digit:]]{2}(\.[[:digit:]]{1,3})?( \([[:digit:]]+(\.[[:digit:]]+)?(µs|ns|ms|s|m)\))?$`)

func stripTimes(in string) string {
	out := timePrefix.ReplaceAllString(in, "")
	out = elapsedSuffix.ReplaceAllString(out, "Elapsed: <elapsed>")
	out = timeSuffix.ReplaceAllString(out, "<time>")
	out = afterSuffix.ReplaceAllString(out, "after <after>.")
	return out
}

// instanceAddr matches " | 0xc0003dec60>"
var instanceAddr = regexp.MustCompile(` \| 0x[0-9a-fA-F]+>`)

func stripAddresses(in string) string {
	return instanceAddr.ReplaceAllString(in, ">")
}

// stackLocation matches "<some path>/<file>.go:75 +0x1f1" after a slash (built
// locally) or one of a few relative paths (built in the Kubernetes CI).
var stackLocation = regexp.MustCompile(`(?:/|vendor/|test/|GOROOT/).*/([[:^space:]]+.go:[[:digit:]]+)( \+0x[0-9a-fA-F]+)?`)

// functionArgs matches "<function name>(...)" where <function name> may be an anonymous function (e.g. "pod_test.glob..func1.1")
var functionArgs = regexp.MustCompile(`([[:alpha:][:digit:].]+)\(.*\)`)

// klogPrefix matches "I0822 16:10:39.343790  989127 "
var klogPrefix = regexp.MustCompile(`(?m)^[IEF][[:digit:]]{4} [[:digit:]]{2}:[[:digit:]]{2}:[[:digit:]]{2}\.[[:digit:]]{6}[[:space:]]+[[:digit:]]+ `)

// testFailureOutput matches TestFailureOutput() and its source followed by additional stack entries:
//
// k8s.io/kubernetes/test/e2e/framework/pod/pod_test.TestFailureOutput(0xc000558800)
//
//	/nvme/gopath/src/k8s.io/kubernetes/test/e2e/framework/pod/wait_test.go:73 +0x1c9
//
// testing.tRunner(0xc000558800, 0x1af2848)
//
//	/nvme/gopath/go/src/testing/testing.go:865 +0xc0
//
// created by testing.(*T).Run
//
//	/nvme/gopath/go/src/testing/testing.go:916 +0x35a
var testFailureOutput = regexp.MustCompile(`(?m)^k8s.io/kubernetes/test/e2e/framework/internal/output\.TestGinkgoOutput\(.*\n\t.*(\n.*\n\t.*)*`)

// normalizeLocation removes path prefix and function parameters and certain stack entries
// that we don't care about.
func normalizeLocation(in string) string {
	out := in
	out = stackLocation.ReplaceAllString(out, "$1")
	out = functionArgs.ReplaceAllString(out, "$1()")
	out = testFailureOutput.ReplaceAllString(out, "")
	out = klogPrefix.ReplaceAllString(out, "<klog> ")
	return out
}

var initFunc = regexp.MustCompile(`(init\.+func|glob\.+func)`)

// normalizeInitFunctions maps both init.func (used by Go >= 1.22) and
// glob..func (used by Go < 1.22) to <init.func>.
func normalizeInitFunctions(in string) string {
	out := initFunc.ReplaceAllString(in, "<init.func>")
	return out
}
