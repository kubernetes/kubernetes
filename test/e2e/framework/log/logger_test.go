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

package log_test

import (
	// "errors"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/reporters"
	"github.com/onsi/gomega"

	// "k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/log"
)

var _ = ginkgo.Describe("log", func() {
	ginkgo.BeforeEach(func() {
		log.Logf("before")
	})
	ginkgo.It("fails", func() {
		func() {
			log.Failf("I'm failing.")
		}()
	})
	ginkgo.It("asserts", func() {
		gomega.Expect(false).To(gomega.Equal(true), "false is never true")
	})
	// ginkgo.It("error", func() { // TODO(pohly): enable again, see comment below.
	// 	err := errors.New("an error with a long, useless description")
	// 	framework.ExpectNoError(err, "hard-coded error")
	// })
	ginkgo.AfterEach(func() {
		log.Logf("after")
		gomega.Expect(true).To(gomega.Equal(false), "true is never false either")
	})
})

func TestFailureOutput(t *testing.T) {
	// Run the Ginkgo suite with output collected by a custom
	// reporter in adddition to the default one. To see what the full
	// Ginkgo report looks like, run this test with "go test -v".
	config.DefaultReporterConfig.FullTrace = true
	gomega.RegisterFailHandler(log.Fail)
	fakeT := &testing.T{}
	reporter := reporters.NewFakeReporter()
	ginkgo.RunSpecsWithDefaultAndCustomReporters(fakeT, "Logging Suite", []ginkgo.Reporter{reporter})

	// Now check the output.
	// TODO: all of the stacks are currently broken because Ginkgo doesn't properly skip
	// over the initial entries returned by runtime.Stack. Fix is pending in
	// https://github.com/onsi/ginkgo/pull/590, "stack" texts need to be updated
	// when updating to a fixed Ginkgo.
	g := gomega.NewGomegaWithT(t)
	actual := normalizeReport(*reporter)
	expected := suiteResults{
		testResult{
			name:    "[Top Level] log asserts",
			output:  "INFO: before\nFAIL: false is never true\nExpected\n    <bool>: false\nto equal\n    <bool>: true\nINFO: after\nFAIL: true is never false either\nExpected\n    <bool>: true\nto equal\n    <bool>: false\n",
			failure: "false is never true\nExpected\n    <bool>: false\nto equal\n    <bool>: true",
			// TODO: should start with k8s.io/kubernetes/test/e2e/framework/log_test.glob..func1.3()
			stack: "\tassertion.go:75\nk8s.io/kubernetes/vendor/github.com/onsi/gomega/internal/assertion.(*Assertion).To()\n\tassertion.go:38\nk8s.io/kubernetes/test/e2e/framework/log_test.glob..func1.3()\n\tlogger_test.go:45\nk8s.io/kubernetes/vendor/github.com/onsi/ginkgo/internal/leafnodes.(*runner).runSync()\n\tlogger_test.go:65\n",
		},
		// That util.go appears in the output is a bug (https://github.com/kubernetes/kubernetes/issues/82013).
		// Because it currently appears, this test case is brittle and breaks when someome makes unrelated
		// changes in util.go which change the line number. Therefore it is commented out.
		// testResult{
		// 	name:    "[Top Level] log error",
		// 	output:  "INFO: before\nFAIL: hard-coded error\nUnexpected error:\n    <*errors.errorString>: {\n        s: \"an error with a long, useless description\",\n    }\n    an error with a long, useless description\noccurred\nINFO: after\nFAIL: true is never false either\nExpected\n    <bool>: true\nto equal\n    <bool>: false\n",
		// 	failure: "hard-coded error\nUnexpected error:\n    <*errors.errorString>: {\n        s: \"an error with a long, useless description\",\n    }\n    an error with a long, useless description\noccurred",
		// 	// TODO: should start with k8s.io/kubernetes/test/e2e/framework/log_test.glob..func1.4()
		// 	stack: "\tutil.go:1362\nk8s.io/kubernetes/test/e2e/framework.ExpectNoError()\n\tutil.go:1356\nk8s.io/kubernetes/test/e2e/framework/log_test.glob..func1.4()\n\tlogger_test.go:49\nk8s.io/kubernetes/vendor/github.com/onsi/ginkgo/internal/leafnodes.(*runner).runSync()\n\tlogger_test.go:65\n",
		// },
		testResult{
			name:    "[Top Level] log fails",
			output:  "INFO: before\nFAIL: I'm failing.\nINFO: after\nFAIL: true is never false either\nExpected\n    <bool>: true\nto equal\n    <bool>: false\n",
			failure: "I'm failing.",
			// TODO: should start with k8s.io/kubernetes/test/e2e/framework/log_test.glob..func1.2.1(...)
			stack: "\tlogger.go:52\nk8s.io/kubernetes/test/e2e/framework/log.Failf()\n\tlogger.go:44\nk8s.io/kubernetes/test/e2e/framework/log_test.glob..func1.2.1(...)\n\tlogger_test.go:41\nk8s.io/kubernetes/test/e2e/framework/log_test.glob..func1.2()\n\tlogger_test.go:42\nk8s.io/kubernetes/vendor/github.com/onsi/ginkgo/internal/leafnodes.(*runner).runSync()\n\tlogger_test.go:65\n",
		},
	}
	// Compare individual fields. Comparing the slices leads to unreadable error output when there is any mismatch.
	g.Expect(len(actual)).To(gomega.Equal(len(expected)), "%d entries in %v", len(expected), actual)
	for i, a := range actual {
		b := expected[i]
		g.Expect(a.name).To(gomega.Equal(b.name), "name in %d", i)
		g.Expect(a.output).To(gomega.Equal(b.output), "output in %d", i)
		g.Expect(a.failure).To(gomega.Equal(b.failure), "failure in %d", i)
		g.Expect(a.stack).To(gomega.Equal(b.stack), "stack in %d", i)
	}
}

type testResult struct {
	name string
	// output written to GinkgoWriter during test.
	output string
	// failure is SpecSummary.Failure.Message with varying parts stripped.
	failure string
	// stack is a normalized version (just file names, function parametes stripped) of
	// Ginkgo's FullStackTrace of a failure. Empty if no failure.
	stack string
}

type suiteResults []testResult

func normalizeReport(report reporters.FakeReporter) suiteResults {
	var results suiteResults
	for _, spec := range report.SpecSummaries {
		results = append(results, testResult{
			name:    strings.Join(spec.ComponentTexts, " "),
			output:  stripAddresses(stripTimes(spec.CapturedOutput)),
			failure: stripAddresses(stripTimes(spec.Failure.Message)),
			stack:   normalizeLocation(spec.Failure.Location.FullStackTrace),
		})
	}
	sort.Slice(results, func(i, j int) bool {
		return strings.Compare(results[i].name, results[j].name) < 0
	})
	return results
}

// timePrefix matches "Jul 17 08:08:25.950: " at the beginning of each line.
var timePrefix = regexp.MustCompile(`(?m)^[[:alpha:]]{3} +[[:digit:]]{1,2} +[[:digit:]]{2}:[[:digit:]]{2}:[[:digit:]]{2}.[[:digit:]]{3}: `)

func stripTimes(in string) string {
	return timePrefix.ReplaceAllString(in, "")
}

// instanceAddr matches " | 0xc0003dec60>"
var instanceAddr = regexp.MustCompile(` \| 0x[0-9a-fA-F]+>`)

func stripAddresses(in string) string {
	return instanceAddr.ReplaceAllString(in, ">")
}

// stackLocation matches "<some path>/<file>.go:75 +0x1f1" after a slash (built
// locally) or one of a few relative paths (built in the Kubernetes CI).
var stackLocation = regexp.MustCompile(`(?:/|vendor/|test/|GOROOT/).*/([[:^space:]]+.go:[[:digit:]]+)( \+0x[0-9a-fA-F]+)?`)

// functionArgs matches "<function name>(...)".
var functionArgs = regexp.MustCompile(`([[:alpha:]]+)\(.*\)`)

// testingStackEntries matches "testing.tRunner" and "created by" entries.
var testingStackEntries = regexp.MustCompile(`(?m)(?:testing\.|created by).*\n\t.*\n`)

// normalizeLocation removes path prefix and function parameters and certain stack entries
// that we don't care about.
func normalizeLocation(in string) string {
	out := in
	out = stackLocation.ReplaceAllString(out, "$1")
	out = functionArgs.ReplaceAllString(out, "$1()")
	out = testingStackEntries.ReplaceAllString(out, "")
	return out
}
