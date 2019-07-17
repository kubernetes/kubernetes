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
	"errors"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/reporters"
	"github.com/onsi/gomega"

	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/ginkgowrapper"
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
	ginkgo.It("error", func() {
		err := errors.New("I'm an error, nice to meet to.")
		framework.ExpectNoError(err, "hard-coded error")
	})
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
	gomega.RegisterFailHandler(ginkgowrapper.Fail)
	fakeT := &testing.T{}
	reporter := reporters.NewFakeReporter()
	ginkgo.RunSpecsWithDefaultAndCustomReporters(fakeT, "Logging Suite", []ginkgo.Reporter{reporter})

	// Now check the output.
	g := gomega.NewGomegaWithT(t)
	g.Expect(normalizeReport(*reporter)).To(gomega.Equal(suiteResults{
		testResult{
			name: "[Top Level] log asserts",
			// TODO: also log the failed assertion as it happens
			output:  "INFO: before\nINFO: after\n",
			failure: "false is never true\nExpected\n    <bool>: false\nto equal\n    <bool>: true",
			// TODO: ginkowrapper.Fail should also prune this stack.
			stack: "\tassertion.go:75\nk8s.io/kubernetes/vendor/github.com/onsi/gomega/internal/assertion.(*Assertion).To()\n\tassertion.go:38\nk8s.io/kubernetes/test/e2e/framework/log_test.glob..func1.3()\n\tlogger_test.go:46\nk8s.io/kubernetes/vendor/github.com/onsi/ginkgo/internal/leafnodes.(*runner).runSync()\n\tlogger_test.go:66\n",
		},
		testResult{
			name: "[Top Level] log error",
			// TODO: the additional information about the error should be logged
			output:  "INFO: before\nINFO: Unexpected error occurred: I'm an error, nice to meet to.\nINFO: after\n",
			failure: "hard-coded error\nUnexpected error:\n    <*errors.errorString>: {\n        s: \"I'm an error, nice to meet to.\",\n    }\n    I'm an error, nice to meet to.\noccurred",
			// TODO: ginkowrapper.Fail should also prune this stack.
			stack: "\tutil.go:1362\nk8s.io/kubernetes/test/e2e/framework.ExpectNoError()\n\tutil.go:1356\nk8s.io/kubernetes/test/e2e/framework/log_test.glob..func1.4()\n\tlogger_test.go:49\nk8s.io/kubernetes/vendor/github.com/onsi/ginkgo/internal/leafnodes.(*runner).runSync()\n\tlogger_test.go:65\n",
		},
		testResult{
			name: "[Top Level] log fails",
			// TODO: why is the failure log as "INFO"?
			output:  "INFO: before\nINFO: I'm failing.\nINFO: after\n",
			failure: "I'm failing.",
			// TODO: ginkowrapper.Fail should also prune this stack.
			stack: "\tlogger.go:52\nk8s.io/kubernetes/test/e2e/framework/log.Failf()\n\tlogger.go:44\nk8s.io/kubernetes/test/e2e/framework/log_test.glob..func1.2.1(...)\n\tlogger_test.go:41\nk8s.io/kubernetes/test/e2e/framework/log_test.glob..func1.2()\n\tlogger_test.go:42\nk8s.io/kubernetes/vendor/github.com/onsi/ginkgo/internal/leafnodes.(*runner).runSync()\n\tlogger_test.go:65\n",
		},
	}))
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
var timePrefix = regexp.MustCompile(`(?m)^[[:alpha:]]{3} [[:digit:]]{1,2} [[:digit:]]{2}:[[:digit:]]{2}:[[:digit:]]{2}.[[:digit:]]{3}: `)

func stripTimes(in string) string {
	return timePrefix.ReplaceAllString(in, "")
}

// instanceAddr matches " | 0xc0003dec60>"
var instanceAddr = regexp.MustCompile(` \| 0x[0-9a-fA-F]+>`)

func stripAddresses(in string) string {
	return instanceAddr.ReplaceAllString(in, ">")
}

// stackLocation matches "\t/<some path>/<file>.go:75 +0x1f1".
var stackLocation = regexp.MustCompile(`/.*/([[:^space:]]+.go:[[:digit:]]+)( \+0x[0-9a-fA-F]+)?`)

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
