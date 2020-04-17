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

package reporters

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"time"

	"k8s.io/klog"

	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/types"
)

// ProgressReporter is a ginkgo reporter which tracks the total number of tests to be run/passed/failed/skipped.
// As new tests are completed it updates the values and prints them to stdout and optionally, sends the updates
// to the configured URL.
type ProgressReporter struct {
	LastMsg string `json:"msg"`

	TestsTotal     int `json:"total"`
	TestsCompleted int `json:"completed"`
	TestsSkipped   int `json:"skipped"`
	TestsFailed    int `json:"failed"`

	Failures []string `json:"failures,omitempty"`

	progressURL string
	client      *http.Client
}

// NewProgressReporter returns a progress reporter which posts updates to the given URL.
func NewProgressReporter(progressReportURL string) *ProgressReporter {
	rep := &ProgressReporter{
		Failures:    []string{},
		progressURL: progressReportURL,
	}
	if len(progressReportURL) > 0 {
		rep.client = &http.Client{
			Timeout: time.Second * 10,
		}
	}
	return rep
}

// SpecSuiteWillBegin is invoked by ginkgo when the suite is about to start and is the first point in which we can
// antipate the number of tests which will be run.
func (reporter *ProgressReporter) SpecSuiteWillBegin(cfg config.GinkgoConfigType, summary *types.SuiteSummary) {
	reporter.TestsTotal = summary.NumberOfSpecsThatWillBeRun
	reporter.LastMsg = "Test Suite starting"
	reporter.sendUpdates()
}

// SpecSuiteDidEnd is the last method invoked by Ginkgo after all the specs are run.
func (reporter *ProgressReporter) SpecSuiteDidEnd(summary *types.SuiteSummary) {
	reporter.LastMsg = "Test Suite completed"
	reporter.sendUpdates()
}

// SpecDidComplete is invoked by Ginkgo each time a spec is completed (including skipped specs).
func (reporter *ProgressReporter) SpecDidComplete(specSummary *types.SpecSummary) {
	testname := strings.Join(specSummary.ComponentTexts[1:], " ")
	switch specSummary.State {
	case types.SpecStateFailed:
		if len(specSummary.ComponentTexts) > 0 {
			reporter.Failures = append(reporter.Failures, testname)
		} else {
			reporter.Failures = append(reporter.Failures, "Unknown test name")
		}
		reporter.TestsFailed++
		reporter.LastMsg = fmt.Sprintf("FAILED %v", testname)
	case types.SpecStatePassed:
		reporter.TestsCompleted++
		reporter.LastMsg = fmt.Sprintf("PASSED %v", testname)
	case types.SpecStateSkipped:
		reporter.TestsSkipped++
		return
	default:
		return
	}

	reporter.sendUpdates()
}

// sendUpdates serializes the current progress and prints it to stdout and also posts it to the configured endpoint if set.
func (reporter *ProgressReporter) sendUpdates() {
	b := reporter.serialize()
	fmt.Println(string(b))
	go reporter.postProgressToURL(b)
}

func (reporter *ProgressReporter) postProgressToURL(b []byte) {
	// If a progressURL and client is set/available then POST to it. Noop otherwise.
	if reporter.client == nil || len(reporter.progressURL) == 0 {
		return
	}

	resp, err := reporter.client.Post(reporter.progressURL, "application/json", bytes.NewReader(b))
	if err != nil {
		klog.Errorf("Failed to post progress update to %v: %v", reporter.progressURL, err)
		return
	}
	if resp.StatusCode >= 400 {
		klog.Errorf("Unexpected response when posting progress update to %v: %v", reporter.progressURL, resp.StatusCode)
		if resp.Body != nil {
			defer resp.Body.Close()
			respBody, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				klog.Errorf("Failed to read response body from posting progress: %v", err)
				return
			}
			klog.Errorf("Response body from posting progress update: %v", respBody)
		}

		return
	}
}

func (reporter *ProgressReporter) serialize() []byte {
	b, err := json.Marshal(reporter)
	if err != nil {
		return []byte(fmt.Sprintf(`{"msg":"%v", "error":"%v"}`, reporter.LastMsg, err))
	}
	return b
}

// SpecWillRun is implemented as a noop to satisfy the reporter interface for ginkgo.
func (reporter *ProgressReporter) SpecWillRun(specSummary *types.SpecSummary) {}

// BeforeSuiteDidRun is implemented as a noop to satisfy the reporter interface for ginkgo.
func (reporter *ProgressReporter) BeforeSuiteDidRun(setupSummary *types.SetupSummary) {}

// AfterSuiteDidRun is implemented as a noop to satisfy the reporter interface for ginkgo.
func (reporter *ProgressReporter) AfterSuiteDidRun(setupSummary *types.SetupSummary) {}
