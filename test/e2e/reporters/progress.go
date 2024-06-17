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
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/ginkgo/v2/types"
	"k8s.io/klog/v2"
)

// ProgressReporter is a ginkgo reporter which tracks the total number of tests to be run/passed/failed/skipped.
// As new tests are completed it updates the values and prints them to stdout and optionally, sends the updates
// to the configured URL.

// One known limitation of the ProgressReporter it this reporter will not consolidate the reporter from each sub-process
// if the tests are run in parallel.
// As what's observed the reporter sent before test suite is started will be assembled correctly but each process will report
// on its own after each test or suite are completed.
// Here is a sample report that 5 testcases are executed totally, and run in parallel with 3 procs,
// 3 of them are failed and other 2 passed.
// {"msg":"","total":5,"completed":0,"skipped":0,"failed":0}
// {"msg":"Test Suite starting","total":5,"completed":0,"skipped":0,"failed":0}
// {"msg":"FAILED [sig-node] NoExecuteTaintManager Single Pod doesn't evict pod with tolerations from tainted nodes","total":0,"completed":0,"skipped":1332,"failed":1,"failures":["[sig-node] NoExecuteTaintManager..."]}
// {"msg":"FAILED [sig-node] NoExecuteTaintManager Single Pod evicts pods from tainted nodes","total":5,"completed":0,"skipped":2524,"failed":1,"failures":["[sig-node] NoExecuteTaintManager Single Pod evicts pods from tainted nodes"]}
// {"msg":"PASSED [sig-node] NoExecuteTaintManager Single Pod removing taint cancels eviction [Disruptive] [Conformance]","total":0,"completed":1,"skipped":1181,"failed":0}
// {"msg":"Test Suite completed","total":0,"completed":1,"skipped":2592,"failed":0}
// {"msg":"PASSED [sig-node] NoExecuteTaintManager Single Pod eventually evict pod with finite tolerations from tainted nodes","total":0,"completed":1,"skipped":1399,"failed":1,"failures":["[sig-node] NoExecuteTaintManager..."]}
// {"msg":"Test Suite completed","total":0,"completed":1,"skipped":1399,"failed":1,"failures":["[sig-node] NoExecuteTaintManager Single Pod doesn't evict pod with tolerations from tainted nodes"]}
// {"msg":"FAILED [sig-node] NoExecuteTaintManager Single Pod pods evicted from tainted nodes...","total":5,"completed":0,"skipped":3076,"failed":2,"failures":["[sig-node] NoExecuteTaintManager...","[sig-node] NoExecuteTaintManager..."]}
// {"msg":"Test Suite completed","total":5,"completed":0,"skipped":3076,"failed":2,"failures":["[sig-node] NoExecuteTaintManager Single Pod evicts pods from tainted nodes","[sig-node] NoExecuteTaintManager..."]}
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

// SendUpdates serializes the current progress and posts it to the configured endpoint if set.
// It does not print to stdout because that interferes with progress reporting by Ginko
// and (when Ginkgo does output redirection) doesn't actually appear on the screen anyway.
func (reporter *ProgressReporter) SendUpdates() {
	// If a progressURL and client is set/available then POST to it. Noop otherwise.
	if reporter.client == nil || len(reporter.progressURL) == 0 {
		return
	}
	b := reporter.serialize()
	go reporter.postProgressToURL(b)
}

func (reporter *ProgressReporter) postProgressToURL(b []byte) {
	resp, err := reporter.client.Post(reporter.progressURL, "application/json", bytes.NewReader(b))
	if err != nil {
		klog.Errorf("Failed to post progress update to %v: %v", reporter.progressURL, err)
		return
	}
	if resp.StatusCode >= 400 {
		klog.Errorf("Unexpected response when posting progress update to %v: %v", reporter.progressURL, resp.StatusCode)
		if resp.Body != nil {
			defer resp.Body.Close()
			respBody, err := io.ReadAll(resp.Body)
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

func (reporter *ProgressReporter) SetStartMsg() {
	reporter.LastMsg = "Test Suite starting"
	reporter.SendUpdates()
}

func (reporter *ProgressReporter) SetTestsTotal(totalSpec int) {
	reporter.TestsTotal = totalSpec
	reporter.SendUpdates()
}

// ProcessSpecReport summarizes the report state and sends the state to the configured endpoint if set.
func (reporter *ProgressReporter) ProcessSpecReport(report ginkgo.SpecReport) {
	testName := strings.Join(report.ContainerHierarchyTexts, " ")
	if len(report.LeafNodeText) > 0 {
		testName = testName + " " + report.LeafNodeText
	}
	switch report.State {
	case types.SpecStateFailed:
		if len(testName) > 0 {
			reporter.Failures = append(reporter.Failures, testName)
		} else {
			reporter.Failures = append(reporter.Failures, "Unknown test name")
		}
		reporter.TestsFailed++
		reporter.LastMsg = fmt.Sprintf("FAILED %v", testName)
	case types.SpecStatePassed:
		reporter.TestsCompleted++
		reporter.LastMsg = fmt.Sprintf("PASSED %v", testName)
	case types.SpecStateSkipped:
		reporter.TestsSkipped++
		return
	default:
		return
	}

	reporter.SendUpdates()
}

func (reporter *ProgressReporter) SetEndMsg() {
	reporter.LastMsg = "Test Suite completed"
	reporter.SendUpdates()
}
