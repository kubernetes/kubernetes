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
// TODO: Number of test specs is not available now, we can add it back when this is fixed in the Ginkgo V2.
// pls see: https://github.com/kubernetes/kubernetes/issues/109744
type ProgressReporter struct {
	LastMsg string `json:"msg"`

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

// SendUpdates serializes the current progress and prints it to stdout and also posts it to the configured endpoint if set.
func (reporter *ProgressReporter) SendUpdates() {
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
