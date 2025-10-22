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

package junit

import (
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/ginkgo/v2/reporters"
	"github.com/onsi/ginkgo/v2/types"
)

// WriteJUnitReport generates a JUnit file that is shorter than the one
// normally written by `ginkgo --junit-report`. This is needed because the full
// report can become too large for tools like Spyglass
// (https://github.com/kubernetes/kubernetes/issues/111510).
func WriteJUnitReport(report ginkgo.Report, filename string) error {
	config := reporters.JunitReportConfig{
		// Remove details for specs where we don't care.
		OmitTimelinesForSpecState: types.SpecStatePassed | types.SpecStateSkipped,

		// Don't write <failure message="summary">. The same text is
		// also in the full text for the failure. If we were to write
		// both, then tools like kettle and spyglass would concatenate
		// the two strings and thus show duplicated information.
		OmitFailureMessageAttr: true,

		// All labels are also part of the spec texts in inline [] tags,
		// so we don't need to write them separately.
		OmitSpecLabels: true,
	}

	// Sort specs by full name. The default is by start (or completion?) time,
	// which is less useful in spyglass because those times are not shown
	// and thus tests seem to be listed with no apparent order.
	slices.SortFunc(report.SpecReports, func(a, b types.SpecReport) int {
		res := strings.Compare(a.FullText(), b.FullText())
		if res == 0 {
			// Use start time as tie-breaker in the unlikely
			// case that two specs have the same full name.
			return a.StartTime.Compare(b.StartTime)
		}
		return res
	})

	// Detect data races in output and mark those tests as failed.
	// Ginkgo itself captures the warning in the output of those
	// tests where it is printed, but does not mark the tests as failed.
	// This is not exactly wrong (the reason might be in code that was
	// started elsewhere), but without a test being marked as "failed",
	// prune-junit-xml will prune the output and spyglass won't show
	// the test as failed.
	//
	// Both variants of captured output get checked, just to be on the safe side.
	for i, spec := range report.SpecReports {
		if spec.State == types.SpecStatePassed {
			ginkgoWriterRace := containsDataRace(spec.CapturedGinkgoWriterOutput)
			stdoutRace := containsDataRace(spec.CapturedStdOutErr)
			if ginkgoWriterRace || stdoutRace {
				spec.State = types.SpecStateFailed
				spec.Failure = types.Failure{
					FailureNodeContext: types.FailureNodeIsLeafNode,
					FailureNodeType:    spec.LeafNodeType,
					Location:           types.NewCustomCodeLocation("output analysis"),
					TimelineLocation: types.TimelineLocation{
						Time: time.Now(),
					},
				}
				// Let's move that text to the failure message.
				if stdoutRace {
					spec.Failure.Message = spec.CapturedStdOutErr
					spec.CapturedStdOutErr = ""
				} else {
					spec.Failure.Message = spec.CapturedGinkgoWriterOutput
					spec.CapturedGinkgoWriterOutput = ""
				}
				report.SpecReports[i] = spec
			}
		}
	}

	return reporters.GenerateJUnitReportWithConfig(report, filename, config)
}

var dataRaceRE = regexp.MustCompile(`(?m)^WARNING: DATA RACE$`)

func containsDataRace(output string) bool {
	return dataRaceRE.MatchString(output)
}
