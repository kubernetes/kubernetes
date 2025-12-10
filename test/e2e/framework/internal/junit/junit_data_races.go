//go:build race

/*
Copyright 2025 The Kubernetes Authors.

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
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/ginkgo/v2/types"
)

var dataRaceRE = regexp.MustCompile(`(?m)^WARNING: DATA RACE$`)

func containsDataRace(output string) bool {
	return dataRaceRE.MatchString(output)
}

// Detect data races in output and mark those tests as failed.
// Ginkgo itself captures the warning in the output of those
// tests where it is printed, but does not mark the tests as failed.
// This is not exactly wrong (the reason might be in code that was
// started elsewhere), but without a test being marked as "failed",
// prune-junit-xml will prune the output and spyglass won't show
// the test as failed.
//
// Modifies the report. Doesn't touch tests which are already failed
// because those already need to be investigated.
func detectDataRaces(report ginkgo.Report) {
	for i, spec := range report.SpecReports {
		if spec.State != types.SpecStatePassed {
			continue
		}

		// Both variants of captured output get checked, just to be on the safe side.
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
