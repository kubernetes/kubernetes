/*
Copyright The Kubernetes Authors.

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

package e2enode

import (
	"fmt"
	"sort"
)

type memqosConformanceVerdict string

const (
	memqosConformanceFulfilled            memqosConformanceVerdict = "FULFILLED"
	memqosConformanceDrifted              memqosConformanceVerdict = "DRIFTED"
	memqosConformanceInsufficientEvidence memqosConformanceVerdict = "INSUFFICIENT_EVIDENCE"
)

type memqosControlMismatch struct {
	Control  string
	Expected string
	Observed string
}

type memqosConformanceResult struct {
	Verdict    memqosConformanceVerdict
	Mismatches []memqosControlMismatch
}

// memqosCheckControlState compares derived MemoryQoS control values with a
// read-back from the effective cgroup. Callers choose the controls relevant to
// the scenario; an absent observed control is insufficient evidence rather
// than a mismatch.
func memqosCheckControlState(expected, observed map[string]string) memqosConformanceResult {
	if len(expected) == 0 {
		return memqosConformanceResult{Verdict: memqosConformanceInsufficientEvidence}
	}

	controls := make([]string, 0, len(expected))
	for control := range expected {
		controls = append(controls, control)
	}
	sort.Strings(controls)

	mismatches := make([]memqosControlMismatch, 0)
	for _, control := range controls {
		observedValue, found := observed[control]
		if !found {
			return memqosConformanceResult{Verdict: memqosConformanceInsufficientEvidence}
		}
		if expected[control] != observedValue {
			mismatches = append(mismatches, memqosControlMismatch{
				Control:  control,
				Expected: expected[control],
				Observed: observedValue,
			})
		}
	}

	if len(mismatches) > 0 {
		return memqosConformanceResult{
			Verdict:    memqosConformanceDrifted,
			Mismatches: mismatches,
		}
	}
	return memqosConformanceResult{Verdict: memqosConformanceFulfilled}
}

func (r memqosConformanceResult) String() string {
	if len(r.Mismatches) == 0 {
		return string(r.Verdict)
	}
	return fmt.Sprintf("%s: %+v", r.Verdict, r.Mismatches)
}
