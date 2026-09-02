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

package coverage

import "testing"

func TestScorecardReportsMissingStates(t *testing.T) {
	rec := NewRecorder()
	targetStates := []State{
		{Verb: VerbGet, ResourceVersion: RVUnset, Outcome: OutcomeSuccess},
		{Verb: VerbGet, ResourceVersion: RVZero, Outcome: OutcomeSuccess},
	}
	rec.Observe(targetStates[0])

	sc := rec.Scorecard(targetStates)
	if sc.Total != 2 || sc.Covered != 1 {
		t.Fatalf("expected 1/2 covered, got %d/%d", sc.Covered, sc.Total)
	}
	if len(sc.Missing) != 1 || sc.Missing[0] != targetStates[1] {
		t.Fatalf("expected %s to be reported missing, got %v", targetStates[1], sc.Missing)
	}
}
