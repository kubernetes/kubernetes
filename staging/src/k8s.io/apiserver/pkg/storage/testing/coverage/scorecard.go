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

import (
	"fmt"
	"sort"
	"strings"
)

// Scorecard reports, for a set of target States, which were observed at
// least once and which were not.
type Scorecard struct {
	Total   int
	Covered int
	Missing []State
}

// Scorecard diffs the Recorder's observations against targetStates, typically
// TargetStates(). Entries not observed at least once are reported in
// Missing, in the same order they appear in targetStates.
func (r *Recorder) Scorecard(targetStates []State) Scorecard {
	counts := r.Counts()
	sc := Scorecard{Total: len(targetStates)}
	for _, state := range targetStates {
		if counts[state] > 0 {
			sc.Covered++
		} else {
			sc.Missing = append(sc.Missing, state)
		}
	}
	return sc
}

// String renders a human-readable coverage report, e.g. for t.Log.
func (sc Scorecard) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "coverage: %d/%d states covered\n", sc.Covered, sc.Total)
	if len(sc.Missing) == 0 {
		return strings.TrimSuffix(b.String(), "\n")
	}
	missing := make([]string, len(sc.Missing))
	for i, s := range sc.Missing {
		missing[i] = s.String()
	}
	sort.Strings(missing)
	fmt.Fprintf(&b, "missing %d states:\n", len(missing))
	for _, s := range missing {
		fmt.Fprintf(&b, "  - %s\n", s)
	}
	return strings.TrimSuffix(b.String(), "\n")
}
