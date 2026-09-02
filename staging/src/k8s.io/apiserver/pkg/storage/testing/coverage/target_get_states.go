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

func getTargetStates() []State {
	var states []State
	// Baseline grid: every ResourceVersion mode, with and without IgnoreNotFound.
	for _, rv := range []ResourceVersionMode{RVUnset, RVZero, RVExact} {
		for _, ignoreNotFound := range []bool{false, true} {
			states = append(states, State{Verb: VerbGet, ResourceVersion: rv, IgnoreNotFound: ignoreNotFound, Outcome: OutcomeSuccess})
		}
	}
	// Outcome variety at the default shape (RVUnset, IgnoreNotFound=false).
	for _, outcome := range []Outcome{OutcomeNotFound, OutcomeUnreachable, OutcomeTimeout, OutcomeCorruptObj, OutcomeOtherError} {
		states = append(states, State{Verb: VerbGet, ResourceVersion: RVUnset, IgnoreNotFound: false, Outcome: outcome})
	}
	// A "not older than" get for a key that no longer exists at that revision.
	states = append(states, State{Verb: VerbGet, ResourceVersion: RVExact, IgnoreNotFound: false, Outcome: OutcomeNotFound})
	return states
}
