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

func deleteTargetStates() []State {
	var states []State
	// Baseline grid: every precondition shape, on success.
	for _, preconditions := range []PreconditionMode{PreconditionNone, PreconditionUID, PreconditionResourceVersion, PreconditionBoth} {
		states = append(states, State{Verb: VerbDelete, Preconditions: preconditions, Outcome: OutcomeSuccess})
	}
	// Outcome variety with no preconditions.
	for _, outcome := range []Outcome{OutcomeNotFound, OutcomeUnreachable, OutcomeTimeout, OutcomeOtherError} {
		states = append(states, State{Verb: VerbDelete, Preconditions: PreconditionNone, Outcome: outcome})
	}
	// A precondition mismatch, which storage reports as an invalid-object error.
	states = append(states, State{Verb: VerbDelete, Preconditions: PreconditionBoth, Outcome: OutcomeInvalidObj})
	return states
}
