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

func listTargetStates() []State {
	var states []State
	// Baseline grid: recursive list, every ResourceVersion x ResourceVersionMatch
	// combination, no pagination, no selector.
	for _, rv := range []ResourceVersionMode{RVUnset, RVZero, RVExact} {
		for _, match := range []ResourceVersionMatchMode{RVMatchUnset, RVMatchNotOlderThan, RVMatchExact} {
			states = append(states, State{
				Verb: VerbList, ResourceVersion: rv, ResourceVersionMatch: match,
				Recursive: true, Pagination: PaginationNone, Selector: SelectorNone, Outcome: OutcomeSuccess,
			})
		}
	}
	// Single-object get-via-list (Recursive=false).
	states = append(states, State{
		Verb: VerbList, ResourceVersion: RVUnset, ResourceVersionMatch: RVMatchUnset,
		Recursive: false, Pagination: PaginationNone, Selector: SelectorNone, Outcome: OutcomeSuccess,
	})
	// Pagination variety at the default shape.
	for _, pagination := range []PaginationMode{PaginationLimit, PaginationContinue} {
		states = append(states, State{
			Verb: VerbList, ResourceVersion: RVZero, ResourceVersionMatch: RVMatchUnset,
			Recursive: true, Pagination: pagination, Selector: SelectorNone, Outcome: OutcomeSuccess,
		})
	}
	// Selector variety at the default shape.
	for _, selector := range []SelectorMode{SelectorLabel, SelectorField, SelectorBoth} {
		states = append(states, State{
			Verb: VerbList, ResourceVersion: RVUnset, ResourceVersionMatch: RVMatchUnset,
			Recursive: true, Pagination: PaginationNone, Selector: selector, Outcome: OutcomeSuccess,
		})
	}
	// Pagination combined with a selector, since predicate filtering interacts
	// with continue-token semantics (matched items may span multiple pages).
	states = append(states,
		State{Verb: VerbList, ResourceVersion: RVZero, Recursive: true, Pagination: PaginationLimit, Selector: SelectorLabel, Outcome: OutcomeSuccess},
		State{Verb: VerbList, ResourceVersion: RVUnset, Recursive: true, Pagination: PaginationContinue, Selector: SelectorBoth, Outcome: OutcomeSuccess},
	)
	// An exact resource version combined with a continue token: continue
	// tokens normally carry their own revision, so this combination is
	// expected to be rejected - worth tracking that it is.
	states = append(states, State{
		Verb: VerbList, ResourceVersion: RVExact, ResourceVersionMatch: RVMatchUnset,
		Recursive: true, Pagination: PaginationContinue, Selector: SelectorNone, Outcome: OutcomeOtherError,
	})
	// Outcome variety at the default shape.
	for _, outcome := range []Outcome{OutcomeUnreachable, OutcomeTimeout, OutcomeCorruptObj, OutcomeOtherError} {
		states = append(states, State{
			Verb: VerbList, ResourceVersion: RVUnset, ResourceVersionMatch: RVMatchUnset,
			Recursive: true, Pagination: PaginationNone, Selector: SelectorNone, Outcome: outcome,
		})
	}
	return states
}
