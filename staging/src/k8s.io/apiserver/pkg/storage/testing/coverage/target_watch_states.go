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

func watchTargetStates() []State {
	var states []State
	// Baseline grid: recursive watch, every ResourceVersion x ResourceVersionMatch
	// combination, no selector, no bookmarks, SendInitialEvents unset.
	for _, rv := range []ResourceVersionMode{RVUnset, RVZero, RVExact} {
		for _, match := range []ResourceVersionMatchMode{RVMatchUnset, RVMatchNotOlderThan, RVMatchExact} {
			states = append(states, State{
				Verb: VerbWatch, ResourceVersion: rv, ResourceVersionMatch: match, Recursive: true,
				Selector: SelectorNone, SendInitialEvents: SendInitialEventsUnset, Outcome: OutcomeSuccess,
			})
		}
	}
	// Single-object watch (Recursive=false).
	states = append(states, State{
		Verb: VerbWatch, ResourceVersion: RVUnset, ResourceVersionMatch: RVMatchUnset,
		Recursive: false, Selector: SelectorNone, SendInitialEvents: SendInitialEventsUnset, Outcome: OutcomeSuccess,
	})
	// Selector variety at the default shape.
	for _, selector := range []SelectorMode{SelectorLabel, SelectorField, SelectorBoth} {
		states = append(states, State{
			Verb: VerbWatch, ResourceVersion: RVUnset, ResourceVersionMatch: RVMatchUnset,
			Recursive: true, Selector: selector, SendInitialEvents: SendInitialEventsUnset, Outcome: OutcomeSuccess,
		})
	}
	// AllowWatchBookmarks at the default shape, and combined with a selector
	// since bookmark delivery interacts with predicate-filtered watches.
	states = append(states,
		State{Verb: VerbWatch, ResourceVersion: RVUnset, Recursive: true, Selector: SelectorNone, AllowWatchBookmarks: true, SendInitialEvents: SendInitialEventsUnset, Outcome: OutcomeSuccess},
		State{Verb: VerbWatch, ResourceVersion: RVUnset, Recursive: true, Selector: SelectorBoth, AllowWatchBookmarks: true, SendInitialEvents: SendInitialEventsUnset, Outcome: OutcomeSuccess},
	)
	// SendInitialEvents variety, including combined with RVZero (a common
	// real request shape for streaming a full initial snapshot).
	states = append(states,
		State{Verb: VerbWatch, ResourceVersion: RVUnset, Recursive: true, Selector: SelectorNone, SendInitialEvents: SendInitialEventsTrue, Outcome: OutcomeSuccess},
		State{Verb: VerbWatch, ResourceVersion: RVUnset, Recursive: true, Selector: SelectorNone, SendInitialEvents: SendInitialEventsFalse, Outcome: OutcomeSuccess},
		State{Verb: VerbWatch, ResourceVersion: RVZero, Recursive: true, Selector: SelectorNone, SendInitialEvents: SendInitialEventsTrue, Outcome: OutcomeSuccess},
	)
	// Outcome variety at the default shape.
	for _, outcome := range []Outcome{OutcomeUnreachable, OutcomeTimeout, OutcomeCorruptObj, OutcomeOtherError} {
		states = append(states, State{
			Verb: VerbWatch, ResourceVersion: RVUnset, ResourceVersionMatch: RVMatchUnset,
			Recursive: true, Selector: SelectorNone, SendInitialEvents: SendInitialEventsUnset, Outcome: outcome,
		})
	}
	return states
}
