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

import "fmt"

// Verb identifies the storage.Interface method a State was classified from.
type Verb string

const (
	VerbGet    Verb = "Get"
	VerbList   Verb = "List"
	VerbWatch  Verb = "Watch"
	VerbDelete Verb = "Delete"
)

// Outcome classifies the terminal result of a storage.Interface call.
type Outcome string

const (
	OutcomeSuccess     Outcome = "Success"
	OutcomeNotFound    Outcome = "NotFound"
	OutcomeExists      Outcome = "Exists"
	OutcomeConflict    Outcome = "Conflict"
	OutcomeInvalidObj  Outcome = "InvalidObj"
	OutcomeUnreachable Outcome = "Unreachable"
	OutcomeTimeout     Outcome = "Timeout"
	OutcomeCorruptObj  Outcome = "CorruptObj"
	OutcomeOtherError  Outcome = "OtherError"
)

// State is a single classified API state: a storage.Interface verb plus the
// combination of request/response dimensions applicable to that verb. Only
// the fields relevant to State.Verb are populated; the rest are left at
// their zero value. State is comparable and safe to use as a map key.
type State struct {
	Verb Verb

	// Get, List, Watch
	ResourceVersion ResourceVersionMode

	// List, Watch
	ResourceVersionMatch ResourceVersionMatchMode
	Recursive            bool
	Selector             SelectorMode

	// Get only
	IgnoreNotFound bool

	// List only
	Pagination PaginationMode

	// Watch only
	AllowWatchBookmarks bool
	SendInitialEvents   SendInitialEventsMode

	// Delete only
	Preconditions PreconditionMode

	// all verbs
	Outcome Outcome
}

// String renders the State as a stable, human-readable label listing only
// the dimensions applicable to its Verb, for use in coverage reports.
func (s State) String() string {
	switch s.Verb {
	case VerbGet:
		return fmt.Sprintf("Get(rv=%s, ignoreNotFound=%t, outcome=%s)", s.ResourceVersion, s.IgnoreNotFound, s.Outcome)
	case VerbList:
		return fmt.Sprintf("List(rv=%s, rvMatch=%s, recursive=%t, pagination=%s, selector=%s, outcome=%s)",
			s.ResourceVersion, s.ResourceVersionMatch, s.Recursive, s.Pagination, s.Selector, s.Outcome)
	case VerbWatch:
		return fmt.Sprintf("Watch(rv=%s, rvMatch=%s, recursive=%t, selector=%s, allowBookmarks=%t, sendInitialEvents=%s, outcome=%s)",
			s.ResourceVersion, s.ResourceVersionMatch, s.Recursive, s.Selector, s.AllowWatchBookmarks, s.SendInitialEvents, s.Outcome)
	case VerbDelete:
		return fmt.Sprintf("Delete(preconditions=%s, outcome=%s)", s.Preconditions, s.Outcome)
	default:
		return fmt.Sprintf("%s(outcome=%s)", s.Verb, s.Outcome)
	}
}
