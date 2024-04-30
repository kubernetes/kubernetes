// Copyright 2011 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package gomock

import (
	"bytes"
	"errors"
	"fmt"
	"sync"
)

// callSet represents a set of expected calls, indexed by receiver and method
// name.
type callSet struct {
	// Calls that are still expected.
	expected   map[callSetKey][]*Call
	expectedMu *sync.Mutex
	// Calls that have been exhausted.
	exhausted map[callSetKey][]*Call
	// when set to true, existing call expectations are overridden when new call expectations are made
	allowOverride bool
}

// callSetKey is the key in the maps in callSet
type callSetKey struct {
	receiver any
	fname    string
}

func newCallSet() *callSet {
	return &callSet{
		expected:   make(map[callSetKey][]*Call),
		expectedMu: &sync.Mutex{},
		exhausted:  make(map[callSetKey][]*Call),
	}
}

func newOverridableCallSet() *callSet {
	return &callSet{
		expected:      make(map[callSetKey][]*Call),
		expectedMu:    &sync.Mutex{},
		exhausted:     make(map[callSetKey][]*Call),
		allowOverride: true,
	}
}

// Add adds a new expected call.
func (cs callSet) Add(call *Call) {
	key := callSetKey{call.receiver, call.method}

	cs.expectedMu.Lock()
	defer cs.expectedMu.Unlock()

	m := cs.expected
	if call.exhausted() {
		m = cs.exhausted
	}
	if cs.allowOverride {
		m[key] = make([]*Call, 0)
	}

	m[key] = append(m[key], call)
}

// Remove removes an expected call.
func (cs callSet) Remove(call *Call) {
	key := callSetKey{call.receiver, call.method}

	cs.expectedMu.Lock()
	defer cs.expectedMu.Unlock()

	calls := cs.expected[key]
	for i, c := range calls {
		if c == call {
			// maintain order for remaining calls
			cs.expected[key] = append(calls[:i], calls[i+1:]...)
			cs.exhausted[key] = append(cs.exhausted[key], call)
			break
		}
	}
}

// FindMatch searches for a matching call. Returns error with explanation message if no call matched.
func (cs callSet) FindMatch(receiver any, method string, args []any) (*Call, error) {
	key := callSetKey{receiver, method}

	cs.expectedMu.Lock()
	defer cs.expectedMu.Unlock()

	// Search through the expected calls.
	expected := cs.expected[key]
	var callsErrors bytes.Buffer
	for _, call := range expected {
		err := call.matches(args)
		if err != nil {
			_, _ = fmt.Fprintf(&callsErrors, "\n%v", err)
		} else {
			return call, nil
		}
	}

	// If we haven't found a match then search through the exhausted calls so we
	// get useful error messages.
	exhausted := cs.exhausted[key]
	for _, call := range exhausted {
		if err := call.matches(args); err != nil {
			_, _ = fmt.Fprintf(&callsErrors, "\n%v", err)
			continue
		}
		_, _ = fmt.Fprintf(
			&callsErrors, "all expected calls for method %q have been exhausted", method,
		)
	}

	if len(expected)+len(exhausted) == 0 {
		_, _ = fmt.Fprintf(&callsErrors, "there are no expected calls of the method %q for that receiver", method)
	}

	return nil, errors.New(callsErrors.String())
}

// Failures returns the calls that are not satisfied.
func (cs callSet) Failures() []*Call {
	cs.expectedMu.Lock()
	defer cs.expectedMu.Unlock()

	failures := make([]*Call, 0, len(cs.expected))
	for _, calls := range cs.expected {
		for _, call := range calls {
			if !call.satisfied() {
				failures = append(failures, call)
			}
		}
	}
	return failures
}

// Satisfied returns true in case all expected calls in this callSet are satisfied.
func (cs callSet) Satisfied() bool {
	cs.expectedMu.Lock()
	defer cs.expectedMu.Unlock()

	for _, calls := range cs.expected {
		for _, call := range calls {
			if !call.satisfied() {
				return false
			}
		}
	}

	return true
}
