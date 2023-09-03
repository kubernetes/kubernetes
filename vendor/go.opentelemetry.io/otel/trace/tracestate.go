// Copyright The OpenTelemetry Authors
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

package trace // import "go.opentelemetry.io/otel/trace"

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

const (
	maxListMembers = 32

	listDelimiter = ","

	// based on the W3C Trace Context specification, see
	// https://www.w3.org/TR/trace-context-1/#tracestate-header
	noTenantKeyFormat   = `[a-z][_0-9a-z\-\*\/]{0,255}`
	withTenantKeyFormat = `[a-z0-9][_0-9a-z\-\*\/]{0,240}@[a-z][_0-9a-z\-\*\/]{0,13}`
	valueFormat         = `[\x20-\x2b\x2d-\x3c\x3e-\x7e]{0,255}[\x21-\x2b\x2d-\x3c\x3e-\x7e]`

	errInvalidKey    errorConst = "invalid tracestate key"
	errInvalidValue  errorConst = "invalid tracestate value"
	errInvalidMember errorConst = "invalid tracestate list-member"
	errMemberNumber  errorConst = "too many list-members in tracestate"
	errDuplicate     errorConst = "duplicate list-member in tracestate"
)

var (
	keyRe    = regexp.MustCompile(`^((` + noTenantKeyFormat + `)|(` + withTenantKeyFormat + `))$`)
	valueRe  = regexp.MustCompile(`^(` + valueFormat + `)$`)
	memberRe = regexp.MustCompile(`^\s*((` + noTenantKeyFormat + `)|(` + withTenantKeyFormat + `))=(` + valueFormat + `)\s*$`)
)

type member struct {
	Key   string
	Value string
}

func newMember(key, value string) (member, error) {
	if !keyRe.MatchString(key) {
		return member{}, fmt.Errorf("%w: %s", errInvalidKey, key)
	}
	if !valueRe.MatchString(value) {
		return member{}, fmt.Errorf("%w: %s", errInvalidValue, value)
	}
	return member{Key: key, Value: value}, nil
}

func parseMember(m string) (member, error) {
	matches := memberRe.FindStringSubmatch(m)
	if len(matches) != 5 {
		return member{}, fmt.Errorf("%w: %s", errInvalidMember, m)
	}

	return member{
		Key:   matches[1],
		Value: matches[4],
	}, nil
}

// String encodes member into a string compliant with the W3C Trace Context
// specification.
func (m member) String() string {
	return fmt.Sprintf("%s=%s", m.Key, m.Value)
}

// TraceState provides additional vendor-specific trace identification
// information across different distributed tracing systems. It represents an
// immutable list consisting of key/value pairs, each pair is referred to as a
// list-member.
//
// TraceState conforms to the W3C Trace Context specification
// (https://www.w3.org/TR/trace-context-1). All operations that create or copy
// a TraceState do so by validating all input and will only produce TraceState
// that conform to the specification. Specifically, this means that all
// list-member's key/value pairs are valid, no duplicate list-members exist,
// and the maximum number of list-members (32) is not exceeded.
type TraceState struct { //nolint:revive // revive complains about stutter of `trace.TraceState`
	// list is the members in order.
	list []member
}

var _ json.Marshaler = TraceState{}

// ParseTraceState attempts to decode a TraceState from the passed
// string. It returns an error if the input is invalid according to the W3C
// Trace Context specification.
func ParseTraceState(tracestate string) (TraceState, error) {
	if tracestate == "" {
		return TraceState{}, nil
	}

	wrapErr := func(err error) error {
		return fmt.Errorf("failed to parse tracestate: %w", err)
	}

	var members []member
	found := make(map[string]struct{})
	for _, memberStr := range strings.Split(tracestate, listDelimiter) {
		if len(memberStr) == 0 {
			continue
		}

		m, err := parseMember(memberStr)
		if err != nil {
			return TraceState{}, wrapErr(err)
		}

		if _, ok := found[m.Key]; ok {
			return TraceState{}, wrapErr(errDuplicate)
		}
		found[m.Key] = struct{}{}

		members = append(members, m)
		if n := len(members); n > maxListMembers {
			return TraceState{}, wrapErr(errMemberNumber)
		}
	}

	return TraceState{list: members}, nil
}

// MarshalJSON marshals the TraceState into JSON.
func (ts TraceState) MarshalJSON() ([]byte, error) {
	return json.Marshal(ts.String())
}

// String encodes the TraceState into a string compliant with the W3C
// Trace Context specification. The returned string will be invalid if the
// TraceState contains any invalid members.
func (ts TraceState) String() string {
	members := make([]string, len(ts.list))
	for i, m := range ts.list {
		members[i] = m.String()
	}
	return strings.Join(members, listDelimiter)
}

// Get returns the value paired with key from the corresponding TraceState
// list-member if it exists, otherwise an empty string is returned.
func (ts TraceState) Get(key string) string {
	for _, member := range ts.list {
		if member.Key == key {
			return member.Value
		}
	}

	return ""
}

// Insert adds a new list-member defined by the key/value pair to the
// TraceState. If a list-member already exists for the given key, that
// list-member's value is updated. The new or updated list-member is always
// moved to the beginning of the TraceState as specified by the W3C Trace
// Context specification.
//
// If key or value are invalid according to the W3C Trace Context
// specification an error is returned with the original TraceState.
//
// If adding a new list-member means the TraceState would have more members
// then is allowed, the new list-member will be inserted and the right-most
// list-member will be dropped in the returned TraceState.
func (ts TraceState) Insert(key, value string) (TraceState, error) {
	m, err := newMember(key, value)
	if err != nil {
		return ts, err
	}

	cTS := ts.Delete(key)
	if cTS.Len()+1 <= maxListMembers {
		cTS.list = append(cTS.list, member{})
	}
	// When the number of members exceeds capacity, drop the "right-most".
	copy(cTS.list[1:], cTS.list)
	cTS.list[0] = m

	return cTS, nil
}

// Delete returns a copy of the TraceState with the list-member identified by
// key removed.
func (ts TraceState) Delete(key string) TraceState {
	members := make([]member, ts.Len())
	copy(members, ts.list)
	for i, member := range ts.list {
		if member.Key == key {
			members = append(members[:i], members[i+1:]...)
			// TraceState should contain no duplicate members.
			break
		}
	}
	return TraceState{list: members}
}

// Len returns the number of list-members in the TraceState.
func (ts TraceState) Len() int {
	return len(ts.list)
}
