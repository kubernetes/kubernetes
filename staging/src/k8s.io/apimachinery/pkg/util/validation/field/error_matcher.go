/*
Copyright 2025 The Kubernetes Authors.

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

package field

import (
	"fmt"
	"reflect"
	"regexp"
	"strings"
)

// ErrorMatcher is a helper for comparing Error objects.
type ErrorMatcher struct {
	// TODO(thockin): consider whether type is ever NOT required, maybe just
	// assume it.
	matchType bool
	// TODO(thockin): consider whether field could be assumed - if the
	// "want" error has a nil field, don't match on field.
	matchField bool
	// TODO(thockin): consider whether value could be assumed - if the
	// "want" error has a nil value, don't match on field.
	matchValue               bool
	matchOrigin              bool
	matchDetail              func(want, got string) bool
	requireOriginWhenInvalid bool
}

// Matches returns true if the two Error objects match according to the
// configured criteria.
func (m ErrorMatcher) Matches(want, got *Error) bool {
	if m.matchType && want.Type != got.Type {
		return false
	}
	if m.matchField && want.Field != got.Field {
		return false
	}
	if m.matchValue && !reflect.DeepEqual(want.BadValue, got.BadValue) {
		return false
	}
	if m.matchOrigin {
		if want.Origin != got.Origin {
			return false
		}
		if m.requireOriginWhenInvalid && want.Type == ErrorTypeInvalid {
			if want.Origin == "" || got.Origin == "" {
				return false
			}
		}
	}
	if m.matchDetail != nil && !m.matchDetail(want.Detail, got.Detail) {
		return false
	}
	return true
}

// Render returns a string representation of the specified Error object,
// according to the criteria configured in the ErrorMatcher.
func (m ErrorMatcher) Render(e *Error) string {
	buf := strings.Builder{}

	comma := func() {
		if buf.Len() > 0 {
			buf.WriteString(", ")
		}
	}

	if m.matchType {
		comma()
		buf.WriteString(fmt.Sprintf("Type=%q", e.Type))
	}
	if m.matchField {
		comma()
		buf.WriteString(fmt.Sprintf("Field=%q", e.Field))
	}
	if m.matchValue {
		comma()
		if s, ok := e.BadValue.(string); ok {
			buf.WriteString(fmt.Sprintf("Value=%q", s))
		} else {
			rv := reflect.ValueOf(e.BadValue)
			if rv.Kind() == reflect.Pointer && !rv.IsNil() {
				rv = rv.Elem()
			}
			if rv.IsValid() && rv.CanInterface() {
				buf.WriteString(fmt.Sprintf("Value=%v", rv.Interface()))
			} else {
				buf.WriteString(fmt.Sprintf("Value=%v", e.BadValue))
			}
		}
	}
	if m.matchOrigin || m.requireOriginWhenInvalid && e.Type == ErrorTypeInvalid {
		comma()
		buf.WriteString(fmt.Sprintf("Origin=%q", e.Origin))
	}
	if m.matchDetail != nil {
		comma()
		buf.WriteString(fmt.Sprintf("Detail=%q", e.Detail))
	}
	return "{" + buf.String() + "}"
}

// Exactly returns a derived ErrorMatcher which matches all fields exactly.
func (m ErrorMatcher) Exactly() ErrorMatcher {
	return m.ByType().ByField().ByValue().ByOrigin().ByDetailExact()
}

// ByType returns a derived ErrorMatcher which also matches by type.
func (m ErrorMatcher) ByType() ErrorMatcher {
	m.matchType = true
	return m
}

// ByField returns a derived ErrorMatcher which also matches by field path.
func (m ErrorMatcher) ByField() ErrorMatcher {
	m.matchField = true
	return m
}

// ByValue returns a derived ErrorMatcher which also matches by the errant
// value.
func (m ErrorMatcher) ByValue() ErrorMatcher {
	m.matchValue = true
	return m
}

// ByOrigin returns a derived ErrorMatcher which also matches by the origin.
func (m ErrorMatcher) ByOrigin() ErrorMatcher {
	m.matchOrigin = true
	return m
}

// RequireOriginWhenInvalid returns a derived ErrorMatcher which also requires
// the Origin field to be set when the Type is Invalid and the matcher is
// matching by Origin.
func (m ErrorMatcher) RequireOriginWhenInvalid() ErrorMatcher {
	m.requireOriginWhenInvalid = true
	return m
}

// ByDetailExact returns a derived ErrorMatcher which also matches errors by
// the exact detail string.
func (m ErrorMatcher) ByDetailExact() ErrorMatcher {
	m.matchDetail = func(want, got string) bool {
		return got == want
	}
	return m
}

// ByDetailSubstring returns a derived ErrorMatcher which also matches errors
// by a substring of the detail string.
func (m ErrorMatcher) ByDetailSubstring() ErrorMatcher {
	m.matchDetail = func(want, got string) bool {
		return strings.Contains(got, want)
	}
	return m
}

// ByDetailRegexp returns a derived ErrorMatcher which also matches errors by a
// regular expression of the detail string, where the "want" string is assumed
// to be a valid regular expression.
func (m ErrorMatcher) ByDetailRegexp() ErrorMatcher {
	m.matchDetail = func(want, got string) bool {
		return regexp.MustCompile(want).MatchString(got)
	}
	return m
}

// TestIntf lets users pass a testing.T while not coupling this package to Go's
// testing package.
type TestIntf interface {
	Helper()
	Errorf(format string, args ...any)
	Logf(format string, args ...any)
}

// Test compares two ErrorLists by the criteria configured in this matcher, and
// fails the test if they don't match. If a given "want" error matches multiple
// "got" errors, they will all be consumed. This might be OK (e.g. if there are
// multiple errors on the same field from the same origin) or it might be an
// insufficiently specific matcher, so these will be logged.
func (m ErrorMatcher) Test(tb TestIntf, want, got ErrorList) {
	tb.Helper()

	remaining := got
	for _, w := range want {
		tmp := make(ErrorList, 0, len(remaining))
		n := 0
		for _, g := range remaining {
			if m.Matches(w, g) {
				n++
			} else {
				tmp = append(tmp, g)
			}
		}
		if n == 0 {
			tb.Errorf("expected an error matching:\n%s", m.Render(w))
		} else if n > 1 {
			// This is not necessarily and error, but it's worth logging in
			// case it's not what the test author intended.
			tb.Logf("multiple errors matched:\n%s", m.Render(w))
		}
		remaining = tmp
	}
	if len(remaining) > 0 {
		for _, e := range remaining {
			exactly := m.Exactly() // makes a copy
			tb.Errorf("unmatched error:\n%s", exactly.Render(e))
		}
	}
}
