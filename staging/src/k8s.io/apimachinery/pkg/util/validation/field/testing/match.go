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

package testing

import (
	"fmt"
	"reflect"
	"regexp"
	"strings"
	"testing"

	field "k8s.io/apimachinery/pkg/util/validation/field"
)

// MatchErrors compares two ErrorLists with the specified matcher, and fails
// the test if they don't match. If a given "want" error matches multiple "got"
// errors, they will all be consumed. This might be OK (e.g. if there are
// multiple errors on the same field from the same origin) or it might be an
// insufficiently specific matcher, so these will be logged.
func MatchErrors(t *testing.T, want, got field.ErrorList, matcher *Matcher) {
	t.Helper()

	remaining := got
	for _, w := range want {
		tmp := make(field.ErrorList, 0, len(remaining))
		n := 0
		for _, g := range remaining {
			if matcher.Matches(w, g) {
				n++
			} else {
				tmp = append(tmp, g)
			}
		}
		if n == 0 {
			t.Errorf("expected an error matching:\n%s", matcher.Render(w))
		} else if n > 1 {
			// This is not necessarily and error, but it's worth logging in
			// case it's not what the test author intended.
			t.Logf("multiple errors matched:\n%s", matcher.Render(w))
		}
		remaining = tmp
	}
	if len(remaining) > 0 {
		for _, e := range remaining {
			t.Errorf("unmatched error:\n%s", Match().Exactly().Render(e))
		}
	}
}

// Match returns a new Matcher.
func Match() *Matcher {
	return &Matcher{}
}

// Matcher is a helper for comparing field.Error objects.
type Matcher struct {
	matchType                bool
	matchField               bool
	matchValue               bool
	matchOrigin              bool
	matchDetail              func(want, got string) bool
	requireOriginWhenInvalid bool
}

// Matches returns true if the two field.Error objects match according to the
// configured criteria.
func (m *Matcher) Matches(want, got *field.Error) bool {
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
		if m.requireOriginWhenInvalid && want.Type == field.ErrorTypeInvalid {
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
// according to the criteria configured in the Matcher.
func (m *Matcher) Render(e *field.Error) string {
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
		buf.WriteString(fmt.Sprintf("Value=%v", e.BadValue))
	}
	if m.matchOrigin || m.requireOriginWhenInvalid && e.Type == field.ErrorTypeInvalid {
		comma()
		buf.WriteString(fmt.Sprintf("Origin=%q", e.Origin))
	}
	if m.matchDetail != nil {
		comma()
		buf.WriteString(fmt.Sprintf("Detail=%q", e.Detail))
	}
	return "{" + buf.String() + "}"
}

// Exactly configures the matcher to match all fields exactly.
func (m *Matcher) Exactly() *Matcher {
	return m.ByType().ByField().ByValue().ByOrigin().ByDetailExact()
}

// Exactly configures the matcher to match errors by type.
func (m *Matcher) ByType() *Matcher {
	m.matchType = true
	return m
}

// ByField configures the matcher to match errors by field path.
func (m *Matcher) ByField() *Matcher {
	m.matchField = true
	return m
}

// ByValue configures the matcher to match errors by the errant value.
func (m *Matcher) ByValue() *Matcher {
	m.matchValue = true
	return m
}

// ByOrigin configures the matcher to match errors by the origin.
func (m *Matcher) ByOrigin() *Matcher {
	m.matchOrigin = true
	return m
}

// RequireOriginWhenInvalid configures the matcher to require the Origin field
// to be set when the Type is Invalid and the matcher is matching by Origin.
func (m *Matcher) RequireOriginWhenInvalid() *Matcher {
	m.requireOriginWhenInvalid = true
	return m
}

// ByDetailExact configures the matcher to match errors by the exact detail
// string.
func (m *Matcher) ByDetailExact() *Matcher {
	m.matchDetail = func(want, got string) bool {
		return got == want
	}
	return m
}

// ByDetailSubstring configures the matcher to match errors by a substring of
// the detail string.
func (m *Matcher) ByDetailSubstring() *Matcher {
	m.matchDetail = func(want, got string) bool {
		return strings.Contains(got, want)
	}
	return m
}

// ByDetailRegexp configures the matcher to match errors by a regular
// expression of the detail string, where the "want" string is assumed to be a
// valid regular expression.
func (m *Matcher) ByDetailRegexp() *Matcher {
	m.matchDetail = func(want, got string) bool {
		return regexp.MustCompile(want).MatchString(got)
	}
	return m
}
