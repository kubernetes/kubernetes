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

// NormalizationRule holds a pre-compiled regular expression and its replacement string
// for normalizing field paths.
type NormalizationRule struct {
	Regexp      *regexp.Regexp
	Replacement string
}

// ErrorMatcher is a helper for comparing Error objects.
type ErrorMatcher struct {
	// TODO(thockin): consider whether type is ever NOT required, maybe just
	// assume it.
	matchType bool
	// TODO(thockin): consider whether field could be assumed - if the
	// "want" error has a nil field, don't match on field.
	matchField bool
	// TODO(thockin): consider whether value could be assumed - if the
	// "want" error has a nil value, don't match on value.
	matchValue                    bool
	matchOrigin                   bool
	matchDetail                   func(want, got string) bool
	requireOriginWhenInvalid      bool
	matchValidationStabilityLevel bool
	matchSource                   bool
	matchAncestorShortCircuit     bool
	matchShortCircuit             bool
	// normalizationRules holds the pre-compiled regex patterns for path normalization.
	normalizationRules []NormalizationRule
}

// isChildPath returns true if child is a descendant path of parent.
// It avoids false positives like "spec.containers" being a child of "spec.container"
// by explicitly checking for '.' or '[' path separators.
func isChildPath(parent, child string) bool {
	// "" as parent path not supported. It can un-intentionally match with any path.
	// theoretically system errors can be matched with any path errors.
	if len(parent) == 0 {
		return false
	}
	if len(child) <= len(parent) {
		return false
	}
	if child[:len(parent)] != parent {
		return false
	}
	sep := child[len(parent)]
	return sep == '.' || sep == '['
}

// Matches returns true if the two Error objects match according to the
// configured criteria. When field normalization is configured, only the
// "got" error's field path is normalized (to bring older API versions up
// to the internal/latest format), while "want" is assumed to already be
// in the canonical internal API format.
func (m ErrorMatcher) Matches(want, got *Error) bool {
	gotField := got.Field
	if want.Field != gotField {
		gotField = m.normalizePath(gotField)
	}
	if m.matchAncestorShortCircuit {
		if got.ShortCircuit && (isChildPath(gotField, want.Field) || isChildPath(got.Field, want.Field)) {
			return true
		}
	}

	if m.matchType && want.Type != got.Type {
		return false
	}
	if m.matchField && want.Field != gotField {
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
	if m.matchValidationStabilityLevel && want.ValidationStabilityLevel != got.ValidationStabilityLevel {
		return false
	}
	if m.matchSource && want.FromImperative != got.FromImperative {
		return false
	}
	if m.matchShortCircuit && want.ShortCircuit != got.ShortCircuit {
		return false
	}

	return true
}

// normalizePath applies configured path normalization rules.
func (m ErrorMatcher) normalizePath(path string) string {
	for _, rule := range m.normalizationRules {
		normalized := rule.Regexp.ReplaceAllString(path, rule.Replacement)
		if normalized != path {
			// Only apply the first matching rule.
			return normalized
		}
	}
	return path
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
		fmt.Fprintf(&buf, "Type=%q", e.Type)
	}
	if m.matchField {
		comma()
		if normalized := m.normalizePath(e.Field); normalized != e.Field {
			fmt.Fprintf(&buf, "Field=%q (aka %q)", normalized, e.Field)
		} else {
			fmt.Fprintf(&buf, "Field=%q", e.Field)
		}
	}
	if m.matchValue {
		comma()
		if s, ok := e.BadValue.(string); ok {
			fmt.Fprintf(&buf, "Value=%q", s)
		} else {
			rv := reflect.ValueOf(e.BadValue)
			if rv.Kind() == reflect.Pointer && !rv.IsNil() {
				rv = rv.Elem()
			}
			if rv.IsValid() && rv.CanInterface() {
				fmt.Fprintf(&buf, "Value=%v", rv.Interface())
			} else {
				fmt.Fprintf(&buf, "Value=%v", e.BadValue)
			}
		}
	}
	if m.matchOrigin || m.requireOriginWhenInvalid && e.Type == ErrorTypeInvalid {
		comma()
		fmt.Fprintf(&buf, "Origin=%q", e.Origin)
	}
	if m.matchDetail != nil {
		comma()
		fmt.Fprintf(&buf, "Detail=%q", e.Detail)
	}
	if m.matchValidationStabilityLevel {
		comma()
		fmt.Fprintf(&buf, "ValidationStabilityLevel=%s", e.ValidationStabilityLevel)
	}
	if m.matchSource {
		comma()
		fmt.Fprintf(&buf, "FromImperative=%t", e.FromImperative)
	}
	if m.matchShortCircuit {
		comma()
		fmt.Fprintf(&buf, "ShortCircuit=%t", e.ShortCircuit)
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
// If you need to mutate the field path (e.g. to normalize across versions),
// see ByFieldNormalized.
func (m ErrorMatcher) ByField() ErrorMatcher {
	m.matchField = true
	return m
}

// ByFieldNormalized returns a derived ErrorMatcher which also matches by field path
// after applying normalization rules to the actual (got) error's field path.
// This allows matching field paths from older API versions against the canonical
// internal API format.
//
// The normalization rules are applied ONLY to the "got" error's field path, bringing
// older API version field paths up to the latest/internal format. The "want" error
// is assumed to always be in the internal API format (latest).
//
// The rules slice holds pre-compiled regular expressions and their replacement strings.
//
// Example:
//
//	rules := []NormalizationRule{
//	  {
//	    Regexp:      regexp.MustCompile(`spec\.devices\.requests\[(\d+)\]\.allocationMode`),
//	    Replacement: "spec.devices.requests[$1].exactly.allocationMode",
//	  },
//	}
//	matcher := ErrorMatcher{}.ByFieldNormalized(rules)
func (m ErrorMatcher) ByFieldNormalized(rules []NormalizationRule) ErrorMatcher {
	m.matchField = true
	m.normalizationRules = rules
	return m
}

// ByValue returns a derived ErrorMatcher which also matches by the errant
// value.
func (m ErrorMatcher) ByValue() ErrorMatcher {
	m.matchValue = true
	return m
}

// ByOrigin returns a derived ErrorMatcher which also matches by the origin.
// When this is used and an origin is set in the error, the matcher will
// consider all expected errors with the same origin to be a match. The only
// expception to this is when it finds two errors which are exactly identical,
// which is too suspicious to ignore. This multi-matching allows tests to
// express a single expectation ("I set the X field to an invalid value, and I
// expect an error from origin Y") without having to know exactly how many
// errors might be returned, or in what order, or with what wording.
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

// BySource returns a derived ErrorMatcher which also matches by the error origination
// value of field errors.
func (m ErrorMatcher) BySource() ErrorMatcher {
	m.matchSource = true
	return m
}

// MatchAncestorShortCircuit returns a derived ErrorMatcher which also matches when the "got" error short-circuited at an ancestor of the "want" error's field path.
func (m ErrorMatcher) MatchAncestorShortCircuit() ErrorMatcher {
	m.matchAncestorShortCircuit = true
	return m
}

// MatchShortCircuit returns a derived ErrorMatcher which also matches by the ShortCircuit value.
func (m ErrorMatcher) MatchShortCircuit() ErrorMatcher {
	m.matchShortCircuit = true
	return m
}

// ByValidationStabilityLevel returns a derived ErrorMatcher which also matches by the validation stability level
// value of field errors.
func (m ErrorMatcher) ByValidationStabilityLevel() ErrorMatcher {
	m.matchValidationStabilityLevel = true
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
}

// Test compares two ErrorLists by the criteria configured in this matcher, and
// fails the test if they don't match. The "want" errors are expected to be in
// the internal API format (latest), while "got" errors may be from any API version
// and will be normalized if field normalization rules are configured.
//
// If matching by origin is enabled and the error has a non-empty origin, a given
// "want" error can match multiple "got" errors, and they will all be consumed.
// The only exception to this is if the matcher got multiple identical (in every way,
// even those not being matched on) errors, which is likely to indicate a bug.
// This doesn't support matchAncestorShortCircuit as it it not needed to be used in the tests.
func (m ErrorMatcher) Test(tb TestIntf, want, got ErrorList) {
	tb.Helper()

	if m.matchAncestorShortCircuit {
		tb.Errorf("matchAncestorShortCircuit is not supported for test")
	}
	exactly := m.Exactly() // makes a copy

	// If we ever find an EXACT duplicate error, it's almost certainly a bug
	// worth reporting. If we ever find a use-case where this is not a bug, we
	// can revisit this assumption.
	seen := map[string]bool{}
	for _, g := range got {
		key := exactly.Render(g)
		if seen[key] {
			tb.Errorf("exact duplicate error:\n%s", key)
		}
		seen[key] = true
	}

	remaining := got
	for _, w := range want {
		tmp := make(ErrorList, 0, len(remaining))
		matched := false
		for i, g := range remaining {
			if m.Matches(w, g) {
				matched = true
				if m.matchOrigin && w.Origin != "" {
					// When origin is included in the match, we allow multiple
					// matches against the same wanted error, so that tests
					// can be insulated from the exact number, order, and
					// wording of cases that might return more than one error.
					continue
				} else {
					// Single-match, save the rest of the "got" errors and move
					// on to the next "want" error.
					tmp = append(tmp, remaining[i+1:]...)
					break
				}
			} else {
				tmp = append(tmp, g)
			}
		}
		if !matched {
			tb.Errorf("expected an error matching:\n%s", m.Render(w))
		}
		remaining = tmp
	}
	if len(remaining) > 0 {
		for _, e := range remaining {
			tb.Errorf("unmatched error:\n%s", exactly.Render(e))
		}
	}
}
