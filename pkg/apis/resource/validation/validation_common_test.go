/*
Copyright 2024 The Kubernetes Authors.

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

package validation

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/dynamic-resource-allocation/api"
)

// assertFailures compares the expected against the actual errors.
//
// If they differ, it also logs what the formatted errors would look
// like to a user. This can be helpful to figure out whether an error
// is informative.
func assertFailures(tb testing.TB, want, got field.ErrorList) bool {
	tb.Helper()
	if diff := cmp.Diff(want, got, cmpopts.IgnoreFields(field.Error{}, "Origin"), cmp.AllowUnexported(api.UniqueString{})); diff != "" {
		tb.Errorf("unexpected field errors (-want, +got):\n%s", diff)
		return false
	}
	return true
}

func TestTruncateIfTooLong(t *testing.T) {
	for name, tc := range map[string]struct {
		str       string
		maxLen    int
		expectStr string
	}{
		"nop": {
			str:       "hello",
			maxLen:    10,
			expectStr: "hello",
		},
		"truncate-to-limit": {
			str:       "hello world how are you",
			maxLen:    18,
			expectStr: "hello wo...are you",
		},
		"truncate-to-builtin-limit": {
			str:       "hello world how are you",
			maxLen:    1, // Too small, gets increased.
			expectStr: "hello w...re you",
		},
		"truncate-odd-string-even-limit": {
			str:       "abcdefghijklmnopqrs",
			maxLen:    16,
			expectStr: "abcdefg...nopqrs",
		},
		"truncate-even-string-even-limit": {
			str:       "abcdefghijklmnopqrst",
			maxLen:    16,
			expectStr: "abcdefg...opqrst",
		},
		"truncate-odd-string-odd-limit": {
			str:       "abcdefghijklmnopqrs",
			maxLen:    17,
			expectStr: "abcdefg...mnopqrs",
		},
		"truncate-even-string-odd-limit": {
			str:       "abcdefghijklmnopqrst",
			maxLen:    17,
			expectStr: "abcdefg...nopqrst",
		},
	} {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.expectStr, truncateIfTooLong(tc.str, tc.maxLen))
		})
	}
}

func TestSemverRe(t *testing.T) {
	for name, tc := range map[string]struct {
		version string
		valid   bool
	}{
		"valid-simple": {
			version: "1.2.3",
			valid:   true,
		},
		"valid-zero": {
			version: "0.0.0",
			valid:   true,
		},
		"valid-prerelease-and-build": {
			version: "1.2.3-alpha.1+build.20240305",
			valid:   true,
		},
		"invalid-leading-zero": {
			version: "01.0.0",
			valid:   false,
		},
		"invalid-not-semver": {
			version: "not-a-version",
			valid:   false,
		},
		// Numeric identifiers are capped at 18 digits so that they
		// always fit into an int64, even though semver.org itself
		// does not impose such a limit.
		"valid-18-digit-number-fits-int64": {
			version: "999999999999999999.0.0",
			valid:   true,
		},
		"invalid-19-digit-number-too-long": {
			version: "9999999999999999999.0.0",
			valid:   false,
		},
		"invalid-uint64-max-too-long": {
			version: "18446744073709551615.0.0",
			valid:   false,
		},
		// Numeric pre-release identifiers are parsed by blang/semver
		// the same way as major/minor/patch, so they are capped too.
		"valid-18-digit-prerelease-fits-int64": {
			version: "1.0.0-999999999999999999",
			valid:   true,
		},
		"invalid-19-digit-prerelease-too-long": {
			version: "1.0.0-9999999999999999999",
			valid:   false,
		},
		"invalid-19-digit-prerelease-after-alpha-too-long": {
			version: "1.0.0-alpha.9999999999999999999",
			valid:   false,
		},
	} {
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.valid, semverRe.MatchString(tc.version))
		})
	}
}
