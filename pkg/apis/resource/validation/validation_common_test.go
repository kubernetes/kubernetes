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

	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

// assertFailures compares the expected against the actual errors.
//
// If they differ, it also logs what the formatted errors would look
// like to a user. This can be helpful to figure out whether an error
// is informative.
func assertFailures(tb testing.TB, want, got field.ErrorList) bool {
	tb.Helper()
	if !assert.Equal(tb, want, got) {
		logFailures(tb, "Wanted failures", want)
		logFailures(tb, "Got failures", got)
		return false
	}
	return true
}

func logFailures(tb testing.TB, header string, errs field.ErrorList) {
	tb.Helper()
	tb.Logf("%s:\n", header)
	for _, err := range errs {
		tb.Logf("- %s\n", err)
	}
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
