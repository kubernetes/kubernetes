/*
Copyright 2014 The Kubernetes Authors.

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

package content

import (
	"strings"
	"testing"
)

func TestIsQualifiedName(t *testing.T) {
	// errors expected
	const nameMinSizeError = "name part: must contain at least 1 character"
	const nameMaxSizeError = "name part: must be no more than 63 characters"
	const nameStartEndError = "name part: must start and end with alphanumeric characters"
	const nameInteriorError = "name part: must contain only alphanumeric characters, '-', '_', or '.'"
	const totalFailError = "must consist of a name .* and an optional DNS subdomain prefix"
	const prefixMinSizeError = "prefix part: must contain at least 1 character"
	const prefixMaxSizeError = "prefix part: must be no more than 253 characters"
	const prefixStartEndError = "prefix part: each part must start and end with lower-case alphanumeric characters"
	const prefixInteriorError = "prefix part: each part must contain only lower-case alphanumeric characters or '-'"

	cases := []struct {
		input  string
		expect []string // regexes
	}{
		// Good values
		{"simple", nil},
		{"now-with-dashes", nil},
		{"now_with_underscores", nil},
		{"now.with.dots", nil},
		{"now-with-dashes_and_underscores.and.dots", nil},
		{"1-starts-with-num", nil},
		{"1234", nil},
		{"simple/simple", nil},
		{"now-with-dashes/simple", nil},
		{"example.com/now-with-dashes", nil},
		{"example.com/now_with_underscores", nil},
		{"example.com/now.with.dots", nil},
		{"example.com/now-with-dashes_and_underscores.and.dots", nil},
		{"1-num.2-num/3-num", nil},
		{"1234/5678", nil},
		{"1.2.3.4/5678", nil},
		{"Uppercase_Is_OK_123", nil},
		{"example.com/Uppercase_Is_OK_123", nil},
		{strings.Repeat("a", 63), nil},
		{strings.Repeat("a", 253) + "/" + strings.Repeat("b", 63), nil},

		// Bad values
		{"", mkMsgs(nameMinSizeError)},
		{" ", mkMsgs(nameStartEndError)},
		{"nospecialchars%^=@", mkMsgs(nameStartEndError, nameInteriorError)},
		{"cantendwithadash-", mkMsgs(nameStartEndError)},
		{"-cantstartwithadash-", mkMsgs(nameStartEndError)},
		{"only/one/slash", mkMsgs(totalFailError)},
		{"Example.com/abc", mkMsgs(prefixStartEndError)},
		{"example_com/abc", mkMsgs(prefixInteriorError)},
		{"example.com/", mkMsgs(nameMinSizeError)},
		{"/simple", mkMsgs(prefixMinSizeError)},
		{strings.Repeat("a", 64), mkMsgs(nameMaxSizeError)},
		{strings.Repeat("a", 254) + "/abc", mkMsgs(prefixMaxSizeError)},
	}

	for i, tc := range cases {
		result := IsQualifiedName(tc.input)
		testVerify(t, i, tc.input, tc.expect, result)
	}
}

func TestIsLabelValue(t *testing.T) {
	// errors expected
	const minSizeError = "must contain at least 1 character"
	const maxSizeError = "must be no more than 63 characters"
	const startEndError = "must start and end with alphanumeric characters"
	const interiorError = "must contain only alphanumeric characters, '-', '_', or '.'"

	cases := []struct {
		input  string
		expect []string // regexes
	}{
		// Good values
		{"", nil},
		{"simple", nil},
		{"now-with-dashes", nil},
		{"now_with_underscores", nil},
		{"now.with.dots", nil},
		{"now-with-dashes_and_underscores.and.dots", nil},
		{"1-starts-with-num", nil},
		{"1234", nil},
		{strings.Repeat("a", 63), nil},

		// Bad values
		{" ", mkMsgs(startEndError)},
		{"simple/simple", mkMsgs(interiorError)},
		{"-starts-with-dash", mkMsgs(startEndError)},
		{"ends-with-dash-", mkMsgs(startEndError)},
		{".starts.with.dot", mkMsgs(startEndError)},
		{"ends.with.dot.", mkMsgs(startEndError)},
		{"1234/5678", mkMsgs(interiorError)},
		{"nospecialchars%^=@", mkMsgs(startEndError, interiorError)},
		{"cantendwithadash-", mkMsgs(startEndError)},
		{"-cantstartwithadash-", mkMsgs(startEndError)},
		{strings.Repeat("a", 64), mkMsgs(maxSizeError)},
	}

	for i, tc := range cases {
		result := IsLabelValue(tc.input)
		testVerify(t, i, tc.input, tc.expect, result)
	}
}
