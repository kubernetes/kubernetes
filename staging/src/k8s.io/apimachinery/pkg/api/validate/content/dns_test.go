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

func TestIsDNS1123Label(t *testing.T) {
	// errors expected
	const minSizeError = "must contain at least 1 character"
	const maxSizeError = "must be no more than 63 characters"
	const startEndError = "must start and end with lower-case alphanumeric characters"
	const interiorError = "must contain only lower-case alphanumeric characters or '-'"

	cases := []struct {
		input  string
		expect []string // regexes
	}{
		// Good values
		{"a", nil},
		{"ab", nil},
		{"abc", nil},
		{"a1", nil},
		{"a-1", nil},
		{"a--1--2--b", nil},
		{"0", nil},
		{"01", nil},
		{"012", nil},
		{"1a", nil},
		{"1-a", nil},
		{"1--a--b--2", nil},
		{strings.Repeat("a", 63), nil},

		// Bad values
		{"", mkMsgs(minSizeError)},
		{"A", mkMsgs(startEndError)},
		{"ABC", mkMsgs(startEndError, interiorError)},
		{"aBc", mkMsgs(interiorError)},
		{"AbC", mkMsgs(startEndError)},
		{"A1", mkMsgs(startEndError)},
		{"A-1", mkMsgs(startEndError)},
		{"1-A", mkMsgs(startEndError)},
		{"-", mkMsgs(startEndError)},
		{"a-", mkMsgs(startEndError)},
		{"-a", mkMsgs(startEndError)},
		{"1-", mkMsgs(startEndError)},
		{"-1", mkMsgs(startEndError)},
		{"_", mkMsgs(startEndError)},
		{"a_", mkMsgs(startEndError)},
		{"_a", mkMsgs(startEndError)},
		{"a_b", mkMsgs(interiorError)},
		{"1_", mkMsgs(startEndError)},
		{"_1", mkMsgs(startEndError)},
		{"1_2", mkMsgs(interiorError)},
		{".", mkMsgs(startEndError)},
		{"a.", mkMsgs(startEndError)},
		{".a", mkMsgs(startEndError)},
		{"a.b", mkMsgs(interiorError)},
		{"1.", mkMsgs(startEndError)},
		{".1", mkMsgs(startEndError)},
		{"1.2", mkMsgs(interiorError)},
		{" ", mkMsgs(startEndError)},
		{"a ", mkMsgs(startEndError)},
		{" a", mkMsgs(startEndError)},
		{"a b", mkMsgs(interiorError)},
		{"1 ", mkMsgs(startEndError)},
		{" 1", mkMsgs(startEndError)},
		{"1 2", mkMsgs(interiorError)},
		{strings.Repeat("a", 64), mkMsgs(maxSizeError)},
		{strings.Repeat("-", 64), mkMsgs(maxSizeError)},
		{strings.Repeat(".", 64), mkMsgs(maxSizeError)},
		{strings.Repeat("aBc", 64), mkMsgs(maxSizeError)},
		{strings.Repeat("AbC", 64), mkMsgs(maxSizeError)},
	}

	for i, tc := range cases {
		result := IsDNS1123Label(tc.input)
		testVerify(t, i, tc.input, tc.expect, result)
	}
}

func TestIsDNS1123Subdomain(t *testing.T) {
	// errors expected
	const minSizeError = "must contain at least 1 character"
	const partMinSizeError = "each part must contain at least 1 character"
	const maxSizeError = "must be no more than 253 characters"
	const startEndError = "must start and end with lower-case alphanumeric characters"
	const partStartEndError = "each part must start and end with lower-case alphanumeric characters"
	const partInteriorError = "each part must contain only lower-case alphanumeric characters or '-'"

	cases := []struct {
		input  string
		expect []string // regexes
	}{
		// Good values
		{"a", nil},
		{"ab", nil},
		{"abc", nil},
		{"a1", nil},
		{"a-1", nil},
		{"a--1--2--b", nil},
		{"0", nil},
		{"01", nil},
		{"012", nil},
		{"1a", nil},
		{"1-a", nil},
		{"1--a--b--2", nil},
		{strings.Repeat("a", 63), nil},
		{strings.Repeat("a", 64), nil},  // allowed for backwards compat
		{strings.Repeat("a", 253), nil}, // allowed for backwards compat
		{"a.a", nil},
		{"ab.a", nil},
		{"abc.a", nil},
		{"a1.a", nil},
		{"a-1.a", nil},
		{"a--1--2--b.a", nil},
		{"0.a", nil},
		{"01.a", nil},
		{"012.a", nil},
		{"1a.a", nil},
		{"1-a.a", nil},
		{"1--a--b--2.a", nil},
		{strings.Repeat("a", 63) + ".a", nil},
		{strings.Repeat("a", 64) + ".a", nil},  // allowed for backwards compat
		{strings.Repeat("a", 251) + ".a", nil}, // allowed for backwards compat
		{"a.1", nil},
		{"ab.1", nil},
		{"abc.1", nil},
		{"a1.1", nil},
		{"a-1.1", nil},
		{"a--1--2--b.1", nil},
		{"0.1", nil},
		{"01.1", nil},
		{"012.1", nil},
		{"1a.1", nil},
		{"1-a.1", nil},
		{"1--a--b--2.1", nil},
		{strings.Repeat("a", 63) + ".1", nil},
		{strings.Repeat("a", 64) + ".1", nil},  // allowed for backwards compat
		{strings.Repeat("a", 251) + ".1", nil}, // allowed for backwards compat
		{"a.b.c.d.e", nil},
		{"aa.bb.cc.dd.ee", nil},
		{"1.2.3.4.5", nil},
		{"11.22.33.44.55", nil},
		{strings.Repeat("a", 126) + "." + strings.Repeat("b", 126), nil},
		{strings.Repeat("0", 126) + "." + strings.Repeat("1", 126), nil},
		{strings.Repeat("0.", 126) + "0", nil},

		// Bad values
		{"", mkMsgs(minSizeError)},
		{"A", mkMsgs(startEndError)},
		{"ABC", mkMsgs(startEndError, partInteriorError)},
		{"aBc", mkMsgs(partInteriorError)},
		{"AbC", mkMsgs(startEndError)},
		{"A1", mkMsgs(startEndError)},
		{"A-1", mkMsgs(startEndError)},
		{"1-A", mkMsgs(startEndError)},
		{"-", mkMsgs(startEndError)},
		{"a-", mkMsgs(startEndError)},
		{"-a", mkMsgs(startEndError)},
		{"1-", mkMsgs(startEndError)},
		{"-1", mkMsgs(startEndError)},
		{"_", mkMsgs(startEndError)},
		{"a_", mkMsgs(startEndError)},
		{"_a", mkMsgs(startEndError)},
		{"a_b", mkMsgs(partInteriorError)},
		{"1_", mkMsgs(startEndError)},
		{"_1", mkMsgs(startEndError)},
		{"1_2", mkMsgs(partInteriorError)},
		{".", mkMsgs(startEndError)},
		{"a.", mkMsgs(startEndError)},
		{".a", mkMsgs(startEndError)},
		{"1.", mkMsgs(startEndError)},
		{".1", mkMsgs(startEndError)},
		{"a..b", mkMsgs(partMinSizeError)},
		{"0..1", mkMsgs(partMinSizeError)},
		{" ", mkMsgs(startEndError)},
		{"a ", mkMsgs(startEndError)},
		{" a", mkMsgs(startEndError)},
		{"a b", mkMsgs(partInteriorError)},
		{"1 ", mkMsgs(startEndError)},
		{" 1", mkMsgs(startEndError)},
		{"1 2", mkMsgs(partInteriorError)},
		{"A.b", mkMsgs(startEndError)},
		{"aB.b", mkMsgs(partStartEndError)},
		{"ab.A", mkMsgs(startEndError)},
		{"A.0", mkMsgs(startEndError)},
		{"0B.0", mkMsgs(partStartEndError)},
		{"00.A", mkMsgs(startEndError)},
		{"A.B.C.D.E", mkMsgs(startEndError)},
		{"AA.BB.CC.DD.EE", mkMsgs(startEndError)},
		{"a.B.c.d.e", mkMsgs(startEndError)},
		{"aa.bB.cc.dd.ee", mkMsgs(startEndError)},
		{"a@b", mkMsgs(partInteriorError)},
		{"a,b", mkMsgs(partInteriorError)},
		{"a_b", mkMsgs(partInteriorError)},
		{"a;b", mkMsgs(partInteriorError)},
		{"a:b", mkMsgs(partInteriorError)},
		{`a%b`, mkMsgs(partInteriorError)},
		{"a?b", mkMsgs(partInteriorError)},
		{"a$b", mkMsgs(partInteriorError)},
		{strings.Repeat("a", 254), mkMsgs(maxSizeError)},
		{strings.Repeat("-", 254), mkMsgs(maxSizeError)},
		{strings.Repeat(".", 254), mkMsgs(maxSizeError)},
	}

	for i, tc := range cases {
		result := IsDNS1123Subdomain(tc.input)
		testVerify(t, i, tc.input, tc.expect, result)
	}
}
