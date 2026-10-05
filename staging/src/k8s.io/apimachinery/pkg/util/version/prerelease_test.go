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

package version

import "testing"

func TestSemanticPrereleaseIdentifierOrdering(t *testing.T) {
	cases := []struct{ low, high string }{
		{"1.0.0-1", "1.0.0-0a"},
		{"1.0.0-1", "1.0.0--"},
		{"1.0.0-9", "1.0.0-10"},
		{"1.0.0-9", "1.0.0-18446744073709551616"},
		{"1.0.0-18446744073709551616", "1.0.0-184467440737095516160"},
		{"1.0.0-18446744073709551616", "1.0.0-2a"},
		{"1.0.0-rc.2", "1.0.0-rc.10"},
		{"1.0.0-rc.1", "1.0.0-rc.0a"},
		{"1.0.0-alpha", "1.0.0-beta"},
		{"1.0.0-alpha", "1.0.0-alpha.1"},
		{"1.0.0-1", "1.0.0"},
	}
	for _, tc := range cases {
		t.Run(tc.low+"_before_"+tc.high, func(t *testing.T) {
			low, err := ParseSemantic(tc.low)
			if err != nil {
				t.Fatal(err)
			}
			high, err := ParseSemantic(tc.high)
			if err != nil {
				t.Fatal(err)
			}
			if !low.LessThan(high) || !high.GreaterThan(low) || low.EqualTo(high) {
				t.Errorf("expected %s < %s", tc.low, tc.high)
			}
		})
	}
}

func TestSemanticLargeNumericPrereleaseLeadingZeros(t *testing.T) {
	for _, input := range []string{"1.0.0-018446744073709551616", "1.0.0-rc.018446744073709551616"} {
		t.Run(input, func(t *testing.T) {
			if _, err := ParseSemantic(input); err == nil {
				t.Errorf("accepted numeric prerelease identifier with a leading zero: %s", input)
			}
		})
	}
}
