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

package names

import (
	"strings"
	"testing"

	utilrand "k8s.io/apimachinery/pkg/util/rand"
)

func TestSimpleNameGenerator(t *testing.T) {
	const maxPrefixLen = maxNameLength - randomLength
	mkBase := func(n int) string {
		return utilrand.String(n)
	}

	cases := []struct {
		base        string
		expectLen   int
		expectTrunc int
	}{{
		// Normal use
		base:      mkBase(3),
		expectLen: 3 + randomLength,
	}, {
		// Edge case: long name, just enough for suffix
		base:      mkBase(maxPrefixLen),
		expectLen: maxNameLength,
	}, {
		// Edge case: long name trucated
		base:        mkBase(maxPrefixLen + 1),
		expectLen:   maxNameLength,
		expectTrunc: 1,
	}, {
		// Edge case: max length name trucated
		base:        mkBase(maxNameLength),
		expectLen:   maxNameLength,
		expectTrunc: randomLength,
	}, {
		// Edge case: very long name trucated
		base:        mkBase(maxNameLength * 2),
		expectLen:   maxNameLength,
		expectTrunc: maxNameLength + randomLength,
	}}
	for i, tc := range cases {
		// Set the first character to be truncated (if any) to something
		// not in the random character set, so we can prove it was replaced by
		// random characters.
		setSentinel := func(base string) string {
			if len(base) <= maxPrefixLen {
				return base
			}
			baseBytes := []byte(base)
			baseBytes[maxPrefixLen] = '!'
			return string(baseBytes)
		}

		base := setSentinel(tc.base)
		name := SimpleNameGenerator.GenerateName(base)
		if len(name) != tc.expectLen {
			t.Errorf("case[%d]: wrong result len: expected %d, got %d (%q)", i, tc.expectLen, len(name), name)
		}
		if name == base {
			t.Errorf("case[%d]: didn't randomize: %q", i, name)
		}
		if pfx := base[0 : len(base)-tc.expectTrunc]; !strings.HasPrefix(name, pfx) {
			t.Errorf("case[%d]: wrong result prefix: expected base %q, got result %q", i, pfx, name)
		}
		if tc.expectTrunc > 0 {
			// pfx = everything we expect to keep + 1, which will not be a
			// valid suffix character.
			if pfx := base[0 : len(base)-(tc.expectTrunc)+1]; strings.HasPrefix(name, pfx) {
				t.Errorf("case[%d]: didn't truncate base: result %q must not be a prefix of %q", i, pfx, name)
			}
		}
	}
}
