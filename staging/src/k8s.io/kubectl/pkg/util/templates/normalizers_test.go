/*
Copyright 2016 The Kubernetes Authors.

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

package templates

import "testing"

func TestLongDesc(t *testing.T) {
	grid := []struct {
		In       string
		Expected string
	}{
		{
			In:       `Single line.`,
			Expected: `Single line.`,
		},
		{
			In: `
Multiple lines.

Line 2.
Line 3.
`,
			Expected: `Multiple lines.

Line 2. Line 3.`,
		},
		{
			In: `
   Whitespace trimming  preserves   inline    spaces.

  Line 2.
    Line 3.
`,
			Expected: `Whitespace trimming  preserves   inline    spaces.

Line 2. Line 3.`,
		},
		{
			In:       `TEXT_WITH_UNDERSCORE`,
			Expected: `TEXT_WITH_UNDERSCORE`,
		},
	}

	for _, g := range grid {
		actual := LongDesc(g.In)
		if actual == g.Expected {
			continue
		}

		t.Errorf("LongDesc(%q) produced unexpected output.  Actual=%q, Expected=%q", g.In, actual, g.Expected)
	}
}
