/*
Copyright 2015 The Kubernetes Authors.

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

package runtime

import (
	"testing"
)

func TestFmtRawDoc(t *testing.T) {
	tests := []struct {
		t, expected      string
		ignoreGenerators bool
	}{
		{"aaa\n  --- asd\n TODO: tooooodo\n toooodoooooo\n", "aaa", true},
		{"aaa\n+xyz\n  --- asd\n TODO: tooooodo\n toooodoooooo\n", "aaa +xyz", false},
		{"aaa\nasd\n TODO: tooooodo\nbbbb\n --- toooodoooooo\n", "aaa asd bbbb", true},
		{" TODO: tooooodo\n", "", true},
		{"Par1\n\nPar2\n\n", "Par1\\n\\nPar2", true},
		{"", "", true},
		{" ", "", true},
		{" \n", "", true},
		{" \n\n ", "", true},
		{"Example:\n\tl1\n\t\tl2\n", "Example:\\n\\tl1\\n\\t\\tl2", true},
	}

	for _, test := range tests {
		if o := fmtRawDoc(test.t, test.ignoreGenerators); o != test.expected {
			t.Fatalf("Expected: %q, got %q", test.expected, o)
		}
	}
}
