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

package strings

import (
	"testing"
)

func TestEscapeQualifiedNameForDisk(t *testing.T) {
	testCases := []struct {
		input  string
		output string
	}{
		{"kubernetes.io/blah", "kubernetes.io~blah"},
		{"blah/blerg/borg", "blah~blerg~borg"},
		{"kubernetes.io", "kubernetes.io"},
	}
	for i, tc := range testCases {
		escapee := EscapeQualifiedName(tc.input)
		if escapee != tc.output {
			t.Errorf("case[%d]: expected (%q), got (%q)", i, tc.output, escapee)
		}
		original := UnescapeQualifiedName(escapee)
		if original != tc.input {
			t.Errorf("case[%d]: expected (%q), got (%q)", i, tc.input, original)
		}
	}
}
