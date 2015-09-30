// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pathutil

import "testing"

func TestCanonicalURLPath(t *testing.T) {
	tests := []struct {
		p  string
		wp string
	}{
		{"/a", "/a"},
		{"", "/"},
		{"a", "/a"},
		{"//a", "/a"},
		{"/a/.", "/a"},
		{"/a/..", "/"},
		{"/a/", "/a/"},
		{"/a//", "/a/"},
	}
	for i, tt := range tests {
		if g := CanonicalURLPath(tt.p); g != tt.wp {
			t.Errorf("#%d: canonical path = %s, want %s", i, g, tt.wp)
		}
	}
}
