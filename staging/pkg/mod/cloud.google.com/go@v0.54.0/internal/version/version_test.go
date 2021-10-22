// Copyright 2016 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package version

import "testing"

func TestGoVer(t *testing.T) {
	for _, tst := range []struct {
		in, want string
	}{
		{"go1.8", "1.8.0"},
		{"go1.7.3", "1.7.3"},
		{"go1.8.typealias", "1.8.0-typealias"},
		{"go1.8beta1", "1.8.0-beta1"},
		{"go1.8rc2", "1.8.0-rc2"},
		{"devel +824f981dd4b7 Tue Apr 29 21:41:54 2014 -0400", "824f981dd4b7"},
		{"foo bar zipzap", ""},
	} {
		if got := goVer(tst.in); got != tst.want {
			t.Errorf("goVer(%q) = %q, want %q", tst.in, got, tst.want)
		}
	}
}
