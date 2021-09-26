// Copyright 2014 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package prometheus

import "testing"

func TestBuildFQName(t *testing.T) {
	scenarios := []struct{ namespace, subsystem, name, result string }{
		{"a", "b", "c", "a_b_c"},
		{"", "b", "c", "b_c"},
		{"a", "", "c", "a_c"},
		{"", "", "c", "c"},
		{"a", "b", "", ""},
		{"a", "", "", ""},
		{"", "b", "", ""},
		{" ", "", "", ""},
	}

	for i, s := range scenarios {
		if want, got := s.result, BuildFQName(s.namespace, s.subsystem, s.name); want != got {
			t.Errorf("%d. want %s, got %s", i, want, got)
		}
	}
}
