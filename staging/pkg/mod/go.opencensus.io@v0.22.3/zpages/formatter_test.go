// Copyright 2018, OpenCensus Authors
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
//

package zpages

import "testing"

func TestCountFormatter(t *testing.T) {
	tests := []struct {
		in   uint64
		want string
	}{
		{0, " "},
		{1, "1"},
		{1024, "1024"},
		{1e5, "100000"},
		{1e6, "1.000 M "},
		{1e9, "1.000 G "},
		{1e8 + 2e9, "2.100 G "},
		{1e12, "1.000 T "},
		{1e15, "1.000 P "},
		{1e18, "1.000 E "},
	}

	for _, tt := range tests {
		if g, w := countFormatter(tt.in), tt.want; g != w {
			t.Errorf("%d got %q want %q", tt.in, g, w)
		}
	}
}
