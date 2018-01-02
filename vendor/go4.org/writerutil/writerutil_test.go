/*
Copyright 2016 The go4 Authors

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

package writerutil

import (
	"io"
	"testing"
)

func TestPrefixSuffixSaver(t *testing.T) {
	tests := []struct {
		N      int
		writes []string
		want   string
	}{
		{
			N:      2,
			writes: nil,
			want:   "",
		},
		{
			N:      2,
			writes: []string{"a"},
			want:   "a",
		},
		{
			N:      2,
			writes: []string{"abc", "d"},
			want:   "abcd",
		},
		{
			N:      2,
			writes: []string{"abc", "d", "e"},
			want:   "ab\n... omitting 1 bytes ...\nde",
		},
		{
			N:      2,
			writes: []string{"ab______________________yz"},
			want:   "ab\n... omitting 22 bytes ...\nyz",
		},
		{
			N:      2,
			writes: []string{"ab_______________________y", "z"},
			want:   "ab\n... omitting 23 bytes ...\nyz",
		},
	}
	for i, tt := range tests {
		w := &PrefixSuffixSaver{N: tt.N}
		for _, s := range tt.writes {
			n, err := io.WriteString(w, s)
			if err != nil || n != len(s) {
				t.Errorf("%d. WriteString(%q) = %v, %v; want %v, %v", i, s, n, err, len(s), nil)
			}
		}
		if got := string(w.Bytes()); got != tt.want {
			t.Errorf("%d. Bytes = %q; want %q", i, got, tt.want)
		}
	}
}
