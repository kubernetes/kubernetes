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

package command

import (
	"bytes"
	"testing"
)

func TestArgOrStdin(t *testing.T) {
	tests := []struct {
		args  []string
		stdin string
		i     int
		w     string
		we    error
	}{
		{
			args: []string{
				"a",
			},
			stdin: "b",
			i:     0,
			w:     "a",
			we:    nil,
		},
		{
			args: []string{
				"a",
			},
			stdin: "b",
			i:     1,
			w:     "b",
			we:    nil,
		},
		{
			args: []string{
				"a",
			},
			stdin: "",
			i:     1,
			w:     "",
			we:    ErrNoAvailSrc,
		},
	}

	for i, tt := range tests {
		var b bytes.Buffer
		b.Write([]byte(tt.stdin))
		g, ge := argOrStdin(tt.args, &b, tt.i)
		if g != tt.w {
			t.Errorf("#%d: expect %v, not %v", i, tt.w, g)
		}
		if ge != tt.we {
			t.Errorf("#%d: expect %v, not %v", i, tt.we, ge)
		}
	}
}
