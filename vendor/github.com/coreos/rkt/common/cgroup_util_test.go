// Copyright 2015 The rkt Authors
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

//+build linux

package common

import (
	"testing"
)

func TestCgEscape(t *testing.T) {
	tests := []struct {
		input  string
		output string
	}{
		{
			input:  "",
			output: "",
		},
		{
			input:  "_abc",
			output: "__abc",
		},
		{
			input:  ".abc",
			output: "_.abc",
		},
		{
			input:  "notify_on_release",
			output: "_notify_on_release",
		},
		{
			input:  "tasks",
			output: "_tasks",
		},
		{
			input:  "cgroup.abc.slice",
			output: "_cgroup.abc.slice",
		},
		{
			input:  "abc.mount",
			output: "abc.mount",
		},
		{
			input:  "a/bc.mount",
			output: "_a/bc.mount",
		},
	}

	for i, tt := range tests {
		o := cgEscape(tt.input)
		if o != tt.output {
			t.Errorf("#%d: expected `%v` got `%v`", i, tt.output, o)
		}
	}
}

func TestSliceToPath(t *testing.T) {
	tests := []struct {
		input  string
		output string
		werr   bool
	}{
		{
			input:  "",
			output: "",
			werr:   true,
		},
		{
			input:  "abc.service",
			output: "",
			werr:   true,
		},
		{
			input:  "ab/c.slice",
			output: "",
			werr:   true,
		},
		{
			input:  "-abc.slice",
			output: "",
			werr:   true,
		},
		{
			input:  "ab--c.slice",
			output: "",
			werr:   true,
		},
		{
			input:  "abc-.slice",
			output: "",
			werr:   true,
		},
		{
			input:  "abc.slice",
			output: "abc.slice",
			werr:   false,
		},
		{
			input:  "foo-bar.slice",
			output: "foo.slice/foo-bar.slice",
			werr:   false,
		},
		{
			input:  "foo-bar-baz.slice",
			output: "foo.slice/foo-bar.slice/foo-bar-baz.slice",
			werr:   false,
		},
	}

	for i, tt := range tests {
		o, err := SliceToPath(tt.input)
		gerr := (err != nil)
		if o != tt.output {
			t.Errorf("#%d: expected `%v` got `%v`", i, tt.output, o)
		}
		if gerr != tt.werr {
			t.Errorf("#%d: gerr=%t, want %t (err=%v)", i, gerr, tt.werr, err)
		}
	}
}
