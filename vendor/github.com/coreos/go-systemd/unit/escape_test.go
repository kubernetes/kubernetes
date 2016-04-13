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

package unit

import (
	"testing"
)

func TestUnitNameEscape(t *testing.T) {
	tests := []struct {
		in     string
		out    string
		isPath bool
	}{
		// turn empty string path into escaped /
		{
			in:     "",
			out:    "-",
			isPath: true,
		},
		// turn redundant ////s into single escaped /
		{
			in:     "/////////",
			out:    "-",
			isPath: true,
		},
		// remove all redundant ////s
		{
			in:     "///foo////bar/////tail//////",
			out:    "foo-bar-tail",
			isPath: true,
		},
		// leave empty string empty
		{
			in:     "",
			out:    "",
			isPath: false,
		},
		// escape leading dot
		{
			in:     ".",
			out:    `\x2e`,
			isPath: true,
		},
		// escape leading dot
		{
			in:     "/.",
			out:    `\x2e`,
			isPath: true,
		},
		// escape leading dot
		{
			in:     "/////////.",
			out:    `\x2e`,
			isPath: true,
		},
		// escape leading dot
		{
			in:     "/////////.///////////////",
			out:    `\x2e`,
			isPath: true,
		},
		// escape leading dot
		{
			in:     ".....",
			out:    `\x2e....`,
			isPath: true,
		},
		// escape leading dot
		{
			in:     "/.foo/.bar",
			out:    `\x2efoo-.bar`,
			isPath: true,
		},
		// escape leading dot
		{
			in:     ".foo/.bar",
			out:    `\x2efoo-.bar`,
			isPath: true,
		},
		// escape leading dot
		{
			in:     ".foo/.bar",
			out:    `\x2efoo-.bar`,
			isPath: false,
		},
		// escape disallowed
		{
			in:     `///..\-!#??///`,
			out:    `---..\x5c\x2d\x21\x23\x3f\x3f---`,
			isPath: false,
		},
		// escape disallowed
		{
			in:     `///..\-!#??///`,
			out:    `\x2e.\x5c\x2d\x21\x23\x3f\x3f`,
			isPath: true,
		},
		// escape real-world example
		{
			in:     `user-cloudinit@/var/lib/coreos/vagrant/vagrantfile-user-data.service`,
			out:    `user\x2dcloudinit\x40-var-lib-coreos-vagrant-vagrantfile\x2duser\x2ddata.service`,
			isPath: false,
		},
	}

	for i, tt := range tests {
		var s string
		if tt.isPath {
			s = UnitNamePathEscape(tt.in)
		} else {
			s = UnitNameEscape(tt.in)
		}
		if s != tt.out {
			t.Errorf("case %d: failed escaping %v isPath: %v - expected %v, got %v", i, tt.in, tt.isPath, tt.out, s)
		}
	}
}

func TestUnitNameUnescape(t *testing.T) {
	tests := []struct {
		in     string
		out    string
		isPath bool
	}{
		// turn empty string path into /
		{
			in:     "",
			out:    "/",
			isPath: true,
		},
		// leave empty string empty
		{
			in:     "",
			out:    "",
			isPath: false,
		},
		// turn ////s into
		{
			in:     "---------",
			out:    "/////////",
			isPath: true,
		},
		// unescape hex
		{
			in:     `---..\x5c\x2d\x21\x23\x3f\x3f---`,
			out:    `///..\-!#??///`,
			isPath: false,
		},
		// unescape hex
		{
			in:     `\x2e.\x5c\x2d\x21\x23\x3f\x3f`,
			out:    `/..\-!#??`,
			isPath: true,
		},
		// unescape hex, retain invalids
		{
			in:     `\x2e.\x5c\x2d\xaZ\x.o\x21\x23\x3f\x3f`,
			out:    `/..\-\xaZ\x.o!#??`,
			isPath: true,
		},
		// unescape hex, retain invalids, partial tail
		{
			in:     `\x2e.\x5c\x\x2d\xaZ\x.o\x21\x23\x3f\x3f\x3`,
			out:    `/..\\x-\xaZ\x.o!#??\x3`,
			isPath: true,
		},
		// unescape hex, retain invalids, partial tail
		{
			in:     `\x2e.\x5c\x\x2d\xaZ\x.o\x21\x23\x3f\x3f\x`,
			out:    `/..\\x-\xaZ\x.o!#??\x`,
			isPath: true,
		},
		// unescape hex, retain invalids, partial tail
		{
			in:     `\x2e.\x5c\x\x2d\xaZ\x.o\x21\x23\x3f\x3f\`,
			out:    `/..\\x-\xaZ\x.o!#??\`,
			isPath: true,
		},
		// unescape real-world example
		{
			in:     `user\x2dcloudinit\x40-var-lib-coreos-vagrant-vagrantfile\x2duser\x2ddata.service`,
			out:    `user-cloudinit@/var/lib/coreos/vagrant/vagrantfile-user-data.service`,
			isPath: false,
		},
	}

	for i, tt := range tests {
		var s string
		if tt.isPath {
			s = UnitNamePathUnescape(tt.in)
		} else {
			s = UnitNameUnescape(tt.in)
		}
		if s != tt.out {
			t.Errorf("case %d: failed unescaping %v isPath: %v - expected %v, got %v", i, tt.in, tt.isPath, tt.out, s)
		}
	}
}
