/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package util

import (
	"testing"
)

func TestSplitPort(t *testing.T) {
	table := []struct {
		in         string
		name, port string
		valid      bool
	}{
		{
			in:    "aoeu:asdf",
			name:  "aoeu",
			port:  "asdf",
			valid: true,
		}, {
			in:    "aoeu:",
			name:  "aoeu",
			valid: true,
		}, {
			in:   ":asdf",
			name: "",
			port: "asdf",
		}, {
			in: "aoeu:asdf:htns",
		}, {
			in:    "aoeu",
			name:  "aoeu",
			valid: true,
		}, {
			in: "",
		},
	}

	for _, item := range table {
		name, port, valid := SplitPort(item.in)
		if e, a := item.name, name; e != a {
			t.Errorf("%q: Wanted %q, got %q", item.in, e, a)
		}
		if e, a := item.port, port; e != a {
			t.Errorf("%q: Wanted %q, got %q", item.in, e, a)
		}
		if e, a := item.valid, valid; e != a {
			t.Errorf("%q: Wanted %t, got %t", item.in, e, a)
		}
	}
}
