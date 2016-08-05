/*
Copyright 2015 The Kubernetes Authors.

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

package net

import (
	"testing"
)

func TestSplitSchemeNamePort(t *testing.T) {
	table := []struct {
		in                 string
		name, port, scheme string
		valid              bool
		normalized         bool
	}{
		{
			in:    "aoeu:asdf",
			name:  "aoeu",
			port:  "asdf",
			valid: true,
		}, {
			in:     "http:aoeu:asdf",
			scheme: "http",
			name:   "aoeu",
			port:   "asdf",
			valid:  true,
		}, {
			in:         "https:aoeu:",
			scheme:     "https",
			name:       "aoeu",
			port:       "",
			valid:      true,
			normalized: false,
		}, {
			in:     "https:aoeu:asdf",
			scheme: "https",
			name:   "aoeu",
			port:   "asdf",
			valid:  true,
		}, {
			in:         "aoeu:",
			name:       "aoeu",
			valid:      true,
			normalized: false,
		}, {
			in:    ":asdf",
			valid: false,
		}, {
			in:    "aoeu:asdf:htns",
			valid: false,
		}, {
			in:    "aoeu",
			name:  "aoeu",
			valid: true,
		}, {
			in:    "",
			valid: false,
		},
	}

	for _, item := range table {
		scheme, name, port, valid := SplitSchemeNamePort(item.in)
		if e, a := item.scheme, scheme; e != a {
			t.Errorf("%q: Wanted %q, got %q", item.in, e, a)
		}
		if e, a := item.name, name; e != a {
			t.Errorf("%q: Wanted %q, got %q", item.in, e, a)
		}
		if e, a := item.port, port; e != a {
			t.Errorf("%q: Wanted %q, got %q", item.in, e, a)
		}
		if e, a := item.valid, valid; e != a {
			t.Errorf("%q: Wanted %t, got %t", item.in, e, a)
		}

		// Make sure valid items round trip through JoinSchemeNamePort
		if item.valid {
			out := JoinSchemeNamePort(scheme, name, port)
			if item.normalized && out != item.in {
				t.Errorf("%q: Wanted %s, got %s", item.in, item.in, out)
			}
			scheme, name, port, valid := SplitSchemeNamePort(out)
			if e, a := item.scheme, scheme; e != a {
				t.Errorf("%q: Wanted %q, got %q", item.in, e, a)
			}
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
}
