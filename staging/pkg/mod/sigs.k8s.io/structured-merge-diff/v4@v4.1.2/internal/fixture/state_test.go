/*
Copyright 2018 The Kubernetes Authors.

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

package fixture

import (
	"fmt"
	"testing"

	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

func TestFixTabs(t *testing.T) {
	cases := []struct {
		in, out     typed.YAMLObject
		shouldPanic bool
	}{{
		in:  "a\n  b\n",
		out: "a\n  b\n",
	}, {
		in:  "\t\ta\n\t\t\tb\n",
		out: "a\n\tb\n",
	}, {
		in:  "\n\t\ta\n\t\tb\n",
		out: "a\nb\n",
	}, {
		in:  "\n\t\ta\n\t\t\tb\n\t",
		out: "a\n\tb\n",
	}, {
		in:  "\t\ta\n\t\t  b\n",
		out: "a\n  b\n",
	}, {
		in:          "\t\ta\n\tb\n",
		shouldPanic: true,
	}}

	for i := range cases {
		tt := cases[i]
		t.Run(fmt.Sprintf("%v-%v", i, []byte(tt.in)), func(t *testing.T) {
			if tt.shouldPanic {
				defer func() {
					if x := recover(); x == nil {
						t.Errorf("expected a panic, but didn't get one")
					}
				}()
			}
			got := FixTabsOrDie(tt.in)
			if e, a := tt.out, got; e != a {
				t.Errorf("mismatch\n   got %v\nwanted %v", []byte(a), []byte(e))
			}
		})
	}
}
