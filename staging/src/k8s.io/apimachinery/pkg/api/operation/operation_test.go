/*
Copyright The Kubernetes Authors.

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

package operation

import "testing"

type fakeOptionGetter map[string]bool

func (f fakeOptionGetter) Get(option string) (bool, bool) {
	enabled, defined := f[option]
	return enabled, defined
}

func TestHasOption(t *testing.T) {
	testCases := []struct {
		name            string
		op              Operation
		option          string
		expectedEnabled bool
		expectedDefined bool
	}{{
		name:            "nil map and nil getter",
		op:              Operation{},
		option:          "foo",
		expectedEnabled: false,
		expectedDefined: false,
	}, {
		name: "found enabled in map with nil getter",
		op: Operation{
			Options: map[string]bool{"foo": true},
		},
		option:          "foo",
		expectedEnabled: true,
		expectedDefined: true,
	}, {
		name: "found disabled in map with nil getter",
		op: Operation{
			Options: map[string]bool{"foo": false},
		},
		option:          "foo",
		expectedEnabled: false,
		expectedDefined: true,
	}, {
		name: "not found in map with nil getter",
		op: Operation{
			Options: map[string]bool{"other": true},
		},
		option:          "foo",
		expectedEnabled: false,
		expectedDefined: false,
	}, {
		name: "nil map falls back to getter (enabled)",
		op: Operation{
			OptionGetter: fakeOptionGetter{"foo": true},
		},
		option:          "foo",
		expectedEnabled: true,
		expectedDefined: true,
	}, {
		name: "nil map falls back to getter (disabled)",
		op: Operation{
			OptionGetter: fakeOptionGetter{"foo": false},
		},
		option:          "foo",
		expectedEnabled: false,
		expectedDefined: true,
	}, {
		name: "nil map falls back to getter (undefined)",
		op: Operation{
			OptionGetter: fakeOptionGetter{"other": true},
		},
		option:          "foo",
		expectedEnabled: false,
		expectedDefined: false,
	}, {
		name: "not found in map falls back to getter",
		op: Operation{
			Options:      map[string]bool{"other": false},
			OptionGetter: fakeOptionGetter{"foo": true},
		},
		option:          "foo",
		expectedEnabled: true,
		expectedDefined: true,
	}, {
		name: "map takes precedence over getter (map true, getter false)",
		op: Operation{
			Options:      map[string]bool{"foo": true},
			OptionGetter: fakeOptionGetter{"foo": false},
		},
		option:          "foo",
		expectedEnabled: true,
		expectedDefined: true,
	}, {
		name: "map takes precedence over getter (map false, getter true)",
		op: Operation{
			Options:      map[string]bool{"foo": false},
			OptionGetter: fakeOptionGetter{"foo": true},
		},
		option:          "foo",
		expectedEnabled: false,
		expectedDefined: true,
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			enabled, defined := tc.op.HasOption(tc.option)
			if want, got := tc.expectedEnabled, enabled; got != want {
				t.Errorf("enabled: expected %v, got %v", want, got)
			}
			if want, got := tc.expectedDefined, defined; got != want {
				t.Errorf("defined: expected %v, got %v", want, got)
			}
		})
	}
}
