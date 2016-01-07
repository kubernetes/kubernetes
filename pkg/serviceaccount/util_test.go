/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package serviceaccount

import "testing"

func TestMakeUsername(t *testing.T) {

	testCases := map[string]struct {
		Namespace   string
		Name        string
		ExpectedErr bool
	}{
		"valid": {
			Namespace:   "foo",
			Name:        "bar",
			ExpectedErr: false,
		},
		"empty": {
			ExpectedErr: true,
		},
		"empty namespace": {
			Namespace:   "",
			Name:        "foo",
			ExpectedErr: true,
		},
		"empty name": {
			Namespace:   "foo",
			Name:        "",
			ExpectedErr: true,
		},
		"extra segments": {
			Namespace:   "foo",
			Name:        "bar:baz",
			ExpectedErr: true,
		},
		"invalid chars in namespace": {
			Namespace:   "foo ",
			Name:        "bar",
			ExpectedErr: true,
		},
		"invalid chars in name": {
			Namespace:   "foo",
			Name:        "bar ",
			ExpectedErr: true,
		},
	}

	for k, tc := range testCases {
		username := MakeUsername(tc.Namespace, tc.Name)

		namespace, name, err := SplitUsername(username)
		if (err != nil) != tc.ExpectedErr {
			t.Errorf("%s: Expected error=%v, got %v", k, tc.ExpectedErr, err)
			continue
		}
		if err != nil {
			continue
		}

		if namespace != tc.Namespace {
			t.Errorf("%s: Expected namespace %q, got %q", k, tc.Namespace, namespace)
		}
		if name != tc.Name {
			t.Errorf("%s: Expected name %q, got %q", k, tc.Name, name)
		}
	}
}
