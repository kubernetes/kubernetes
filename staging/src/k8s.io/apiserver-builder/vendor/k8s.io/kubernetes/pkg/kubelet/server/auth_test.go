/*
Copyright 2016 The Kubernetes Authors.

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

package server

import "testing"

func TestIsSubPath(t *testing.T) {
	testcases := map[string]struct {
		subpath  string
		path     string
		expected bool
	}{
		"empty": {subpath: "", path: "", expected: true},

		"match 1": {subpath: "foo", path: "foo", expected: true},
		"match 2": {subpath: "/foo", path: "/foo", expected: true},
		"match 3": {subpath: "/foo/", path: "/foo/", expected: true},
		"match 4": {subpath: "/foo/bar", path: "/foo/bar", expected: true},

		"subpath of root 1": {subpath: "/foo", path: "/", expected: true},
		"subpath of root 2": {subpath: "/foo/", path: "/", expected: true},
		"subpath of root 3": {subpath: "/foo/bar", path: "/", expected: true},

		"subpath of path 1": {subpath: "/foo", path: "/foo", expected: true},
		"subpath of path 2": {subpath: "/foo/", path: "/foo", expected: true},
		"subpath of path 3": {subpath: "/foo/bar", path: "/foo", expected: true},

		"mismatch 1": {subpath: "/foo", path: "/bar", expected: false},
		"mismatch 2": {subpath: "/foo", path: "/foobar", expected: false},
		"mismatch 3": {subpath: "/foobar", path: "/foo", expected: false},
	}

	for k, tc := range testcases {
		result := isSubpath(tc.subpath, tc.path)
		if result != tc.expected {
			t.Errorf("%s: expected %v, got %v", k, tc.expected, result)
		}
	}
}
