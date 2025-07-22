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

package protobuf

import "testing"

func TestProtoSafePackage(t *testing.T) {
	tests := []struct {
		pkg      string
		expected string
	}{
		{
			pkg:      "foo",
			expected: "foo",
		},
		{
			pkg:      "foo/bar",
			expected: "foo.bar",
		},
		{
			pkg:      "foo/bar/baz",
			expected: "foo.bar.baz",
		},
		{
			pkg:      "foo/bar-baz/x/y-z/q",
			expected: "foo.bar_baz.x.y_z.q",
		},
	}

	for _, test := range tests {
		actual := protoSafePackage(test.pkg)
		if e, a := test.expected, actual; e != a {
			t.Errorf("%s: expected %s, got %s", test.pkg, e, a)
		}
	}
}
