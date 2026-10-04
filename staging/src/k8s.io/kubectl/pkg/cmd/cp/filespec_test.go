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

package cp

import "testing"

func TestIsRelative(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		base     string
		expected bool
	}{
		{
			name:     "test single path shortcut prefix",
			input:    "../foo/bar",
			base:     "../",
			expected: true,
		},
		{
			name:     "test single path shortcut prefix (Windows)",
			input:    `..\foo\bar`,
			base:     `..\`,
			expected: true,
		},
		{
			name:     "test multiple path shortcuts",
			input:    "../../foo/bar",
			base:     "../",
			expected: false,
		},
		{
			name:     "test multiple path shortcuts (Windows)",
			input:    `..\..\foo\bar`,
			base:     `..\`,
			expected: false,
		},
		{
			name:     "test multiple path shortcuts with absolute path",
			input:    "/tmp/one/two/../../foo/bar",
			base:     "/",
			expected: true,
		},
		{
			name:     "test multiple path shortcuts with absolute path (Windows)",
			input:    `\tmp\one\two\..\..\foo\bar`,
			base:     `\`,
			expected: true,
		},
		{
			name:     "test multiple path shortcuts with no named directory",
			input:    "../../",
			base:     "../",
			expected: false,
		},
		{
			name:     "test multiple path shortcuts with no named directory (Windows)",
			input:    `..\..\`,
			base:     `..\`,
			expected: false,
		},
		{
			name:     "test multiple path shortcuts with no named directory and no trailing slash",
			input:    "../..",
			base:     "../",
			expected: false,
		},
		{
			name:     "test multiple path shortcuts with no named directory and no trailing slash (Windows)",
			input:    `..\..`,
			base:     `..\`,
			expected: false,
		},
		{
			name:     "test multiple path shortcuts with absolute path and filename containing leading dots",
			input:    "/tmp/one/two/../../foo/..bar",
			base:     "/",
			expected: true,
		},
		{
			name:     "test multiple path shortcuts with absolute path and filename containing leading dots (Windows)",
			input:    `\tmp\one\two\..\..\foo\..bar`,
			base:     `\`,
			expected: true,
		},
		{
			name:     "test multiple path shortcuts with no named directory and filename containing leading dots",
			input:    "../...foo",
			base:     "../",
			expected: true,
		},
		{
			name:     "test multiple path shortcuts with no named directory and filename containing leading dots (Windows)",
			input:    `..\...foo`,
			base:     `..\`,
			expected: true,
		},
		{
			name:     "test filename containing leading dots",
			input:    "/...foo",
			base:     "/",
			expected: true,
		},
		{
			name:     "test root directory",
			input:    "/",
			base:     "/",
			expected: true,
		},
		{
			name:     "test root directory (Windows)",
			input:    `\`,
			base:     `\`,
			expected: true,
		},
		{
			name:     "test basic relative path",
			input:    "/a/b/c",
			base:     "/a",
			expected: true,
		},
		{
			name:     "test basic relative path (Windows)",
			input:    `\a\b\c`,
			base:     `\a`,
			expected: true,
		},
		{
			name:     "test basic non relative path",
			input:    "/a/b/c",
			base:     "/f",
			expected: false,
		},
		{
			name:     "test basic non relative path (Windows)",
			input:    `\a\b\c`,
			base:     `\f`,
			expected: false,
		},
		{
			name:     "test mix of Windows base and linux input",
			input:    `\a\b\c`,
			base:     `/f`,
			expected: false,
		},
	}

	for _, test := range tests {
		result := isRelative(newLocalPath(test.base), newLocalPath(test.input))
		if result != test.expected {
			t.Errorf("%s: %t, saw: %t", test.name, test.expected, result)
		}
	}
}
