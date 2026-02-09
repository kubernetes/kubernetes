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

package util

import "testing"

func TestParseFileSource(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		key      string
		filepath string
		err      bool
	}{
		{
			name:     "success 1",
			input:    "boo=zoo",
			key:      "boo",
			filepath: "zoo",
			err:      false,
		},
		{
			name:     "success 2",
			input:    "boo=/path/to/zoo",
			key:      "boo",
			filepath: "/path/to/zoo",
			err:      false,
		},
		{
			name:     "success 3",
			input:    "boo-2=/1/2/3/4/5/zab.txt",
			key:      "boo-2",
			filepath: "/1/2/3/4/5/zab.txt",
			err:      false,
		},
		{
			name:     "success 4",
			input:    "boo-=this/seems/weird.txt",
			key:      "boo-",
			filepath: "this/seems/weird.txt",
			err:      false,
		},
		{
			name:     "success 5",
			input:    "-key=some/path",
			key:      "-key",
			filepath: "some/path",
			err:      false,
		},
		{
			name:  "invalid 1",
			input: "key==some/path",
			err:   true,
		},
		{
			name:  "invalid 2",
			input: "=key=some/path",
			err:   true,
		},
		{
			name:  "invalid 3",
			input: "==key=/some/other/path",
			err:   true,
		},
		{
			name:  "invalid 4",
			input: "=key",
			err:   true,
		},
		{
			name:  "invalid 5",
			input: "key=",
			err:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			key, filepath, err := ParseFileSource(tt.input)
			if err != nil {
				if tt.err {
					return
				}

				t.Errorf("%v: unexpected error: %v", tt.name, err)
				return
			}

			if tt.err {
				t.Errorf("%v: unexpected success", tt.name)
				return
			}

			if e, a := tt.key, key; e != a {
				t.Errorf("%v: expected key %v; got %v", tt.name, e, a)
				return
			}

			if e, a := tt.filepath, filepath; e != a {
				t.Errorf("%v: expected filepath %v; got %v", tt.name, e, a)
			}
		})
	}
}

func TestParseLiteralSource(t *testing.T) {
	tests := []struct {
		name  string
		input string
		key   string
		value string
		err   bool
	}{
		{
			name:  "success 1",
			input: "key=value",
			key:   "key",
			value: "value",
			err:   false,
		},
		{
			name:  "success 2",
			input: "key=value/with/slashes",
			key:   "key",
			value: "value/with/slashes",
			err:   false,
		},
		{
			name:  "err 1",
			input: "key==value",
			key:   "key",
			value: "=value",
			err:   false,
		},
		{
			name:  "err 2",
			input: "key=value=",
			key:   "key",
			value: "value=",
			err:   false,
		},
		{
			name:  "err 3",
			input: "key2=value==",
			key:   "key2",
			value: "value==",
			err:   false,
		},
		{
			name:  "err 4",
			input: "==key",
			err:   true,
		},
		{
			name:  "err 5",
			input: "=key=",
			err:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			key, value, err := ParseLiteralSource(tt.input)
			if err != nil {
				if tt.err {
					return
				}

				t.Errorf("%v: unexpected error: %v", tt.name, err)
				return
			}

			if tt.err {
				t.Errorf("%v: unexpected success", tt.name)
				return
			}

			if e, a := tt.key, key; e != a {
				t.Errorf("%v: expected key %v; got %v", tt.name, e, a)
				return
			}

			if e, a := tt.value, value; e != a {
				t.Errorf("%v: expected value %v; got %v", tt.name, e, a)
			}
		})
	}
}
