/*
Copyright 2025 The Kubernetes Authors.

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
	"errors"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestExtractCommentTagsWithoutArguments(t *testing.T) {
	cases := []struct {
		tagNames    []string
		lines       []string
		expected    map[string][]string
		expectedErr error
	}{{
		tagNames: nil,
		lines: []string{
			"Human comment that is ignored.",
			"+foo=value1",
			"+bar",
			"+foo=value2",
			"+foo()=value3",
			"+baz=qux,zrb=true",
			"+bip=\"value3\"",
		},
		expected: map[string][]string{
			"foo": {"value1", "value2", "value3"},
			"bar": {""},
			"baz": {"qux,zrb=true"},
			"bip": {`"value3"`},
		},
		expectedErr: nil,
	}, {
		tagNames: nil,
		lines: []string{
			"+foo=value1",
			"+bar",
			"+foo=value2",
			"+foo()=value3",
			"+bip(arg)=",
			"+baz=qux,zrb=true",
		},
		expected:    nil,
		expectedErr: errors.New(`failed to parse tag bip(arg): expected no arguments, found "arg"`),
	}, {
		tagNames: []string{"foo", "bar"},
		lines: []string{
			"+foo=value1",
			"+bar",
			"+foo=value2",
			"+foo()=value3",
			"+bip(arg)=",
			"+baz=qux,zrb=true",
		},
		expected: map[string][]string{
			"foo": {"value1", "value2", "value3"},
			"bar": {""},
		},
		expectedErr: nil,
	}, {
		tagNames: []string{"lorem"},
		lines: []string{
			"+foo=value1",
			"+bar",
		},
		expected:    map[string][]string{},
		expectedErr: nil,
	}}

	for _, tc := range cases {
		values, err := ExtractCommentTagsWithoutArguments("+", tc.tagNames, tc.lines)

		if tc.expectedErr == nil && err != nil {
			t.Errorf("Failed to parse comments: %v", err)
		}
		if tc.expectedErr != nil && tc.expectedErr.Error() != err.Error() {
			t.Errorf("Expectng error %v, got %v", tc.expectedErr.Error(), err.Error())
		}
		if !reflect.DeepEqual(tc.expected, values) {
			t.Errorf("Wrong result:\n%v", cmp.Diff(tc.expected, values))
		}
	}
}
