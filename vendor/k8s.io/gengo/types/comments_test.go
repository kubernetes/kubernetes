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

package types

import (
	"reflect"
	"strings"
	"testing"
)

func TestExtractCommentTags(t *testing.T) {
	commentLines := []string{
		"Human comment that is ignored.",
		"+foo=value1",
		"+bar",
		"+foo=value2",
		"+baz=qux,zrb=true",
	}

	a := ExtractCommentTags("+", commentLines)
	e := map[string][]string{
		"foo": {"value1", "value2"},
		"bar": {""},
		"baz": {"qux,zrb=true"},
	}
	if !reflect.DeepEqual(e, a) {
		t.Errorf("Wanted %q, got %q", e, a)
	}
}

func TestExtractSingleBoolCommentTag(t *testing.T) {
	commentLines := []string{
		"Human comment that is ignored.",
		"+TRUE=true",
		"+FALSE=false",
		"+MULTI=true",
		"+MULTI=false",
		"+MULTI=multi",
		"+NOTBOOL=blue",
		"+EMPTY",
	}

	testCases := []struct {
		key string
		def bool
		exp bool
		err string // if set, ignore exp.
	}{
		{"TRUE", false, true, ""},
		{"FALSE", true, false, ""},
		{"MULTI", false, true, ""},
		{"NOTBOOL", false, true, "is not boolean"},
		{"EMPTY", false, true, "is not boolean"},
		{"ABSENT", true, true, ""},
		{"ABSENT", false, false, ""},
	}

	for i, tc := range testCases {
		v, err := ExtractSingleBoolCommentTag("+", tc.key, tc.def, commentLines)
		if err != nil && tc.err == "" {
			t.Errorf("[%d]: unexpected failure: %v", i, err)
		} else if err == nil && tc.err != "" {
			t.Errorf("[%d]: expected failure: %v", i, tc.err)
		} else if err != nil {
			if !strings.Contains(err.Error(), tc.err) {
				t.Errorf("[%d]: unexpected error: expected %q, got %q", i, tc.err, err)
			}
		} else if v != tc.exp {
			t.Errorf("[%d]: unexpected value: expected %t, got %t", i, tc.exp, v)
		}
	}
}
