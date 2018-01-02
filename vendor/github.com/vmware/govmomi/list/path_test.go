/*
Copyright (c) 2014 VMware, Inc. All Rights Reserved.

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

package list

import (
	"reflect"
	"testing"
)

func TestToParts(t *testing.T) {
	tests := []struct {
		In  string
		Out []string
	}{
		{
			In:  "/",
			Out: []string{},
		},
		{
			In:  "/foo",
			Out: []string{"foo"},
		},
		{
			In:  "/foo/..",
			Out: []string{},
		},
		{
			In:  "/./foo",
			Out: []string{"foo"},
		},
		{
			In:  "/../foo",
			Out: []string{"foo"},
		},
		{
			In:  "/foo/bar",
			Out: []string{"foo", "bar"},
		},
		{
			In:  "/foo/bar/..",
			Out: []string{"foo"},
		},
		{
			In:  "",
			Out: []string{"."},
		},
		{
			In:  ".",
			Out: []string{"."},
		},
		{
			In:  "foo",
			Out: []string{".", "foo"},
		},
		{
			In:  "foo/..",
			Out: []string{"."},
		},
		{
			In:  "./foo",
			Out: []string{".", "foo"},
		},
		{
			In:  "../foo", // Special case...
			Out: []string{"..", "foo"},
		},
		{
			In:  "foo/bar/..",
			Out: []string{".", "foo"},
		},
	}

	for _, test := range tests {
		out := ToParts(test.In)
		if !reflect.DeepEqual(test.Out, out) {
			t.Errorf("Expected %s to return: %#v, actual: %#v", test.In, test.Out, out)
		}
	}
}
