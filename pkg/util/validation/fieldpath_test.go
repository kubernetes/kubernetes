/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package validation

import "testing"

func TestFieldPath(t *testing.T) {
	testCases := []struct {
		op       func(*FieldPath) *FieldPath
		expected string
	}{
		{
			func(fp *FieldPath) *FieldPath { return fp },
			"root",
		},
		{
			func(fp *FieldPath) *FieldPath { return fp.Child("first") },
			"root.first",
		},
		{
			func(fp *FieldPath) *FieldPath { return fp.Child("second") },
			"root.first.second",
		},
		{
			func(fp *FieldPath) *FieldPath { return fp.Index(0) },
			"root.first.second[0]",
		},
		{
			func(fp *FieldPath) *FieldPath { return fp.Child("third") },
			"root.first.second[0].third",
		},
		{
			func(fp *FieldPath) *FieldPath { return fp.Index(93) },
			"root.first.second[0].third[93]",
		},
		{
			func(fp *FieldPath) *FieldPath { return fp.parent },
			"root.first.second[0].third",
		},
		{
			func(fp *FieldPath) *FieldPath { return fp.parent },
			"root.first.second[0]",
		},
		{
			func(fp *FieldPath) *FieldPath { return fp.Key("key") },
			"root.first.second[0][key]",
		},
	}

	root := NewFieldPath("root")
	fp := root
	for i, tc := range testCases {
		fp = tc.op(fp)
		if fp.String() != tc.expected {
			t.Errorf("[%d] Expected %q, got %q", i, tc.expected, fp.String())
		}
		if fp.Root() != root {
			t.Errorf("[%d] Wrong root: %#v", i, fp.Root())
		}
	}
}

func TestFieldPathMultiArg(t *testing.T) {
	testCases := []struct {
		op       func(*FieldPath) *FieldPath
		expected string
	}{
		{
			func(fp *FieldPath) *FieldPath { return fp },
			"root.first",
		},
		{
			func(fp *FieldPath) *FieldPath { return fp.Child("second", "third") },
			"root.first.second.third",
		},
		{
			func(fp *FieldPath) *FieldPath { return fp.Index(0) },
			"root.first.second.third[0]",
		},
		{
			func(fp *FieldPath) *FieldPath { return fp.parent },
			"root.first.second.third",
		},
		{
			func(fp *FieldPath) *FieldPath { return fp.parent },
			"root.first.second",
		},
		{
			func(fp *FieldPath) *FieldPath { return fp.parent },
			"root.first",
		},
		{
			func(fp *FieldPath) *FieldPath { return fp.parent },
			"root",
		},
	}

	root := NewFieldPath("root", "first")
	fp := root
	for i, tc := range testCases {
		fp = tc.op(fp)
		if fp.String() != tc.expected {
			t.Errorf("[%d] Expected %q, got %q", i, tc.expected, fp.String())
		}
		if fp.Root() != root.Root() {
			t.Errorf("[%d] Wrong root: %#v", i, fp.Root())
		}
	}
}
