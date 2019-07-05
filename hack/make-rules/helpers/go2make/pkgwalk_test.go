/*
Copyright 2017 The Kubernetes Authors.

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

package main

import (
	"path"
	"reflect"
	"sort"
	"testing"
)

func Test_WalkPkg(t *testing.T) {
	testCases := []struct {
		pkg      string
		fail     bool
		expected []string
	}{
		{
			pkg:  "./testdata/nonexistent-dir",
			fail: true,
		},
		{
			pkg:      "./testdata/dir-without-gofiles",
			expected: []string{"./testdata/dir-without-gofiles"},
		},
		{
			pkg:      "./testdata/dir-with-gofiles",
			expected: []string{"./testdata/dir-with-gofiles", "./testdata/dir-with-gofiles/subdir"},
		},
	}

	for i, tc := range testCases {
		visited := []string{}
		err := WalkPkg(tc.pkg, func(imp, abs string) error {
			if _, base := path.Split(imp); base == "skipme" {
				return ErrSkipPkg
			}
			visited = append(visited, imp)
			return nil
		})
		if err != nil && tc.fail {
			continue
		}
		if err != nil {
			t.Errorf("[%d] unexpected error: %v", i, err)
			continue
		}
		if tc.fail {
			t.Errorf("[%d] expected error", i)
			continue
		}
		if !reflect.DeepEqual(visited, tc.expected) {
			t.Errorf("[%d] unexpected results: %v", i, visited)
		}
	}
}
func Test_findPackage(t *testing.T) {
	testCases := []struct {
		pkg  string
		fail bool
	}{
		{
			pkg:  "./testdata/nonexistent-dir",
			fail: true,
		},
		{
			pkg: "./testdata/dir-without-gofiles",
		},
		{
			pkg: "./testdata/dir-with-gofiles",
		},
	}

	for i, tc := range testCases {
		_, err := findPackage(tc.pkg)
		if err != nil && tc.fail {
			continue
		}
		if err != nil {
			t.Errorf("[%d] unexpected error: %v", i, err)
			continue
		}
		if tc.fail {
			t.Errorf("[%d] expected error", i)
			continue
		}
	}
}

func Test_readDirInfos(t *testing.T) {
	testCases := []struct {
		dir      string
		fail     bool
		expected map[string]bool
	}{
		{
			dir:  "./testdata/nonexistent-dir",
			fail: true,
		},
		{
			dir:      "./testdata/dir-without-gofiles",
			expected: map[string]bool{"README": true},
		},
		{
			dir: "./testdata/dir-with-gofiles",
			expected: map[string]bool{
				"README":      true,
				"foo.go":      true,
				"bar.go":      true,
				"subdir":      true,
				"testdata":    true,
				"_underscore": true,
				".dot":        true,
				"skipme":      true,
			},
		},
	}

	for i, tc := range testCases {
		infos, err := readDirInfos(tc.dir)
		if err != nil && tc.fail {
			continue
		}
		if err != nil {
			t.Errorf("[%d] unexpected error: %v", i, err)
			continue
		}
		if tc.fail {
			t.Errorf("[%d] expected error", i)
			continue
		}
		result := make([]string, len(infos))
		sorted := make([]string, len(infos))
		for i, inf := range infos {
			result[i] = inf.Name()
			sorted[i] = inf.Name()
		}
		sort.Strings(sorted)
		if !reflect.DeepEqual(result, sorted) {
			t.Errorf("[%d] result was not sorted: %v", i, result)
		}
		for _, r := range result {
			if !tc.expected[r] {
				t.Errorf("[%d] got unexpected result: %s", i, r)
			} else {
				delete(tc.expected, r)
			}
		}
		for r := range tc.expected {
			t.Errorf("[%d] missing expected result: %s", i, r)
		}
	}
}

func Test_readDirNames(t *testing.T) {
	testCases := []struct {
		dir      string
		fail     bool
		expected map[string]bool
	}{
		{
			dir:  "./testdata/nonexistent-dir",
			fail: true,
		},
		{
			dir:      "./testdata/dir-without-gofiles",
			expected: map[string]bool{"README": true},
		},
		{
			dir: "./testdata/dir-with-gofiles",
			expected: map[string]bool{
				"README":      true,
				"foo.go":      true,
				"bar.go":      true,
				"subdir":      true,
				"testdata":    true,
				"_underscore": true,
				".dot":        true,
				"skipme":      true,
			},
		},
	}

	for i, tc := range testCases {
		result, err := readDirNames(tc.dir)
		if err != nil && tc.fail {
			continue
		}
		if err != nil {
			t.Errorf("[%d] unexpected error: %v", i, err)
			continue
		}
		if tc.fail {
			t.Errorf("[%d] expected error", i)
			continue
		}
		for _, r := range result {
			if !tc.expected[r] {
				t.Errorf("[%d] got unexpected result: %s", i, r)
			} else {
				delete(tc.expected, r)
			}
		}
		for r := range tc.expected {
			t.Errorf("[%d] missing expected result: %s", i, r)
		}
	}
}
