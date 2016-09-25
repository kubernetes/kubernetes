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

package storage

import (
	"math/rand"
	"sync"
	"testing"
)

func TestEtcdParseWatchResourceVersion(t *testing.T) {
	testCases := []struct {
		Version       string
		ExpectVersion uint64
		Err           bool
	}{
		{Version: "", ExpectVersion: 0},
		{Version: "a", Err: true},
		{Version: " ", Err: true},
		{Version: "1", ExpectVersion: 1},
		{Version: "10", ExpectVersion: 10},
	}
	for _, testCase := range testCases {
		version, err := ParseWatchResourceVersion(testCase.Version)
		switch {
		case testCase.Err:
			if err == nil {
				t.Errorf("%s: unexpected non-error", testCase.Version)
				continue
			}
			if !IsInvalidError(err) {
				t.Errorf("%s: unexpected error: %v", testCase.Version, err)
				continue
			}
		case !testCase.Err && err != nil:
			t.Errorf("%s: unexpected error: %v", testCase.Version, err)
			continue
		}
		if version != testCase.ExpectVersion {
			t.Errorf("%s: expected version %d but was %d", testCase.Version, testCase.ExpectVersion, version)
		}
	}
}

func TestHasPathPrefix(t *testing.T) {
	validTestcases := []struct {
		s      string
		prefix string
	}{
		// Exact matches
		{"", ""},
		{"a", "a"},
		{"a/", "a/"},
		{"a/../", "a/../"},

		// Path prefix matches
		{"a/b", "a"},
		{"a/b", "a/"},
		{"中文/", "中文"},
	}
	for i, tc := range validTestcases {
		if !hasPathPrefix(tc.s, tc.prefix) {
			t.Errorf(`%d: Expected hasPathPrefix("%s","%s") to be true`, i, tc.s, tc.prefix)
		}
	}

	invalidTestcases := []struct {
		s      string
		prefix string
	}{
		// Mismatch
		{"a", "b"},

		// Dir requirement
		{"a", "a/"},

		// Prefix mismatch
		{"ns2", "ns"},
		{"ns2", "ns/"},
		{"中文文", "中文"},

		// Ensure no normalization is applied
		{"a/c/../b/", "a/b/"},
		{"a/", "a/b/.."},
	}
	for i, tc := range invalidTestcases {
		if hasPathPrefix(tc.s, tc.prefix) {
			t.Errorf(`%d: Expected hasPathPrefix("%s","%s") to be false`, i, tc.s, tc.prefix)
		}
	}
}

func TestHighWaterMark(t *testing.T) {
	var h HighWaterMark

	for i := int64(10); i < 20; i++ {
		if !h.Update(i) {
			t.Errorf("unexpected false for %v", i)
		}
		if h.Update(i - 1) {
			t.Errorf("unexpected true for %v", i-1)
		}
	}

	m := int64(0)
	wg := sync.WaitGroup{}
	for i := 0; i < 300; i++ {
		wg.Add(1)
		v := rand.Int63()
		go func(v int64) {
			defer wg.Done()
			h.Update(v)
		}(v)
		if v > m {
			m = v
		}
	}
	wg.Wait()
	if m != int64(h) {
		t.Errorf("unexpected value, wanted %v, got %v", m, int64(h))
	}
}
