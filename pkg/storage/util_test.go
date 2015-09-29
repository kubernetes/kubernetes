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

package storage

import (
	"testing"

	"k8s.io/kubernetes/pkg/api/errors"
)

func TestEtcdParseWatchResourceVersion(t *testing.T) {
	testCases := []struct {
		Version       string
		Kind          string
		ExpectVersion uint64
		Err           bool
	}{
		{Version: "", ExpectVersion: 0},
		{Version: "a", Err: true},
		{Version: " ", Err: true},
		{Version: "1", ExpectVersion: 2},
		{Version: "10", ExpectVersion: 11},
	}
	for _, testCase := range testCases {
		version, err := ParseWatchResourceVersion(testCase.Version, testCase.Kind)
		switch {
		case testCase.Err:
			if err == nil {
				t.Errorf("%s: unexpected non-error", testCase.Version)
				continue
			}
			if !errors.IsInvalid(err) {
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
