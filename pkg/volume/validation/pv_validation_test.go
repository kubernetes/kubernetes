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

package validation

import (
	"testing"
)

func TestValidatePathNoBacksteps(t *testing.T) {
	testCases := map[string]struct {
		path        string
		expectedErr bool
	}{
		"valid path": {
			path: "/foo/bar",
		},
		"invalid path": {
			path:        "/foo/bar/..",
			expectedErr: true,
		},
	}

	for name, tc := range testCases {
		err := ValidatePathNoBacksteps(tc.path)

		if err == nil && tc.expectedErr {
			t.Fatalf("expected test `%s` to return an error but it didn't", name)
		}

		if err != nil && !tc.expectedErr {
			t.Fatalf("expected test `%s` to return no error but got `%v`", name, err)
		}
	}
}
