// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package docker

import (
	"reflect"
	"regexp"
	"testing"
)

func TestParseDockerAPIVersion(t *testing.T) {
	tests := []struct {
		version       string
		regex         *regexp.Regexp
		length        int
		expected      []int
		expectedError string
	}{
		{"17.03.0", version_re, 3, []int{17, 03, 0}, ""},
		{"17.a3.0", version_re, 3, []int{}, `version string "17.a3.0" doesn't match expected regular expression: "(\d+)\.(\d+)\.(\d+)"`},
		{"1.20", apiversion_re, 2, []int{1, 20}, ""},
		{"1.a", apiversion_re, 2, []int{}, `version string "1.a" doesn't match expected regular expression: "(\d+)\.(\d+)"`},
	}

	for _, test := range tests {
		actual, err := parseVersion(test.version, test.regex, test.length)
		if err != nil {
			if len(test.expectedError) == 0 {
				t.Errorf("%s: expected no error, got %v", test.version, err)
			} else if err.Error() != test.expectedError {
				t.Errorf("%s: expected error %v, got %v", test.version, test.expectedError, err)
			}
		} else {
			if !reflect.DeepEqual(actual, test.expected) {
				t.Errorf("%s: expected array %v, got %v", test.version, test.expected, actual)
			}
		}
	}
}
