/*
Copyright 2021 The Kubernetes Authors.

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

func TestMaskTrailingDash(t *testing.T) {
	testCases := []struct {
		beforeMasking        string
		expectedAfterMasking string
		description          string
	}{
		{
			beforeMasking:        "",
			expectedAfterMasking: "",
			description:          "empty string",
		},
		{
			beforeMasking:        "-",
			expectedAfterMasking: "-",
			description:          "only a single dash",
		},
		{
			beforeMasking:        "-foo",
			expectedAfterMasking: "-foo",
			description:          "has leading dash",
		},
		{
			beforeMasking:        "-foo-",
			expectedAfterMasking: "-foa",
			description:          "has both leading and trailing dashes",
		},
		{
			beforeMasking:        "b-",
			expectedAfterMasking: "a",
			description:          "has trailing dash",
		},
		{
			beforeMasking:        "ab",
			expectedAfterMasking: "ab",
			description:          "has neither leading nor trailing dashes",
		},
	}

	for _, tc := range testCases {
		afterMasking := maskTrailingDash(tc.beforeMasking)
		if afterMasking != tc.expectedAfterMasking {
			t.Errorf("error in test case: %s. expected: %s, actual: %s", tc.description, tc.expectedAfterMasking, afterMasking)
		}
	}
}
