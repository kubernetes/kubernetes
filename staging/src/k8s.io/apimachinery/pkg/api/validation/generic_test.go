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

import (
	"strings"
	"testing"
)

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
			expectedAfterMasking: "-fooa",
			description:          "has both leading and trailing dashes",
		},
		{
			beforeMasking:        "b-",
			expectedAfterMasking: "ba",
			description:          "has trailing dash",
		},
		{
			beforeMasking:        "a?-",
			expectedAfterMasking: "a?a",
			description:          "preserves invalid character before trailing dash",
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

func TestNameValidationAfterMaskingTrailingDash(t *testing.T) {
	for _, validator := range []struct {
		name      string
		validate  ValidateNameFunc
		maxLength int
	}{
		{"DNSSubdomain", NameIsDNSSubdomain, 253},
		{"DNSLabel", NameIsDNSLabel, 63},
		{"DNS1035Label", NameIsDNS1035Label, 63},
	} {
		t.Run(validator.name, func(t *testing.T) {
			for _, tc := range []struct {
				name   string
				prefix bool
				valid  bool
			}{
				{"foo-", true, true},
				{"foo-", false, false},
				{"a?-", true, false},
				{"--", true, false},
				{"-", true, false},
				{strings.Repeat("a", validator.maxLength-1) + "-", true, true},
				{strings.Repeat("a", validator.maxLength) + "-", true, false},
			} {
				if errs := validator.validate(tc.name, tc.prefix); (len(errs) == 0) != tc.valid {
					t.Errorf("validate(%q, prefix=%t) = %v, want valid=%t", tc.name, tc.prefix, errs, tc.valid)
				}
			}
		})
	}
}
