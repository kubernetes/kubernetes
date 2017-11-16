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

package annotations

import "testing"

func TestValidateAndParse(t *testing.T) {
	cases := []struct {
		s             string
		expectCorrect bool
		expectedKey   string
	}{
		{
			s:             "metadata.annotations['foo']",
			expectCorrect: true,
			expectedKey:   "foo",
		},
		{
			s:             "metadata.annotations['\\[let\\'s\"escape\\[some\\]characters']",
			expectCorrect: true,
			expectedKey:   "\\[let\\'s\"escape\\[some\\]characters",
		},
		{
			s:             "metadata.annotations['and]here[we[]failed[with]escaping']",
			expectCorrect: false,
			expectedKey:   "",
		},
		{
			s:             "metadata.annotations['and'here[]even'worse']",
			expectCorrect: false,
			expectedKey:   "",
		},
		{
			s:             "metadata.annotations[\"foo\"]",
			expectCorrect: false,
			expectedKey:   "",
		},
		{
			s:             "metadata.annotations['']",
			expectCorrect: false,
			expectedKey:   "",
		},
		{
			s:             "metadata.annotations[]",
			expectCorrect: false,
			expectedKey:   "",
		},
		{
			s:             "metadata.annotations['foo']someunwantedtext",
			expectCorrect: false,
			expectedKey:   "",
		},
		{
			s:             "metadata.foo",
			expectCorrect: false,
			expectedKey:   "",
		},
	}

	for _, c := range cases {
		isCorrect, key := ValidateAndParse(c.s)

		if c.expectCorrect && !isCorrect {
			t.Errorf("Expected %s to match, but it doesn't", c.s)
		}

		if !c.expectCorrect && isCorrect {
			t.Errorf("Expected %s to not match, but it does", c.s)
		}

		if c.expectedKey != key {
			t.Errorf("Expected %s key, instead got %s", c.expectedKey, key)
		}
	}
}
