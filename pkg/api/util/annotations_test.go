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

package util

import "testing"

func TestValidate(t *testing.T) {
	cases := []struct {
		s             string
		expectCorrect bool
	}{
		{
			s:             "metadata.annotations['foo']",
			expectCorrect: true,
		},
		{
			s:             "metadata.annotations['\\[let\\'s\"escape\\[some\\]characters']",
			expectCorrect: true,
		},
		{
			s:             "metadata.annotations['and]here[we[]failed[with]escaping']",
			expectCorrect: false,
		},
		{
			s:             "metadata.annotations['and'here[]even'worse']",
			expectCorrect: false,
		},
		{
			s:             "metadata.annotations[\"foo\"]",
			expectCorrect: false,
		},
		{
			s:             "metadata.annotations['']",
			expectCorrect: false,
		},
		{
			s:             "metadata.annotations['foo']someunwantedtext",
			expectCorrect: false,
		},
		{
			s:             "metadata.foo",
			expectCorrect: false,
		},
	}

	parser, err := NewAnnotationFieldPathParser()
	if err != nil {
		t.Fatal(err)
	}

	for _, c := range cases {
		isCorrect := parser.Validate(c.s)

		if c.expectCorrect && !isCorrect {
			t.Errorf("Expected %s to match, but it doesn't", c.s)
		}

		if !c.expectCorrect && isCorrect {
			t.Errorf("Expected %s to not match, but it does", c.s)
		}
	}
}

func TestParseKey(t *testing.T) {
	cases := []struct {
		s           string
		expectedKey string
	}{
		{
			s:           "metadata.annotations['foo']",
			expectedKey: "foo",
		},
		{
			s:           "metadata.annotations['spec.pod.beta.kubernetes.io/statefulset-index']",
			expectedKey: "spec.pod.beta.kubernetes.io/statefulset-index",
		},
	}

	parser, err := NewAnnotationFieldPathParser()
	if err != nil {
		t.Fatal(err)
	}

	for _, c := range cases {
		key, err := parser.ParseKey(c.s)
		if err != nil {
			t.Fatal(err)
		}

		if key != c.expectedKey {
			t.Errorf("Expected %s, instead got %s", c.expectedKey, key)
		}
	}
}
