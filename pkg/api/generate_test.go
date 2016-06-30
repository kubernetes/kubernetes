/*
Copyright 2014 The Kubernetes Authors.

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

package api

import (
	"strings"
	"testing"
)

type nameGeneratorFunc func(base string) string

func (fn nameGeneratorFunc) GenerateName(base string) string {
	return fn(base)
}

func TestGenerateName(t *testing.T) {
	testCases := []struct {
		meta ObjectMeta

		base     string
		returned string
	}{
		{
			returned: "",
		},
		{
			meta: ObjectMeta{
				GenerateName: "test",
			},
			base:     "test",
			returned: "test",
		},
		{
			meta: ObjectMeta{
				Name:         "foo",
				GenerateName: "test",
			},
			base:     "test",
			returned: "foo",
		},
	}

	for i, testCase := range testCases {
		GenerateName(nameGeneratorFunc(func(base string) string {
			if base != testCase.base {
				t.Errorf("%d: unexpected call with base", i)
			}
			return "test"
		}), &testCase.meta)
		expect := testCase.returned
		if expect != testCase.meta.Name {
			t.Errorf("%d: unexpected name: %#v", i, testCase.meta)
		}
	}
}

func TestSimpleNameGenerator(t *testing.T) {
	meta := &ObjectMeta{
		GenerateName: "foo",
	}
	GenerateName(SimpleNameGenerator, meta)
	if !strings.HasPrefix(meta.Name, "foo") || meta.Name == "foo" {
		t.Errorf("unexpected name: %#v", meta)
	}
}
