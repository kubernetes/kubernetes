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

package v1_test

import (
	"strings"
	"testing"

	core "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/api/v1"
)

func TestGenerateName(t *testing.T) {
	testCases := []struct {
		objectMeta   *core.ObjectMeta
		expectedName string
		expectPrefix string
	}{
		{
			objectMeta: &core.ObjectMeta{
				GenerateName: "",
				Name:         "",
			},
			expectedName: "",
		},
		{
			objectMeta: &core.ObjectMeta{
				GenerateName: "",
				Name:         "test-name",
			},
			expectedName: "test-name",
		},
		{
			objectMeta: &core.ObjectMeta{
				GenerateName: "test-generate-name-",
				Name:         "",
			},
			expectPrefix: "test-generate-name-",
		},
		{
			objectMeta: &core.ObjectMeta{
				GenerateName: "test-generate-name-",
				Name:         "test-name",
			},
			expectedName: "test-name",
		},
	}

	for i, tc := range testCases {
		v1.GenerateName(v1.SimpleNameGenerator, tc.objectMeta)
		if tc.expectedName != "" && tc.expectedName != tc.objectMeta.Name {
			t.Errorf("[%v] expected name converted to %s, got %s", i, tc.expectedName, tc.objectMeta.Name)
		} else if tc.expectPrefix != "" && !strings.Contains(tc.objectMeta.Name, tc.expectPrefix) {
			t.Errorf("[%v] expected prefix %s, got name %s", i, tc.expectPrefix, tc.objectMeta.Name)
		}
	}
}

func TestSimpleNameGenerator(t *testing.T) {
	testCases := []struct {
		base         string
		expectPrefix string
	}{
		{
			base:         "test-base-",
			expectPrefix: "test-base-",
		},
		{
			// Length of base is 60, and the max length is 58
			base: strings.Join([]string{
				"0123456789",
				"0123456789",
				"0123456789",
				"0123456789",
				"0123456789",
				"0123456789",
			}, ""),
			expectPrefix: strings.Join([]string{
				"0123456789",
				"0123456789",
				"0123456789",
				"0123456789",
				"0123456789",
				"01234567",
			}, ""),
		},
	}

	for i, tc := range testCases {
		generatedName := v1.SimpleNameGenerator.GenerateName(tc.base)
		if !strings.Contains(generatedName, tc.expectPrefix) {
			t.Errorf("[%v] expected prefix %s, got name %s", i, tc.expectPrefix, generatedName)
		}
	}
}
