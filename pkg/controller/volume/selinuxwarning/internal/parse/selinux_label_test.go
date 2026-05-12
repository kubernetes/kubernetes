/*
Copyright The Kubernetes Authors.

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

package parse

import (
	"reflect"
	"testing"
)

func TestParseSELinuxLabel(t *testing.T) {
	tests := []struct {
		name          string
		label         string
		expectedParts []string
	}{
		{
			name:          "empty label",
			label:         "",
			expectedParts: []string{"", "", "", ""},
		},
		{
			name:          "complete label with all components",
			label:         "system_u:system_r:container_t:s0:c0,c1",
			expectedParts: []string{"system_u", "system_r", "container_t", "s0:c0,c1"},
		},
		{
			name:          "label with user, role, and type only",
			label:         "system_u:system_r:container_t",
			expectedParts: []string{"system_u", "system_r", "container_t", ""},
		},
		{
			name:          "label with user and role only",
			label:         "system_u:system_r",
			expectedParts: []string{"system_u", "system_r", "", ""},
		},
		{
			name:          "label with user only",
			label:         "system_u",
			expectedParts: []string{"system_u", "", "", ""},
		},
		{
			name:          "label missing user but with role and type",
			label:         ":system_r:container_t",
			expectedParts: []string{"", "system_r", "container_t", ""},
		},
		{
			name:          "label missing user and role but with type",
			label:         "::container_t",
			expectedParts: []string{"", "", "container_t", ""},
		},
		{
			name:          "label missing user and role but with type and level",
			label:         "::container_t:s0",
			expectedParts: []string{"", "", "container_t", "s0"},
		},
		{
			name:          "label with all empty components except level",
			label:         ":::s0:c0,c1",
			expectedParts: []string{"", "", "", "s0:c0,c1"},
		},
		{
			name:          "label with special characters in components",
			label:         "user_with_underscore:role-with-dash:type.with.dots:s0:c0.c1",
			expectedParts: []string{"user_with_underscore", "role-with-dash", "type.with.dots", "s0:c0.c1"},
		},
		{
			name:          "label with extra colons in level component",
			label:         "user:role:type:s0:c0,c1:extra",
			expectedParts: []string{"user", "role", "type", "s0:c0,c1:extra"},
		},
		{
			name:          "multiple colons only",
			label:         ":::",
			expectedParts: []string{"", "", "", ""},
		},
		{
			name:          "five colons",
			label:         ":::::",
			expectedParts: []string{"", "", "", "::"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parts := ParseSELinuxLabel(tt.label)
			partsSlice := parts[:]
			if !reflect.DeepEqual(partsSlice, tt.expectedParts) {
				t.Errorf("ParseSELinuxLabel(%q) = %v, expected parts = %v", tt.label, partsSlice, tt.expectedParts)
			}
		})
	}
}
