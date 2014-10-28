/*
Copyright 2014 Google Inc. All rights reserved.

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

package resources

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestGetInteger(t *testing.T) {
	tests := []struct {
		res      api.ResourceList
		name     api.ResourceName
		expected int
		def      int
		test     string
	}{
		{
			res:      api.ResourceList{},
			name:     CPU,
			expected: 1,
			def:      1,
			test:     "nothing present",
		},
		{
			res: api.ResourceList{
				CPU: util.NewIntOrStringFromInt(2),
			},
			name:     CPU,
			expected: 2,
			def:      1,
			test:     "present",
		},
		{
			res: api.ResourceList{
				Memory: util.NewIntOrStringFromInt(2),
			},
			name:     CPU,
			expected: 1,
			def:      1,
			test:     "not-present",
		},
		{
			res: api.ResourceList{
				CPU: util.NewIntOrStringFromString("2"),
			},
			name:     CPU,
			expected: 2,
			def:      1,
			test:     "present-string",
		},
		{
			res: api.ResourceList{
				CPU: util.NewIntOrStringFromString("foo"),
			},
			name:     CPU,
			expected: 1,
			def:      1,
			test:     "present-invalid",
		},
	}

	for _, test := range tests {
		val := GetIntegerResource(test.res, test.name, test.def)
		if val != test.expected {
			t.Errorf("expected: %d found %d", test.expected, val)
		}
	}
}
func TestGetFloat(t *testing.T) {
	tests := []struct {
		res      api.ResourceList
		name     api.ResourceName
		expected float64
		def      float64
		test     string
	}{
		{
			res:      api.ResourceList{},
			name:     CPU,
			expected: 1.5,
			def:      1.5,
			test:     "nothing present",
		},
		{
			res: api.ResourceList{
				CPU: util.NewIntOrStringFromInt(2),
			},
			name:     CPU,
			expected: 2.0,
			def:      1.5,
			test:     "present",
		},
		{
			res: api.ResourceList{
				CPU: util.NewIntOrStringFromString("2.5"),
			},
			name:     CPU,
			expected: 2.5,
			def:      1,
			test:     "present-string",
		},
		{
			res: api.ResourceList{
				CPU: util.NewIntOrStringFromString("foo"),
			},
			name:     CPU,
			expected: 1,
			def:      1,
			test:     "present-invalid",
		},
	}

	for _, test := range tests {
		val := GetFloatResource(test.res, test.name, test.def)
		if val != test.expected {
			t.Errorf("expected: %d found %d", test.expected, val)
		}
	}
}
func TestGetString(t *testing.T) {
	tests := []struct {
		res      api.ResourceList
		name     api.ResourceName
		expected string
		def      string
		test     string
	}{
		{
			res:      api.ResourceList{},
			name:     CPU,
			expected: "foo",
			def:      "foo",
			test:     "nothing present",
		},
		{
			res: api.ResourceList{
				CPU: util.NewIntOrStringFromString("bar"),
			},
			name:     CPU,
			expected: "bar",
			def:      "foo",
			test:     "present",
		},
	}

	for _, test := range tests {
		val := GetStringResource(test.res, test.name, test.def)
		if val != test.expected {
			t.Errorf("expected: %d found %d", test.expected, val)
		}
	}
}
