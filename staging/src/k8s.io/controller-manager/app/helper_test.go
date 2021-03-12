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

package app

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/stretchr/testify/assert"
)

func TestIsControllerEnabled(t *testing.T) {
	tcs := []struct {
		name                         string
		controllerName               string
		controllers                  []string
		disabledByDefaultControllers []string
		expected                     bool
	}{
		{
			name:                         "on by name",
			controllerName:               "bravo",
			controllers:                  []string{"alpha", "bravo", "-charlie"},
			disabledByDefaultControllers: []string{"delta", "echo"},
			expected:                     true,
		},
		{
			name:                         "off by name",
			controllerName:               "charlie",
			controllers:                  []string{"alpha", "bravo", "-charlie"},
			disabledByDefaultControllers: []string{"delta", "echo"},
			expected:                     false,
		},
		{
			name:                         "on by default",
			controllerName:               "alpha",
			controllers:                  []string{"*"},
			disabledByDefaultControllers: []string{"delta", "echo"},
			expected:                     true,
		},
		{
			name:                         "off by default",
			controllerName:               "delta",
			controllers:                  []string{"*"},
			disabledByDefaultControllers: []string{"delta", "echo"},
			expected:                     false,
		},
		{
			name:                         "on by star, not off by name",
			controllerName:               "alpha",
			controllers:                  []string{"*", "-charlie"},
			disabledByDefaultControllers: []string{"delta", "echo"},
			expected:                     true,
		},
		{
			name:                         "off by name with star",
			controllerName:               "charlie",
			controllers:                  []string{"*", "-charlie"},
			disabledByDefaultControllers: []string{"delta", "echo"},
			expected:                     false,
		},
		{
			name:                         "off by default implicit, no star",
			controllerName:               "foxtrot",
			controllers:                  []string{"alpha", "bravo", "-charlie"},
			disabledByDefaultControllers: []string{"delta", "echo"},
			expected:                     false,
		},
	}

	for _, tc := range tcs {
		actual := IsControllerEnabled(tc.controllerName, sets.NewString(tc.disabledByDefaultControllers...), tc.controllers)
		assert.Equal(t, tc.expected, actual, "%v: expected %v, got %v", tc.name, tc.expected, actual)
	}

}
