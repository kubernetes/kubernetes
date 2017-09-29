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

package options

import (
	"fmt"
	"testing"
)

func TestEnabledPluginNamesMethod(t *testing.T) {
	scenarios := []struct {
		expectedPluginNames       []string
		setDefaultOffPluginNames  []string
		setRecommendedPluginOrder []string
	}{
		// scenario 1: check if a call to enabledPluginNames sets expected values.
		{
			expectedPluginNames: []string{"NamespaceLifecycle"},
		},

		// scenario 2: overwrite RecommendedPluginOrder and set DefaultOffPluginNames
		// make sure that plugins which are on DefaultOffPluginNames list do not get to PluginNames list.
		{
			expectedPluginNames:       []string{"pluginA"},
			setRecommendedPluginOrder: []string{"pluginA", "pluginB"},
			setDefaultOffPluginNames:  []string{"pluginB"},
		},
	}

	// act
	for index, scenario := range scenarios {
		t.Run(fmt.Sprintf("scenario %d", index), func(t *testing.T) {
			target := NewAdmissionOptions()

			if len(scenario.setDefaultOffPluginNames) > 0 {
				target.DefaultOffPlugins = scenario.setDefaultOffPluginNames
			}
			if len(scenario.setRecommendedPluginOrder) > 0 {
				target.RecommendedPluginOrder = scenario.setRecommendedPluginOrder
			}

			actualPluginNames := target.enabledPluginNames()

			if len(actualPluginNames) != len(scenario.expectedPluginNames) {
				t.Errorf("incorrect number of items, got %d, expected = %d", len(actualPluginNames), len(scenario.expectedPluginNames))
			}
			for i := range actualPluginNames {
				if scenario.expectedPluginNames[i] != actualPluginNames[i] {
					t.Errorf("missmatch at index = %d, got = %s, expected = %s", i, actualPluginNames[i], scenario.expectedPluginNames[i])
				}
			}
		})
	}
}
