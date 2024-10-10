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
	"io"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
)

func TestEnabledPluginNames(t *testing.T) {
	scenarios := []struct {
		expectedPluginNames       []string
		setDefaultOffPlugins      sets.Set[string]
		setRecommendedPluginOrder []string
		setEnablePlugins          []string
		setDisablePlugins         []string
		setAdmissionControl       []string
	}{
		// scenario 0: check if a call to enabledPluginNames sets expected values.
		{
			expectedPluginNames: []string{"NamespaceLifecycle", "MutatingAdmissionPolicy", "MutatingAdmissionWebhook", "ValidatingAdmissionPolicy", "ValidatingAdmissionWebhook"},
		},

		// scenario 1: use default off plugins if no specified
		{
			expectedPluginNames:       []string{"pluginB"},
			setRecommendedPluginOrder: []string{"pluginA", "pluginB", "pluginC", "pluginD"},
			setDefaultOffPlugins:      sets.New("pluginA", "pluginC", "pluginD"),
		},

		// scenario 2: use default off plugins and with RecommendedPluginOrder
		{
			expectedPluginNames:       []string{"pluginA", "pluginB", "pluginC", "pluginD"},
			setRecommendedPluginOrder: []string{"pluginA", "pluginB", "pluginC", "pluginD"},
			setDefaultOffPlugins:      sets.Set[string]{},
		},

		// scenario 3: use default off plugins and specified by enable-admission-plugins with RecommendedPluginOrder
		{
			expectedPluginNames:       []string{"pluginA", "pluginB", "pluginC", "pluginD"},
			setRecommendedPluginOrder: []string{"pluginA", "pluginB", "pluginC", "pluginD"},
			setDefaultOffPlugins:      sets.New("pluginC", "pluginD"),
			setEnablePlugins:          []string{"pluginD", "pluginC"},
		},

		// scenario 4: use default off plugins and specified by disable-admission-plugins with RecommendedPluginOrder
		{
			expectedPluginNames:       []string{"pluginB"},
			setRecommendedPluginOrder: []string{"pluginA", "pluginB", "pluginC", "pluginD"},
			setDefaultOffPlugins:      sets.New("pluginC", "pluginD"),
			setDisablePlugins:         []string{"pluginA"},
		},

		// scenario 5: use default off plugins and specified by enable-admission-plugins and disable-admission-plugins with RecommendedPluginOrder
		{
			expectedPluginNames:       []string{"pluginA", "pluginC"},
			setRecommendedPluginOrder: []string{"pluginA", "pluginB", "pluginC", "pluginD"},
			setDefaultOffPlugins:      sets.New("pluginC", "pluginD"),
			setEnablePlugins:          []string{"pluginC"},
			setDisablePlugins:         []string{"pluginB"},
		},

		// scenario 6: use default off plugins and specified by admission-control with RecommendedPluginOrder
		{
			expectedPluginNames:       []string{"pluginA", "pluginB", "pluginC"},
			setRecommendedPluginOrder: []string{"pluginA", "pluginB", "pluginC", "pluginD"},
			setDefaultOffPlugins:      sets.New("pluginD"),
			setAdmissionControl:       []string{"pluginA", "pluginB"},
		},

		// scenario 7: use default off plugins and specified by admission-control with RecommendedPluginOrder
		{
			expectedPluginNames:       []string{"pluginA", "pluginB", "pluginC"},
			setRecommendedPluginOrder: []string{"pluginA", "pluginB", "pluginC", "pluginD"},
			setDefaultOffPlugins:      sets.New("pluginC", "pluginD"),
			setAdmissionControl:       []string{"pluginA", "pluginB", "pluginC"},
		},
	}

	// act
	for index, scenario := range scenarios {
		t.Run(fmt.Sprintf("scenario %d", index), func(t *testing.T) {
			target := NewAdmissionOptions()

			if scenario.setDefaultOffPlugins != nil {
				target.DefaultOffPlugins = scenario.setDefaultOffPlugins
			}
			if scenario.setRecommendedPluginOrder != nil {
				target.RecommendedPluginOrder = scenario.setRecommendedPluginOrder
			}
			if scenario.setEnablePlugins != nil {
				target.EnablePlugins = scenario.setEnablePlugins
			}
			if scenario.setDisablePlugins != nil {
				target.DisablePlugins = scenario.setDisablePlugins
			}
			if scenario.setAdmissionControl != nil {
				target.EnablePlugins = scenario.setAdmissionControl
			}

			actualPluginNames := target.enabledPluginNames()

			if len(actualPluginNames) != len(scenario.expectedPluginNames) {
				t.Fatalf("incorrect number of items, got %d, expected = %d", len(actualPluginNames), len(scenario.expectedPluginNames))
			}
			for i := range actualPluginNames {
				if scenario.expectedPluginNames[i] != actualPluginNames[i] {
					t.Errorf("missmatch at index = %d, got = %s, expected = %s", i, actualPluginNames[i], scenario.expectedPluginNames[i])
				}
			}
		})
	}
}

func TestValidate(t *testing.T) {
	scenarios := []struct {
		setEnablePlugins           []string
		setDisablePlugins          []string
		setRecommendedPluginsOrder []string
		expectedResult             bool
	}{
		// scenario 0: not set any flag
		{
			expectedResult: true,
		},

		// scenario 1: set both `--enable-admission-plugins` `--disable-admission-plugins`
		{
			setEnablePlugins:  []string{"pluginA", "pluginB"},
			setDisablePlugins: []string{"pluginC"},
			expectedResult:    true,
		},

		// scenario 2: set invalid `--enable-admission-plugins` `--disable-admission-plugins`
		{
			setEnablePlugins:  []string{"pluginA", "pluginB"},
			setDisablePlugins: []string{"pluginB"},
			expectedResult:    false,
		},

		// scenario 3: set only invalid `--enable-admission-plugins`
		{
			setEnablePlugins: []string{"pluginA", "pluginE"},
			expectedResult:   false,
		},

		// scenario 4: set only invalid `--disable-admission-plugins`
		{
			setDisablePlugins: []string{"pluginA", "pluginE"},
			expectedResult:    false,
		},

		// scenario 5: set valid `--enable-admission-plugins`
		{
			setEnablePlugins: []string{"pluginA", "pluginB"},
			expectedResult:   true,
		},

		// scenario 6: set valid `--disable-admission-plugins`
		{
			setDisablePlugins: []string{"pluginA"},
			expectedResult:    true,
		},

		// scenario 7: RecommendedPluginOrder has duplicated plugin
		{
			setRecommendedPluginsOrder: []string{"pluginA", "pluginB", "pluginB", "pluginC"},
			expectedResult:             false,
		},

		// scenario 8: RecommendedPluginOrder not equal to registered
		{
			setRecommendedPluginsOrder: []string{"pluginA", "pluginB", "pluginC"},
			expectedResult:             false,
		},

		// scenario 9: RecommendedPluginOrder equal to registered
		{
			setRecommendedPluginsOrder: []string{"pluginA", "pluginB", "pluginC", "pluginD"},
			expectedResult:             true,
		},

		// scenario 10: RecommendedPluginOrder not equal to registered
		{
			setRecommendedPluginsOrder: []string{"pluginA", "pluginB", "pluginC", "pluginE"},
			expectedResult:             false,
		},
	}

	for index, scenario := range scenarios {
		t.Run(fmt.Sprintf("scenario %d", index), func(t *testing.T) {
			options := NewAdmissionOptions()
			options.DefaultOffPlugins = sets.New("pluginC", "pluginD")
			options.RecommendedPluginOrder = []string{"pluginA", "pluginB", "pluginC", "pluginD"}
			options.Plugins = &admission.Plugins{}
			for _, plugin := range options.RecommendedPluginOrder {
				options.Plugins.Register(plugin, func(config io.Reader) (admission.Interface, error) {
					return nil, nil
				})
			}

			if scenario.setEnablePlugins != nil {
				options.EnablePlugins = scenario.setEnablePlugins
			}
			if scenario.setDisablePlugins != nil {
				options.DisablePlugins = scenario.setDisablePlugins
			}
			if scenario.setRecommendedPluginsOrder != nil {
				options.RecommendedPluginOrder = scenario.setRecommendedPluginsOrder
			}

			err := options.Validate()
			if len(err) > 0 && scenario.expectedResult {
				t.Errorf("Unexpected err: %v", err)
			}
			if len(err) == 0 && !scenario.expectedResult {
				t.Errorf("Expect error, but got none")
			}
		})
	}
}
