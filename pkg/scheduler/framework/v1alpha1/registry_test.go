/*
Copyright 2019 The Kubernetes Authors.

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

package v1alpha1

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
)

func TestDecodeInto(t *testing.T) {
	type PluginFooConfig struct {
		FooTest string `json:"foo_test,omitempty"`
	}
	tests := []struct {
		name            string
		schedulerConfig string
		expected        PluginFooConfig
	}{
		{
			name: "test decode for JSON config",
			schedulerConfig: `{
				"kind": "KubeSchedulerConfiguration",
				"apiVersion": "kubescheduler.config.k8s.io/v1alpha1",
				"plugins": {
				"permit": {
						"enabled": [
							{
								"name": "foo"
							}
						]
					}
				},
				"pluginConfig": [
					{
						"name": "foo",
						"args": {
							"foo_test": "test decode"
						}
					}
				]
			}`,
			expected: PluginFooConfig{
				FooTest: "test decode",
			},
		},
		{
			name: "test decode for YAML config",
			schedulerConfig: `
apiVersion: kubescheduler.config.k8s.io/v1alpha1
kind: KubeSchedulerConfiguration
plugins:
  permit:
    enabled:
      - name: foo
pluginConfig:
  - name: foo
    args:
      foo_test: "test decode"`,
			expected: PluginFooConfig{
				FooTest: "test decode",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			schedulerConf, err := loadConfig([]byte(test.schedulerConfig))
			if err != nil {
				t.Errorf("loadConfig(): failed to load scheduler config: %v", err)
			}
			var pluginFooConf PluginFooConfig
			if err := DecodeInto(&schedulerConf.PluginConfig[0].Args, &pluginFooConf); err != nil {
				t.Errorf("DecodeInto(): failed to decode args %+v: %v",
					schedulerConf.PluginConfig[0].Args, err)
			}
			if !reflect.DeepEqual(test.expected, pluginFooConf) {
				t.Errorf("DecodeInto(): failed to decode plugin config, expected: %+v, got: %+v",
					test.expected, pluginFooConf)
			}
		})
	}
}

func loadConfig(data []byte) (*config.KubeSchedulerConfiguration, error) {
	configObj := &config.KubeSchedulerConfiguration{}
	if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), data, configObj); err != nil {
		return nil, err
	}

	return configObj, nil
}

// isRegistryEqual compares two registries for equality. This function is used in place of
// reflect.DeepEqual() and cmp() as they don't compare function values.
func isRegistryEqual(registryX, registryY Registry) bool {
	for name, pluginFactory := range registryY {
		if val, ok := registryX[name]; ok {
			if reflect.ValueOf(pluginFactory) != reflect.ValueOf(val) {
				// pluginFactory functions are not the same.
				return false
			}
		} else {
			// registryY contains an entry that is not present in registryX
			return false
		}
	}

	for name := range registryX {
		if _, ok := registryY[name]; !ok {
			// registryX contains an entry that is not present in registryY
			return false
		}
	}

	return true
}

type mockNoopPlugin struct{}

func (p *mockNoopPlugin) Name() string {
	return "MockNoop"
}

func NewMockNoopPluginFactory() PluginFactory {
	return func(_ *runtime.Unknown, _ FrameworkHandle) (Plugin, error) {
		return &mockNoopPlugin{}, nil
	}
}

func TestMerge(t *testing.T) {
	tests := []struct {
		name            string
		primaryRegistry Registry
		registryToMerge Registry
		expected        Registry
		shouldError     bool
	}{
		{
			name: "valid Merge",
			primaryRegistry: Registry{
				"pluginFactory1": NewMockNoopPluginFactory(),
			},
			registryToMerge: Registry{
				"pluginFactory2": NewMockNoopPluginFactory(),
			},
			expected: Registry{
				"pluginFactory1": NewMockNoopPluginFactory(),
				"pluginFactory2": NewMockNoopPluginFactory(),
			},
			shouldError: false,
		},
		{
			name: "Merge duplicate factories",
			primaryRegistry: Registry{
				"pluginFactory1": NewMockNoopPluginFactory(),
			},
			registryToMerge: Registry{
				"pluginFactory1": NewMockNoopPluginFactory(),
			},
			expected: Registry{
				"pluginFactory1": NewMockNoopPluginFactory(),
			},
			shouldError: true,
		},
	}

	for _, scenario := range tests {
		t.Run(scenario.name, func(t *testing.T) {
			err := scenario.primaryRegistry.Merge(scenario.registryToMerge)

			if (err == nil) == scenario.shouldError {
				t.Errorf("Merge() shouldError is: %v, however err is: %v.", scenario.shouldError, err)
				return
			}

			if !isRegistryEqual(scenario.expected, scenario.primaryRegistry) {
				t.Errorf("Merge(). Expected %v. Got %v instead.", scenario.expected, scenario.primaryRegistry)
			}
		})
	}
}

func TestRegister(t *testing.T) {
	tests := []struct {
		name              string
		registry          Registry
		nameToRegister    string
		factoryToRegister PluginFactory
		expected          Registry
		shouldError       bool
	}{
		{
			name:              "valid Register",
			registry:          Registry{},
			nameToRegister:    "pluginFactory1",
			factoryToRegister: NewMockNoopPluginFactory(),
			expected: Registry{
				"pluginFactory1": NewMockNoopPluginFactory(),
			},
			shouldError: false,
		},
		{
			name: "Register duplicate factories",
			registry: Registry{
				"pluginFactory1": NewMockNoopPluginFactory(),
			},
			nameToRegister:    "pluginFactory1",
			factoryToRegister: NewMockNoopPluginFactory(),
			expected: Registry{
				"pluginFactory1": NewMockNoopPluginFactory(),
			},
			shouldError: true,
		},
	}

	for _, scenario := range tests {
		t.Run(scenario.name, func(t *testing.T) {
			err := scenario.registry.Register(scenario.nameToRegister, scenario.factoryToRegister)

			if (err == nil) == scenario.shouldError {
				t.Errorf("Register() shouldError is: %v however err is: %v.", scenario.shouldError, err)
				return
			}

			if !isRegistryEqual(scenario.expected, scenario.registry) {
				t.Errorf("Register(). Expected %v. Got %v instead.", scenario.expected, scenario.registry)
			}
		})
	}
}

func TestUnregister(t *testing.T) {
	tests := []struct {
		name             string
		registry         Registry
		nameToUnregister string
		expected         Registry
		shouldError      bool
	}{
		{
			name: "valid Unregister",
			registry: Registry{
				"pluginFactory1": NewMockNoopPluginFactory(),
				"pluginFactory2": NewMockNoopPluginFactory(),
			},
			nameToUnregister: "pluginFactory1",
			expected: Registry{
				"pluginFactory2": NewMockNoopPluginFactory(),
			},
			shouldError: false,
		},
		{
			name:             "Unregister non-existent plugin factory",
			registry:         Registry{},
			nameToUnregister: "pluginFactory1",
			expected:         Registry{},
			shouldError:      true,
		},
	}

	for _, scenario := range tests {
		t.Run(scenario.name, func(t *testing.T) {
			err := scenario.registry.Unregister(scenario.nameToUnregister)

			if (err == nil) == scenario.shouldError {
				t.Errorf("Unregister() shouldError is: %v however err is: %v.", scenario.shouldError, err)
				return
			}

			if !isRegistryEqual(scenario.expected, scenario.registry) {
				t.Errorf("Unregister(). Expected %v. Got %v instead.", scenario.expected, scenario.registry)
			}
		})
	}
}
