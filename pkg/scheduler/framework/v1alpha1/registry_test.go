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
		expeted         PluginFooConfig
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
			expeted: PluginFooConfig{
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
			expeted: PluginFooConfig{
				FooTest: "test decode",
			},
		},
	}
	for i, test := range tests {
		schedulerConf, err := loadConfig([]byte(test.schedulerConfig))
		if err != nil {
			t.Errorf("Test #%v(%s): failed to load scheduler config: %v", i, test.name, err)
		}
		var pluginFooConf PluginFooConfig
		if err := DecodeInto(&schedulerConf.PluginConfig[0].Args, &pluginFooConf); err != nil {
			t.Errorf("Test #%v(%s): failed to decode args %+v: %v",
				i, test.name, schedulerConf.PluginConfig[0].Args, err)
		}
		if !reflect.DeepEqual(pluginFooConf, test.expeted) {
			t.Errorf("Test #%v(%s): failed to decode plugin config, expected: %+v, got: %+v",
				i, test.name, test.expeted, pluginFooConf)
		}
	}
}

func loadConfig(data []byte) (*config.KubeSchedulerConfiguration, error) {
	configObj := &config.KubeSchedulerConfiguration{}
	if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), data, configObj); err != nil {
		return nil, err
	}

	return configObj, nil
}
