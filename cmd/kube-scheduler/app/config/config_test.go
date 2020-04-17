/*
Copyright 2018 The Kubernetes Authors.

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

package config

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	apiserver "k8s.io/apiserver/pkg/server"
)

func TestConfigComplete(t *testing.T) {
	scenarios := []struct {
		name   string
		want   *Config
		config *Config
	}{
		{
			name: "SetInsecureServingName",
			want: &Config{
				InsecureServing: &apiserver.DeprecatedInsecureServingInfo{
					Name: "healthz",
				},
			},
			config: &Config{
				InsecureServing: &apiserver.DeprecatedInsecureServingInfo{},
			},
		},
		{
			name: "SetMetricsInsecureServingName",
			want: &Config{
				InsecureMetricsServing: &apiserver.DeprecatedInsecureServingInfo{
					Name: "metrics",
				},
			},
			config: &Config{
				InsecureMetricsServing: &apiserver.DeprecatedInsecureServingInfo{},
			},
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			cc := scenario.config.Complete()

			returnValue := cc.completedConfig.Config

			if diff := cmp.Diff(scenario.want, returnValue); diff != "" {
				t.Errorf("Complete(): Unexpected return value (-want, +got): %s", diff)
			}
		})
	}
}
