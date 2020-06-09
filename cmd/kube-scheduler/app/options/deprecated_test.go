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

package options

import (
	"testing"
)

func TestValidateDeprecatedKubeSchedulerConfiguration(t *testing.T) {
	scenarios := map[string]struct {
		expectedToFail bool
		config         *DeprecatedOptions
	}{
		"good": {
			config: &DeprecatedOptions{
				PolicyConfigFile:      "/some/file",
				UseLegacyPolicyConfig: true,
				AlgorithmProvider:     "",
			},
		},
		"bad-policy-config-file-null": {
			expectedToFail: true,
			config: &DeprecatedOptions{
				PolicyConfigFile:      "",
				UseLegacyPolicyConfig: true,
				AlgorithmProvider:     "",
			},
		},
		"good affinity weight": {
			config: &DeprecatedOptions{
				HardPodAffinitySymmetricWeight: 50,
			},
		},
		"bad affinity weight": {
			expectedToFail: true,
			config: &DeprecatedOptions{
				HardPodAffinitySymmetricWeight: -1,
			},
		},
	}

	for name, scenario := range scenarios {
		errs := scenario.config.Validate()
		if len(errs) == 0 && scenario.expectedToFail {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if len(errs) > 0 && !scenario.expectedToFail {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, errs)
		}
	}
}
