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

package validation

import (
	"k8s.io/apimachinery/pkg/apis/config"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"testing"
)

func TestValidateClientConnectionConfiguration(t *testing.T) {
	validConfig := &config.ClientConnectionConfiguration{
		AcceptContentTypes: "application/json",
		ContentType:        "application/json",
		QPS:                10,
		Burst:              10,
	}

	qpsLessThanZero := validConfig.DeepCopy()
	qpsLessThanZero.QPS = -1

	burstLessThanZero := validConfig.DeepCopy()
	burstLessThanZero.Burst = -1

	scenarios := map[string]struct {
		expectedToFail bool
		config         *config.ClientConnectionConfiguration
	}{
		"good": {
			expectedToFail: false,
			config:         validConfig,
		},
		"good-qps-less-than-zero": {
			expectedToFail: false,
			config:         qpsLessThanZero,
		},
		"bad-burst-less-then-zero": {
			expectedToFail: true,
			config:         burstLessThanZero,
		},
	}

	for name, scenario := range scenarios {
		errs := ValidateClientConnectionConfiguration(scenario.config, field.NewPath("clientConnectionConfiguration"))
		if len(errs) == 0 && scenario.expectedToFail {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if len(errs) > 0 && !scenario.expectedToFail {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, errs)
		}
	}
}
