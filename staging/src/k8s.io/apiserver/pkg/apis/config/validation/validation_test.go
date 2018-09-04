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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/apis/config"
	"testing"
	"time"
)

func TestValidateLeaderElectionConfiguration(t *testing.T) {
	validConfig := &config.LeaderElectionConfiguration{
		ResourceLock:  "configmap",
		LeaderElect:   true,
		LeaseDuration: metav1.Duration{Duration: 30 * time.Second},
		RenewDeadline: metav1.Duration{Duration: 15 * time.Second},
		RetryPeriod:   metav1.Duration{Duration: 5 * time.Second},
	}

	renewDeadlineExceedsLeaseDuration := validConfig.DeepCopy()
	renewDeadlineExceedsLeaseDuration.RenewDeadline = metav1.Duration{Duration: 45 * time.Second}

	renewDeadlineZero := validConfig.DeepCopy()
	renewDeadlineZero.RenewDeadline = metav1.Duration{Duration: 0 * time.Second}

	leaseDurationZero := validConfig.DeepCopy()
	leaseDurationZero.LeaseDuration = metav1.Duration{Duration: 0 * time.Second}

	negativeValForRetryPeriod := validConfig.DeepCopy()
	negativeValForRetryPeriod.RetryPeriod = metav1.Duration{Duration: -45 * time.Second}

	negativeValForLeaseDuration := validConfig.DeepCopy()
	negativeValForLeaseDuration.LeaseDuration = metav1.Duration{Duration: -45 * time.Second}

	negativeValForRenewDeadline := validConfig.DeepCopy()
	negativeValForRenewDeadline.RenewDeadline = metav1.Duration{Duration: -45 * time.Second}

	LeaderElectButLeaderElectNotEnabled := validConfig.DeepCopy()
	LeaderElectButLeaderElectNotEnabled.LeaderElect = false
	LeaderElectButLeaderElectNotEnabled.LeaseDuration = metav1.Duration{Duration: -45 * time.Second}

	resourceLockNotDefined := validConfig.DeepCopy()
	resourceLockNotDefined.ResourceLock = ""

	scenarios := map[string]struct {
		expectedToFail bool
		config         *config.LeaderElectionConfiguration
	}{
		"good": {
			expectedToFail: false,
			config:         validConfig,
		},
		"good-dont-check-leader-config-if-not-enabled": {
			expectedToFail: false,
			config:         LeaderElectButLeaderElectNotEnabled,
		},
		"bad-renew-deadline-exceeds-lease-duration": {
			expectedToFail: true,
			config:         renewDeadlineExceedsLeaseDuration,
		},
		"bad-negative-value-for-retry-period": {
			expectedToFail: true,
			config:         negativeValForRetryPeriod,
		},
		"bad-negative-value-for-lease-duration": {
			expectedToFail: true,
			config:         negativeValForLeaseDuration,
		},
		"bad-negative-value-for-renew-deadline": {
			expectedToFail: true,
			config:         negativeValForRenewDeadline,
		},
		"bad-renew-deadline-zero": {
			expectedToFail: true,
			config:         renewDeadlineZero,
		},
		"bad-lease-duration-zero": {
			expectedToFail: true,
			config:         leaseDurationZero,
		},
		"bad-resource-lock-not-defined": {
			expectedToFail: true,
			config:         resourceLockNotDefined,
		},
	}

	for name, scenario := range scenarios {
		errs := ValidateLeaderElectionConfiguration(scenario.config, field.NewPath("leaderElectionConfiguration"))
		if len(errs) == 0 && scenario.expectedToFail {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if len(errs) > 0 && !scenario.expectedToFail {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, errs)
		}
	}
}
