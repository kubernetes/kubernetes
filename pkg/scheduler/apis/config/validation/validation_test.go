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
	"testing"
	"time"

	apimachinery "k8s.io/apimachinery/pkg/apis/config"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiserver "k8s.io/apiserver/pkg/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
)

func TestValidateKubeSchedulerConfiguration(t *testing.T) {
	testTimeout := int64(0)
	validConfig := &config.KubeSchedulerConfiguration{
		SchedulerName:                  "me",
		HealthzBindAddress:             "0.0.0.0:10254",
		MetricsBindAddress:             "0.0.0.0:10254",
		HardPodAffinitySymmetricWeight: 80,
		ClientConnection: apimachinery.ClientConnectionConfiguration{
			AcceptContentTypes: "application/json",
			ContentType:        "application/json",
			QPS:                10,
			Burst:              10,
		},
		AlgorithmSource: config.SchedulerAlgorithmSource{
			Policy: &config.SchedulerPolicySource{
				ConfigMap: &config.SchedulerPolicyConfigMapSource{
					Namespace: "name",
					Name:      "name",
				},
			},
		},
		LeaderElection: config.KubeSchedulerLeaderElectionConfiguration{
			LockObjectNamespace: "name",
			LockObjectName:      "name",
			LeaderElectionConfiguration: apiserver.LeaderElectionConfiguration{
				ResourceLock:  "configmap",
				LeaderElect:   true,
				LeaseDuration: metav1.Duration{Duration: 30 * time.Second},
				RenewDeadline: metav1.Duration{Duration: 15 * time.Second},
				RetryPeriod:   metav1.Duration{Duration: 5 * time.Second},
			},
		},
		BindTimeoutSeconds:       &testTimeout,
		PercentageOfNodesToScore: 35,
	}

	HardPodAffinitySymmetricWeightGt100 := validConfig.DeepCopy()
	HardPodAffinitySymmetricWeightGt100.HardPodAffinitySymmetricWeight = 120

	HardPodAffinitySymmetricWeightLt0 := validConfig.DeepCopy()
	HardPodAffinitySymmetricWeightLt0.HardPodAffinitySymmetricWeight = -1

	lockObjectNameNotSet := validConfig.DeepCopy()
	lockObjectNameNotSet.LeaderElection.LockObjectName = ""

	lockObjectNamespaceNotSet := validConfig.DeepCopy()
	lockObjectNamespaceNotSet.LeaderElection.LockObjectNamespace = ""

	metricsBindAddrHostInvalid := validConfig.DeepCopy()
	metricsBindAddrHostInvalid.MetricsBindAddress = "0.0.0.0.0:9090"

	metricsBindAddrPortInvalid := validConfig.DeepCopy()
	metricsBindAddrPortInvalid.MetricsBindAddress = "0.0.0.0:909090"

	healthzBindAddrHostInvalid := validConfig.DeepCopy()
	healthzBindAddrHostInvalid.HealthzBindAddress = "0.0.0.0.0:9090"

	healthzBindAddrPortInvalid := validConfig.DeepCopy()
	healthzBindAddrPortInvalid.HealthzBindAddress = "0.0.0.0:909090"

	enableContentProfilingSetWithoutEnableProfiling := validConfig.DeepCopy()
	enableContentProfilingSetWithoutEnableProfiling.EnableProfiling = false
	enableContentProfilingSetWithoutEnableProfiling.EnableContentionProfiling = true

	bindTimeoutUnset := validConfig.DeepCopy()
	bindTimeoutUnset.BindTimeoutSeconds = nil

	percentageOfNodesToScore101 := validConfig.DeepCopy()
	percentageOfNodesToScore101.PercentageOfNodesToScore = int32(101)

	scenarios := map[string]struct {
		expectedToFail bool
		config         *config.KubeSchedulerConfiguration
	}{
		"good": {
			expectedToFail: false,
			config:         validConfig,
		},
		"bad-lock-object-names-not-set": {
			expectedToFail: true,
			config:         lockObjectNameNotSet,
		},
		"bad-lock-object-namespace-not-set": {
			expectedToFail: true,
			config:         lockObjectNamespaceNotSet,
		},
		"bad-healthz-port-invalid": {
			expectedToFail: true,
			config:         healthzBindAddrPortInvalid,
		},
		"bad-healthz-host-invalid": {
			expectedToFail: true,
			config:         healthzBindAddrHostInvalid,
		},
		"bad-metrics-port-invalid": {
			expectedToFail: true,
			config:         metricsBindAddrPortInvalid,
		},
		"bad-metrics-host-invalid": {
			expectedToFail: true,
			config:         metricsBindAddrHostInvalid,
		},
		"bad-hard-pod-affinity-symmetric-weight-lt-0": {
			expectedToFail: true,
			config:         HardPodAffinitySymmetricWeightGt100,
		},
		"bad-hard-pod-affinity-symmetric-weight-gt-100": {
			expectedToFail: true,
			config:         HardPodAffinitySymmetricWeightLt0,
		},
		"bind-timeout-unset": {
			expectedToFail: true,
			config:         bindTimeoutUnset,
		},
		"bad-percentage-of-nodes-to-score": {
			expectedToFail: true,
			config:         percentageOfNodesToScore101,
		},
	}

	for name, scenario := range scenarios {
		errs := ValidateKubeSchedulerConfiguration(scenario.config)
		if len(errs) == 0 && scenario.expectedToFail {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if len(errs) > 0 && !scenario.expectedToFail {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, errs)
		}
	}
}
