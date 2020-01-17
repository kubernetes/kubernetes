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
	"errors"
	"fmt"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	componentbaseconfig "k8s.io/component-base/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
)

func TestValidateKubeSchedulerConfiguration(t *testing.T) {
	testTimeout := int64(0)
	podInitialBackoffSeconds := int64(1)
	podMaxBackoffSeconds := int64(1)
	validConfig := &config.KubeSchedulerConfiguration{
		SchedulerName:                  "me",
		HealthzBindAddress:             "0.0.0.0:10254",
		MetricsBindAddress:             "0.0.0.0:10254",
		HardPodAffinitySymmetricWeight: 80,
		ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
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
			LeaderElectionConfiguration: componentbaseconfig.LeaderElectionConfiguration{
				ResourceLock:      "configmap",
				LeaderElect:       true,
				LeaseDuration:     metav1.Duration{Duration: 30 * time.Second},
				RenewDeadline:     metav1.Duration{Duration: 15 * time.Second},
				RetryPeriod:       metav1.Duration{Duration: 5 * time.Second},
				ResourceNamespace: "name",
				ResourceName:      "name",
			},
		},
		PodInitialBackoffSeconds: podInitialBackoffSeconds,
		PodMaxBackoffSeconds:     podMaxBackoffSeconds,
		BindTimeoutSeconds:       testTimeout,
		PercentageOfNodesToScore: 35,
	}

	HardPodAffinitySymmetricWeightGt100 := validConfig.DeepCopy()
	HardPodAffinitySymmetricWeightGt100.HardPodAffinitySymmetricWeight = 120

	HardPodAffinitySymmetricWeightLt0 := validConfig.DeepCopy()
	HardPodAffinitySymmetricWeightLt0.HardPodAffinitySymmetricWeight = -1

	resourceNameNotSet := validConfig.DeepCopy()
	resourceNameNotSet.LeaderElection.ResourceName = ""

	resourceNamespaceNotSet := validConfig.DeepCopy()
	resourceNamespaceNotSet.LeaderElection.ResourceNamespace = ""

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
		"bad-resource-name-not-set": {
			expectedToFail: true,
			config:         resourceNameNotSet,
		},
		"bad-resource-namespace-not-set": {
			expectedToFail: true,
			config:         resourceNamespaceNotSet,
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

func TestValidatePolicy(t *testing.T) {
	tests := []struct {
		policy   config.Policy
		expected error
		name     string
	}{
		{
			name:     "no weight defined in policy",
			policy:   config.Policy{Priorities: []config.PriorityPolicy{{Name: "NoWeightPriority"}}},
			expected: errors.New("Priority NoWeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			name:     "policy weight is not positive",
			policy:   config.Policy{Priorities: []config.PriorityPolicy{{Name: "NoWeightPriority", Weight: 0}}},
			expected: errors.New("Priority NoWeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			name:     "valid weight priority",
			policy:   config.Policy{Priorities: []config.PriorityPolicy{{Name: "WeightPriority", Weight: 2}}},
			expected: nil,
		},
		{
			name:     "invalid negative weight policy",
			policy:   config.Policy{Priorities: []config.PriorityPolicy{{Name: "WeightPriority", Weight: -2}}},
			expected: errors.New("Priority WeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			name:     "policy weight exceeds maximum",
			policy:   config.Policy{Priorities: []config.PriorityPolicy{{Name: "WeightPriority", Weight: config.MaxWeight}}},
			expected: errors.New("Priority WeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			name:     "valid weight in policy extender config",
			policy:   config.Policy{Extenders: []config.Extender{{URLPrefix: "http://127.0.0.1:8081/extender", PrioritizeVerb: "prioritize", Weight: 2}}},
			expected: nil,
		},
		{
			name:     "invalid negative weight in policy extender config",
			policy:   config.Policy{Extenders: []config.Extender{{URLPrefix: "http://127.0.0.1:8081/extender", PrioritizeVerb: "prioritize", Weight: -2}}},
			expected: errors.New("Priority for extender http://127.0.0.1:8081/extender should have a positive weight applied to it"),
		},
		{
			name:     "valid filter verb and url prefix",
			policy:   config.Policy{Extenders: []config.Extender{{URLPrefix: "http://127.0.0.1:8081/extender", FilterVerb: "filter"}}},
			expected: nil,
		},
		{
			name:     "valid preemt verb and urlprefix",
			policy:   config.Policy{Extenders: []config.Extender{{URLPrefix: "http://127.0.0.1:8081/extender", PreemptVerb: "preempt"}}},
			expected: nil,
		},
		{
			name: "invalid multiple extenders",
			policy: config.Policy{
				Extenders: []config.Extender{
					{URLPrefix: "http://127.0.0.1:8081/extender", BindVerb: "bind"},
					{URLPrefix: "http://127.0.0.1:8082/extender", BindVerb: "bind"},
				}},
			expected: errors.New("Only one extender can implement bind, found 2"),
		},
		{
			name: "invalid duplicate extender resource name",
			policy: config.Policy{
				Extenders: []config.Extender{
					{URLPrefix: "http://127.0.0.1:8081/extender", ManagedResources: []config.ExtenderManagedResource{{Name: "foo.com/bar"}}},
					{URLPrefix: "http://127.0.0.1:8082/extender", BindVerb: "bind", ManagedResources: []config.ExtenderManagedResource{{Name: "foo.com/bar"}}},
				}},
			expected: errors.New("Duplicate extender managed resource name foo.com/bar"),
		},
		{
			name: "invalid extended resource name",
			policy: config.Policy{
				Extenders: []config.Extender{
					{URLPrefix: "http://127.0.0.1:8081/extender", ManagedResources: []config.ExtenderManagedResource{{Name: "kubernetes.io/foo"}}},
				}},
			expected: errors.New("kubernetes.io/foo is an invalid extended resource name"),
		},
		{
			name: "invalid redeclared RequestedToCapacityRatio custom priority",
			policy: config.Policy{
				Priorities: []config.PriorityPolicy{
					{Name: "customPriority1", Weight: 1, Argument: &config.PriorityArgument{RequestedToCapacityRatioArguments: &config.RequestedToCapacityRatioArguments{}}},
					{Name: "customPriority2", Weight: 1, Argument: &config.PriorityArgument{RequestedToCapacityRatioArguments: &config.RequestedToCapacityRatioArguments{}}},
				},
			},
			expected: errors.New("Priority \"customPriority2\" redeclares custom priority \"RequestedToCapacityRatio\", from:\"customPriority1\""),
		},
		{
			name: "different weights for LabelPreference custom priority",
			policy: config.Policy{
				Priorities: []config.PriorityPolicy{
					{Name: "customPriority1", Weight: 1, Argument: &config.PriorityArgument{LabelPreference: &config.LabelPreference{}}},
					{Name: "customPriority2", Weight: 2, Argument: &config.PriorityArgument{LabelPreference: &config.LabelPreference{}}},
				},
			},
			expected: errors.New("LabelPreference  priority \"customPriority2\" has a different weight with \"customPriority1\""),
		},
		{
			name: "different weights for ServiceAntiAffinity custom priority",
			policy: config.Policy{
				Priorities: []config.PriorityPolicy{
					{Name: "customPriority1", Weight: 1, Argument: &config.PriorityArgument{ServiceAntiAffinity: &config.ServiceAntiAffinity{}}},
					{Name: "customPriority2", Weight: 2, Argument: &config.PriorityArgument{ServiceAntiAffinity: &config.ServiceAntiAffinity{}}},
				},
			},
			expected: errors.New("ServiceAntiAffinity  priority \"customPriority2\" has a different weight with \"customPriority1\""),
		},
		{
			name: "invalid hardPodAffinitySymmetricWeight, above the range",
			policy: config.Policy{
				HardPodAffinitySymmetricWeight: 101,
			},
			expected: errors.New("hardPodAffinitySymmetricWeight: Invalid value: 101: not in valid range [0-100]"),
		},
		{
			name: "invalid hardPodAffinitySymmetricWeight, below the range",
			policy: config.Policy{
				HardPodAffinitySymmetricWeight: -1,
			},
			expected: errors.New("hardPodAffinitySymmetricWeight: Invalid value: -1: not in valid range [0-100]"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := ValidatePolicy(test.policy)
			if fmt.Sprint(test.expected) != fmt.Sprint(actual) {
				t.Errorf("expected: %s, actual: %s", test.expected, actual)
			}
		})
	}
}
