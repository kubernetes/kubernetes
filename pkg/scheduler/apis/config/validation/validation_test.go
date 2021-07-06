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
	"k8s.io/kubernetes/pkg/scheduler/apis/config/v1beta1"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/v1beta2"
)

func TestValidateKubeSchedulerConfiguration(t *testing.T) {
	podInitialBackoffSeconds := int64(1)
	podMaxBackoffSeconds := int64(1)
	validConfig := &config.KubeSchedulerConfiguration{
		TypeMeta: metav1.TypeMeta{
			APIVersion: v1beta2.SchemeGroupVersion.String(),
		},
		Parallelism:        8,
		HealthzBindAddress: "0.0.0.0:10254",
		MetricsBindAddress: "0.0.0.0:10254",
		ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
			AcceptContentTypes: "application/json",
			ContentType:        "application/json",
			QPS:                10,
			Burst:              10,
		},
		LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
			ResourceLock:      "configmap",
			LeaderElect:       true,
			LeaseDuration:     metav1.Duration{Duration: 30 * time.Second},
			RenewDeadline:     metav1.Duration{Duration: 15 * time.Second},
			RetryPeriod:       metav1.Duration{Duration: 5 * time.Second},
			ResourceNamespace: "name",
			ResourceName:      "name",
		},
		PodInitialBackoffSeconds: podInitialBackoffSeconds,
		PodMaxBackoffSeconds:     podMaxBackoffSeconds,
		PercentageOfNodesToScore: 35,
		Profiles: []config.KubeSchedulerProfile{
			{
				SchedulerName: "me",
				Plugins: &config.Plugins{
					QueueSort: config.PluginSet{
						Enabled: []config.Plugin{{Name: "CustomSort"}},
					},
					Score: config.PluginSet{
						Disabled: []config.Plugin{{Name: "*"}},
					},
				},
				PluginConfig: []config.PluginConfig{
					{
						Name: "DefaultPreemption",
						Args: &config.DefaultPreemptionArgs{MinCandidateNodesPercentage: 10, MinCandidateNodesAbsolute: 100},
					},
				},
			},
			{
				SchedulerName: "other",
				Plugins: &config.Plugins{
					QueueSort: config.PluginSet{
						Enabled: []config.Plugin{{Name: "CustomSort"}},
					},
					Bind: config.PluginSet{
						Enabled: []config.Plugin{{Name: "CustomBind"}},
					},
				},
			},
		},
		Extenders: []config.Extender{
			{
				PrioritizeVerb: "prioritize",
				Weight:         1,
			},
		},
	}

	invalidParallelismValue := validConfig.DeepCopy()
	invalidParallelismValue.Parallelism = 0

	resourceNameNotSet := validConfig.DeepCopy()
	resourceNameNotSet.LeaderElection.ResourceName = ""

	resourceNamespaceNotSet := validConfig.DeepCopy()
	resourceNamespaceNotSet.LeaderElection.ResourceNamespace = ""

	metricsBindAddrHostInvalid := validConfig.DeepCopy()
	metricsBindAddrHostInvalid.MetricsBindAddress = "0.0.0.0.0:9090"

	metricsBindAddrPortInvalid := validConfig.DeepCopy()
	metricsBindAddrPortInvalid.MetricsBindAddress = "0.0.0.0:909090"

	metricsBindAddrHostOnlyInvalid := validConfig.DeepCopy()
	metricsBindAddrHostOnlyInvalid.MetricsBindAddress = "999.999.999.999"

	healthzBindAddrHostInvalid := validConfig.DeepCopy()
	healthzBindAddrHostInvalid.HealthzBindAddress = "0.0.0.0.0:9090"

	healthzBindAddrPortInvalid := validConfig.DeepCopy()
	healthzBindAddrPortInvalid.HealthzBindAddress = "0.0.0.0:909090"

	healthzBindAddrHostOnlyInvalid := validConfig.DeepCopy()
	healthzBindAddrHostOnlyInvalid.HealthzBindAddress = "999.999.999.999"

	enableContentProfilingSetWithoutEnableProfiling := validConfig.DeepCopy()
	enableContentProfilingSetWithoutEnableProfiling.EnableProfiling = false
	enableContentProfilingSetWithoutEnableProfiling.EnableContentionProfiling = true

	percentageOfNodesToScore101 := validConfig.DeepCopy()
	percentageOfNodesToScore101.PercentageOfNodesToScore = int32(101)

	schedulerNameNotSet := validConfig.DeepCopy()
	schedulerNameNotSet.Profiles[1].SchedulerName = ""

	repeatedSchedulerName := validConfig.DeepCopy()
	repeatedSchedulerName.Profiles[0].SchedulerName = "other"

	differentQueueSort := validConfig.DeepCopy()
	differentQueueSort.Profiles[1].Plugins.QueueSort.Enabled[0].Name = "AnotherSort"

	oneEmptyQueueSort := validConfig.DeepCopy()
	oneEmptyQueueSort.Profiles[0].Plugins = nil

	extenderNegativeWeight := validConfig.DeepCopy()
	extenderNegativeWeight.Extenders[0].Weight = -1

	invalidNodePercentage := validConfig.DeepCopy()
	invalidNodePercentage.Profiles[0].PluginConfig = []config.PluginConfig{
		{
			Name: "DefaultPreemption",
			Args: &config.DefaultPreemptionArgs{MinCandidateNodesPercentage: 200, MinCandidateNodesAbsolute: 100},
		},
	}

	invalidPluginArgs := validConfig.DeepCopy()
	invalidPluginArgs.Profiles[0].PluginConfig = []config.PluginConfig{
		{
			Name: "DefaultPreemption",
			Args: &config.InterPodAffinityArgs{},
		},
	}

	duplicatedPluginConfig := validConfig.DeepCopy()
	duplicatedPluginConfig.Profiles[0].PluginConfig = []config.PluginConfig{
		{
			Name: "config",
		},
		{
			Name: "config",
		},
	}

	mismatchQueueSort := validConfig.DeepCopy()
	mismatchQueueSort.Profiles = []config.KubeSchedulerProfile{
		{
			SchedulerName: "me",
			Plugins: &config.Plugins{
				QueueSort: config.PluginSet{
					Enabled: []config.Plugin{{Name: "PrioritySort"}},
				},
			},
			PluginConfig: []config.PluginConfig{
				{
					Name: "PrioritySort",
				},
			},
		},
		{
			SchedulerName: "other",
			Plugins: &config.Plugins{
				QueueSort: config.PluginSet{
					Enabled: []config.Plugin{{Name: "CustomSort"}},
				},
			},
			PluginConfig: []config.PluginConfig{
				{
					Name: "CustomSort",
				},
			},
		},
	}

	extenderDuplicateManagedResource := validConfig.DeepCopy()
	extenderDuplicateManagedResource.Extenders[0].ManagedResources = []config.ExtenderManagedResource{
		{Name: "foo", IgnoredByScheduler: false},
		{Name: "foo", IgnoredByScheduler: false},
	}

	extenderDuplicateBind := validConfig.DeepCopy()
	extenderDuplicateBind.Extenders[0].BindVerb = "foo"
	extenderDuplicateBind.Extenders = append(extenderDuplicateBind.Extenders, config.Extender{
		PrioritizeVerb: "prioritize",
		BindVerb:       "bar",
	})

	badRemovedPlugins1 := validConfig.DeepCopy() // default v1beta2
	badRemovedPlugins1.Profiles[0].Plugins.Score.Enabled = append(badRemovedPlugins1.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "ServiceAffinity", Weight: 2})

	badRemovedPlugins2 := validConfig.DeepCopy()
	badRemovedPlugins2.APIVersion = "kubescheduler.config.k8s.io/v1beta3" // hypothetical, v1beta3 doesn't exist
	badRemovedPlugins2.Profiles[0].Plugins.Score.Enabled = append(badRemovedPlugins2.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "ServiceAffinity", Weight: 2})

	badRemovedPlugins3 := validConfig.DeepCopy() // default v1beta2
	badRemovedPlugins3.Profiles[0].Plugins.Score.Enabled = append(badRemovedPlugins3.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "NodeResourcesMostAllocated", Weight: 2})

	goodRemovedPlugins1 := validConfig.DeepCopy() // ServiceAffinity is okay in v1beta1.
	goodRemovedPlugins1.APIVersion = v1beta1.SchemeGroupVersion.String()
	goodRemovedPlugins1.Profiles[0].Plugins.Score.Enabled = append(goodRemovedPlugins1.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "ServiceAffinity", Weight: 2})

	goodRemovedPlugins2 := validConfig.DeepCopy()
	goodRemovedPlugins2.Profiles[0].Plugins.Score.Enabled = append(goodRemovedPlugins2.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "PodTopologySpread", Weight: 2})

	goodRemovedPlugins3 := validConfig.DeepCopy()
	goodRemovedPlugins3.APIVersion = v1beta1.SchemeGroupVersion.String()
	goodRemovedPlugins3.Profiles[0].Plugins.Score.Enabled = append(goodRemovedPlugins3.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "NodeResourcesMostAllocated", Weight: 2})

	badConflictPlugins1 := validConfig.DeepCopy()
	badConflictPlugins1.APIVersion = v1beta1.SchemeGroupVersion.String()
	badConflictPlugins1.Profiles[0].Plugins.Score.Enabled = append(badConflictPlugins1.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "NodeResourcesFit", Weight: 2})
	badConflictPlugins1.Profiles[0].Plugins.Score.Enabled = append(badConflictPlugins1.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "NodeResourcesLeastAllocated", Weight: 2})
	badConflictPlugins1.Profiles[0].Plugins.Score.Enabled = append(badConflictPlugins1.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "NodeResourcesMostAllocated", Weight: 2})
	badConflictPlugins1.Profiles[0].Plugins.Score.Enabled = append(badConflictPlugins1.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "RequestedToCapacityRatio", Weight: 2})

	goodConflictPlugins1 := validConfig.DeepCopy()
	goodConflictPlugins1.APIVersion = v1beta1.SchemeGroupVersion.String()
	goodConflictPlugins1.Profiles[0].Plugins.Score.Enabled = append(goodConflictPlugins1.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "NodeResourcesLeastAllocated", Weight: 2})
	goodConflictPlugins1.Profiles[0].Plugins.Score.Enabled = append(goodConflictPlugins1.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "NodeResourcesMostAllocated", Weight: 2})
	goodConflictPlugins1.Profiles[0].Plugins.Score.Enabled = append(goodConflictPlugins1.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "RequestedToCapacityRatio", Weight: 2})

	goodConflictPlugins2 := validConfig.DeepCopy()
	goodConflictPlugins2.APIVersion = v1beta1.SchemeGroupVersion.String()
	goodConflictPlugins2.Profiles[0].Plugins.Filter.Enabled = append(goodConflictPlugins2.Profiles[0].Plugins.Filter.Enabled, config.Plugin{Name: "NodeResourcesFit", Weight: 2})
	goodConflictPlugins2.Profiles[0].Plugins.Score.Enabled = append(goodConflictPlugins2.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "NodeResourcesLeastAllocated", Weight: 2})
	goodConflictPlugins2.Profiles[0].Plugins.Score.Enabled = append(goodConflictPlugins2.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "NodeResourcesMostAllocated", Weight: 2})
	goodConflictPlugins2.Profiles[0].Plugins.Score.Enabled = append(goodConflictPlugins2.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "RequestedToCapacityRatio", Weight: 2})

	deprecatedPluginsConfig := validConfig.DeepCopy()
	deprecatedPluginsConfig.Profiles[0].PluginConfig = append(deprecatedPluginsConfig.Profiles[0].PluginConfig, config.PluginConfig{
		Name: "NodeResourcesLeastAllocated",
		Args: &config.NodeResourcesLeastAllocatedArgs{},
	})

	scenarios := map[string]struct {
		expectedToFail bool
		config         *config.KubeSchedulerConfiguration
		errorString    string
	}{
		"good": {
			expectedToFail: false,
			config:         validConfig,
		},
		"bad-parallelism-invalid-value": {
			expectedToFail: true,
			config:         invalidParallelismValue,
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
		"bad-healthz-host-only-invalid": {
			expectedToFail: true,
			config:         healthzBindAddrHostOnlyInvalid,
		},
		"bad-metrics-port-invalid": {
			expectedToFail: true,
			config:         metricsBindAddrPortInvalid,
		},
		"bad-metrics-host-invalid": {
			expectedToFail: true,
			config:         metricsBindAddrHostInvalid,
		},
		"bad-metrics-host-only-invalid": {
			expectedToFail: true,
			config:         metricsBindAddrHostOnlyInvalid,
		},
		"bad-percentage-of-nodes-to-score": {
			expectedToFail: true,
			config:         percentageOfNodesToScore101,
		},
		"scheduler-name-not-set": {
			expectedToFail: true,
			config:         schedulerNameNotSet,
		},
		"repeated-scheduler-name": {
			expectedToFail: true,
			config:         repeatedSchedulerName,
		},
		"different-queue-sort": {
			expectedToFail: true,
			config:         differentQueueSort,
		},
		"one-empty-queue-sort": {
			expectedToFail: true,
			config:         oneEmptyQueueSort,
		},
		"extender-negative-weight": {
			expectedToFail: true,
			config:         extenderNegativeWeight,
		},
		"extender-duplicate-managed-resources": {
			expectedToFail: true,
			config:         extenderDuplicateManagedResource,
		},
		"extender-duplicate-bind": {
			expectedToFail: true,
			config:         extenderDuplicateBind,
		},
		"invalid-node-percentage": {
			expectedToFail: true,
			config:         invalidNodePercentage,
		},
		"invalid-plugin-args": {
			expectedToFail: true,
			config:         invalidPluginArgs,
		},
		"duplicated-plugin-config": {
			expectedToFail: true,
			config:         duplicatedPluginConfig,
		},
		"mismatch-queue-sort": {
			expectedToFail: true,
			config:         mismatchQueueSort,
		},
		"bad-removed-plugins-1": {
			expectedToFail: true,
			config:         badRemovedPlugins1,
		},
		"bad-removed-plugins-2": {
			expectedToFail: true,
			config:         badRemovedPlugins2,
		},
		"bad-removed-plugins-3": {
			expectedToFail: true,
			config:         badRemovedPlugins3,
		},
		"good-removed-plugins-1": {
			expectedToFail: false,
			config:         goodRemovedPlugins1,
		},
		"good-removed-plugins-2": {
			expectedToFail: false,
			config:         goodRemovedPlugins2,
		},
		"good-removed-plugins-3": {
			expectedToFail: false,
			config:         goodRemovedPlugins3,
		},
		"bad-conflict-plugins-1": {
			expectedToFail: true,
			config:         badConflictPlugins1,
			errorString:    "profiles[0].plugins.score.enabled[0]: Invalid value: \"NodeResourcesFit\": was conflict with [\"NodeResourcesLeastAllocated\" \"NodeResourcesMostAllocated\" \"RequestedToCapacityRatio\"] in version \"kubescheduler.config.k8s.io/v1beta1\" (KubeSchedulerConfiguration is version \"kubescheduler.config.k8s.io/v1beta1\")",
		},
		"good-conflict-plugins-1": {
			expectedToFail: false,
			config:         goodConflictPlugins1,
		},
		"good-conflict-plugins-2": {
			expectedToFail: false,
			config:         goodConflictPlugins2,
		},
		"bad-plugins-config": {
			expectedToFail: true,
			config:         deprecatedPluginsConfig,
			errorString:    "profiles[0].pluginConfig[1]: Invalid value: \"NodeResourcesLeastAllocated\": was removed in version \"kubescheduler.config.k8s.io/v1beta2\" (KubeSchedulerConfiguration is version \"kubescheduler.config.k8s.io/v1beta2\")",
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateKubeSchedulerConfiguration(scenario.config)
			if errs == nil && scenario.expectedToFail {
				t.Error("Unexpected success")
			}
			if errs != nil && !scenario.expectedToFail {
				t.Errorf("Unexpected failure: %+v", errs)
			}
			fmt.Println(errs)

			if errs != nil && scenario.errorString != "" && errs.Error() != scenario.errorString {
				t.Errorf("Unexpected error string\n want:\t%s\n got:\t%s", scenario.errorString, errs.Error())
			}
		})
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
			expected: errors.New("priority NoWeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			name:     "policy weight is not positive",
			policy:   config.Policy{Priorities: []config.PriorityPolicy{{Name: "NoWeightPriority", Weight: 0}}},
			expected: errors.New("priority NoWeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			name:     "valid weight priority",
			policy:   config.Policy{Priorities: []config.PriorityPolicy{{Name: "WeightPriority", Weight: 2}}},
			expected: nil,
		},
		{
			name:     "invalid negative weight policy",
			policy:   config.Policy{Priorities: []config.PriorityPolicy{{Name: "WeightPriority", Weight: -2}}},
			expected: errors.New("priority WeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			name:     "policy weight exceeds maximum",
			policy:   config.Policy{Priorities: []config.PriorityPolicy{{Name: "WeightPriority", Weight: config.MaxWeight}}},
			expected: errors.New("priority WeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			name:     "valid weight in policy extender config",
			policy:   config.Policy{Extenders: []config.Extender{{URLPrefix: "http://127.0.0.1:8081/extender", PrioritizeVerb: "prioritize", Weight: 2}}},
			expected: nil,
		},
		{
			name:     "invalid negative weight in policy extender config",
			policy:   config.Policy{Extenders: []config.Extender{{URLPrefix: "http://127.0.0.1:8081/extender", PrioritizeVerb: "prioritize", Weight: -2}}},
			expected: errors.New("extenders[0].weight: Invalid value: -2: must have a positive weight applied to it"),
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
			expected: errors.New("extenders: Invalid value: \"found 2 extenders implementing bind\": only one extender can implement bind"),
		},
		{
			name: "invalid duplicate extender resource name",
			policy: config.Policy{
				Extenders: []config.Extender{
					{URLPrefix: "http://127.0.0.1:8081/extender", ManagedResources: []config.ExtenderManagedResource{{Name: "foo.com/bar"}}},
					{URLPrefix: "http://127.0.0.1:8082/extender", BindVerb: "bind", ManagedResources: []config.ExtenderManagedResource{{Name: "foo.com/bar"}}},
				}},
			expected: errors.New("extenders[1].managedResources[0].name: Invalid value: \"foo.com/bar\": duplicate extender managed resource name"),
		},
		{
			name: "invalid extended resource name",
			policy: config.Policy{
				Extenders: []config.Extender{
					{URLPrefix: "http://127.0.0.1:8081/extender", ManagedResources: []config.ExtenderManagedResource{{Name: "kubernetes.io/foo"}}},
				}},
			expected: errors.New("extenders[0].managedResources[0].name: Invalid value: \"kubernetes.io/foo\": is an invalid extended resource name"),
		},
		{
			name: "invalid redeclared RequestedToCapacityRatio custom priority",
			policy: config.Policy{
				Priorities: []config.PriorityPolicy{
					{Name: "customPriority1", Weight: 1, Argument: &config.PriorityArgument{RequestedToCapacityRatioArguments: &config.RequestedToCapacityRatioArguments{}}},
					{Name: "customPriority2", Weight: 1, Argument: &config.PriorityArgument{RequestedToCapacityRatioArguments: &config.RequestedToCapacityRatioArguments{}}},
				},
			},
			expected: errors.New("priority \"customPriority2\" redeclares custom priority \"RequestedToCapacityRatio\", from: \"customPriority1\""),
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
