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
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	componentbaseconfig "k8s.io/component-base/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	configv1 "k8s.io/kubernetes/pkg/scheduler/apis/config/v1"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/v1beta2"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/v1beta3"
	"k8s.io/utils/pointer"
)

func TestValidateKubeSchedulerConfigurationV1beta2(t *testing.T) {
	podInitialBackoffSeconds := int64(1)
	podMaxBackoffSeconds := int64(1)
	validConfig := &config.KubeSchedulerConfiguration{
		TypeMeta: metav1.TypeMeta{
			APIVersion: v1beta2.SchemeGroupVersion.String(),
		},
		Parallelism: 8,
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
		PercentageOfNodesToScore: pointer.Int32(35),
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

	enableContentProfilingSetWithoutEnableProfiling := validConfig.DeepCopy()
	enableContentProfilingSetWithoutEnableProfiling.EnableProfiling = false
	enableContentProfilingSetWithoutEnableProfiling.EnableContentionProfiling = true

	metricsBindAddrInvalid := validConfig.DeepCopy()
	metricsBindAddrInvalid.MetricsBindAddress = "0.0.0.0:9090"

	healthzBindAddrInvalid := validConfig.DeepCopy()
	healthzBindAddrInvalid.HealthzBindAddress = "0.0.0.0:9090"

	percentageOfNodesToScore101 := validConfig.DeepCopy()
	percentageOfNodesToScore101.PercentageOfNodesToScore = pointer.Int32(101)

	percentageOfNodesToScoreNegative := validConfig.DeepCopy()
	percentageOfNodesToScoreNegative.PercentageOfNodesToScore = pointer.Int32(-1)

	schedulerNameNotSet := validConfig.DeepCopy()
	schedulerNameNotSet.Profiles[1].SchedulerName = ""

	repeatedSchedulerName := validConfig.DeepCopy()
	repeatedSchedulerName.Profiles[0].SchedulerName = "other"

	profilePercentageOfNodesToScore101 := validConfig.DeepCopy()
	profilePercentageOfNodesToScore101.Profiles[1].PercentageOfNodesToScore = pointer.Int32(101)

	profilePercentageOfNodesToScoreNegative := validConfig.DeepCopy()
	profilePercentageOfNodesToScoreNegative.Profiles[1].PercentageOfNodesToScore = pointer.Int32(-1)

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
		Weight:         1,
	})

	validPlugins := validConfig.DeepCopy()
	validPlugins.Profiles[0].Plugins.Score.Enabled = append(validPlugins.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "PodTopologySpread", Weight: 2})

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
			errorString:    "should be an integer value greater than zero",
		},
		"bad-resource-name-not-set": {
			expectedToFail: true,
			config:         resourceNameNotSet,
			errorString:    "resourceName is required",
		},
		"bad-resource-namespace-not-set": {
			expectedToFail: true,
			config:         resourceNamespaceNotSet,
			errorString:    "resourceNamespace is required",
		},
		"non-empty-metrics-bind-addr": {
			expectedToFail: true,
			config:         metricsBindAddrInvalid,
			errorString:    "must be empty or with an explicit 0 port",
		},
		"non-empty-healthz-bind-addr": {
			expectedToFail: true,
			config:         healthzBindAddrInvalid,
			errorString:    "must be empty or with an explicit 0 port",
		},
		"greater-than-100-percentage-of-nodes-to-score": {
			expectedToFail: true,
			config:         percentageOfNodesToScore101,
			errorString:    "not in valid range [0-100]",
		},
		"negative-percentage-of-nodes-to-score": {
			expectedToFail: true,
			config:         percentageOfNodesToScoreNegative,
			errorString:    "not in valid range [0-100]",
		},
		"scheduler-name-not-set": {
			expectedToFail: true,
			config:         schedulerNameNotSet,
			errorString:    "Required value",
		},
		"repeated-scheduler-name": {
			expectedToFail: true,
			config:         repeatedSchedulerName,
			errorString:    "Duplicate value",
		},
		"greater-than-100-profile-percentage-of-nodes-to-score": {
			expectedToFail: true,
			config:         profilePercentageOfNodesToScore101,
			errorString:    "not in valid range [0-100]",
		},
		"negative-100-profile-percentage-of-nodes-to-score": {
			expectedToFail: true,
			config:         profilePercentageOfNodesToScoreNegative,
			errorString:    "not in valid range [0-100]",
		},
		"different-queue-sort": {
			expectedToFail: true,
			config:         differentQueueSort,
			errorString:    "has to match for all profiles",
		},
		"one-empty-queue-sort": {
			expectedToFail: true,
			config:         oneEmptyQueueSort,
			errorString:    "has to match for all profiles",
		},
		"extender-negative-weight": {
			expectedToFail: true,
			config:         extenderNegativeWeight,
			errorString:    "must have a positive weight applied to it",
		},
		"extender-duplicate-managed-resources": {
			expectedToFail: true,
			config:         extenderDuplicateManagedResource,
			errorString:    "duplicate extender managed resource name",
		},
		"extender-duplicate-bind": {
			expectedToFail: true,
			config:         extenderDuplicateBind,
			errorString:    "only one extender can implement bind",
		},
		"invalid-node-percentage": {
			expectedToFail: true,
			config:         invalidNodePercentage,
			errorString:    "not in valid range [0, 100]",
		},
		"invalid-plugin-args": {
			expectedToFail: true,
			config:         invalidPluginArgs,
			errorString:    "has to match plugin args",
		},
		"duplicated-plugin-config": {
			expectedToFail: true,
			config:         duplicatedPluginConfig,
			errorString:    "Duplicate value: \"config\"",
		},
		"mismatch-queue-sort": {
			expectedToFail: true,
			config:         mismatchQueueSort,
			errorString:    "has to match for all profiles",
		},
		"valid-plugins": {
			expectedToFail: false,
			config:         validPlugins,
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

			if errs != nil && scenario.errorString != "" && !strings.Contains(errs.Error(), scenario.errorString) {
				t.Errorf("Unexpected error string\n want:\t%s\n got:\t%s", scenario.errorString, errs.Error())
			}
		})
	}
}

func TestValidateKubeSchedulerConfigurationV1beta3(t *testing.T) {
	podInitialBackoffSeconds := int64(1)
	podMaxBackoffSeconds := int64(1)
	validConfig := &config.KubeSchedulerConfiguration{
		TypeMeta: metav1.TypeMeta{
			APIVersion: v1beta3.SchemeGroupVersion.String(),
		},
		Parallelism: 8,
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
		PercentageOfNodesToScore: pointer.Int32(35),
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

	enableContentProfilingSetWithoutEnableProfiling := validConfig.DeepCopy()
	enableContentProfilingSetWithoutEnableProfiling.EnableProfiling = false
	enableContentProfilingSetWithoutEnableProfiling.EnableContentionProfiling = true

	metricsBindAddrInvalid := validConfig.DeepCopy()
	metricsBindAddrInvalid.MetricsBindAddress = "0.0.0.0:9090"

	healthzBindAddrInvalid := validConfig.DeepCopy()
	healthzBindAddrInvalid.HealthzBindAddress = "0.0.0.0:9090"

	percentageOfNodesToScore101 := validConfig.DeepCopy()
	percentageOfNodesToScore101.PercentageOfNodesToScore = pointer.Int32(101)

	percentageOfNodesToScoreNegative := validConfig.DeepCopy()
	percentageOfNodesToScoreNegative.Profiles[1].PercentageOfNodesToScore = pointer.Int32(-1)

	schedulerNameNotSet := validConfig.DeepCopy()
	schedulerNameNotSet.Profiles[1].SchedulerName = ""

	repeatedSchedulerName := validConfig.DeepCopy()
	repeatedSchedulerName.Profiles[0].SchedulerName = "other"

	profilePercentageOfNodesToScore101 := validConfig.DeepCopy()
	profilePercentageOfNodesToScore101.Profiles[1].PercentageOfNodesToScore = pointer.Int32(101)

	profilePercentageOfNodesToScoreNegative := validConfig.DeepCopy()
	profilePercentageOfNodesToScoreNegative.Profiles[1].PercentageOfNodesToScore = pointer.Int32(-1)

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
		Weight:         1,
	})

	validPlugins := validConfig.DeepCopy()
	validPlugins.Profiles[0].Plugins.Score.Enabled = append(validPlugins.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "PodTopologySpread", Weight: 2})

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
			errorString:    "should be an integer value greater than zero",
		},
		"bad-resource-name-not-set": {
			expectedToFail: true,
			config:         resourceNameNotSet,
			errorString:    "resourceName is required",
		},
		"bad-resource-namespace-not-set": {
			expectedToFail: true,
			config:         resourceNamespaceNotSet,
			errorString:    "resourceNamespace is required",
		},
		"non-empty-metrics-bind-addr": {
			expectedToFail: true,
			config:         metricsBindAddrInvalid,
			errorString:    "must be empty or with an explicit 0 port",
		},
		"non-empty-healthz-bind-addr": {
			expectedToFail: true,
			config:         healthzBindAddrInvalid,
			errorString:    "must be empty or with an explicit 0 port",
		},
		"bad-percentage-of-nodes-to-score": {
			expectedToFail: true,
			config:         percentageOfNodesToScore101,
			errorString:    "not in valid range [0-100]",
		},
		"negative-percentage-of-nodes-to-score": {
			expectedToFail: true,
			config:         percentageOfNodesToScoreNegative,
			errorString:    "not in valid range [0-100]",
		},
		"scheduler-name-not-set": {
			expectedToFail: true,
			config:         schedulerNameNotSet,
			errorString:    "Required value",
		},
		"repeated-scheduler-name": {
			expectedToFail: true,
			config:         repeatedSchedulerName,
			errorString:    "Duplicate value",
		},
		"greater-than-100-profile-percentage-of-nodes-to-score": {
			expectedToFail: true,
			config:         profilePercentageOfNodesToScore101,
			errorString:    "not in valid range [0-100]",
		},
		"negative-100-profile-percentage-of-nodes-to-score": {
			expectedToFail: true,
			config:         profilePercentageOfNodesToScoreNegative,
			errorString:    "not in valid range [0-100]",
		},
		"different-queue-sort": {
			expectedToFail: true,
			config:         differentQueueSort,
			errorString:    "has to match for all profiles",
		},
		"one-empty-queue-sort": {
			expectedToFail: true,
			config:         oneEmptyQueueSort,
			errorString:    "has to match for all profiles",
		},
		"extender-negative-weight": {
			expectedToFail: true,
			config:         extenderNegativeWeight,
			errorString:    "must have a positive weight applied to it",
		},
		"extender-duplicate-managed-resources": {
			expectedToFail: true,
			config:         extenderDuplicateManagedResource,
			errorString:    "duplicate extender managed resource name",
		},
		"extender-duplicate-bind": {
			expectedToFail: true,
			config:         extenderDuplicateBind,
			errorString:    "only one extender can implement bind",
		},
		"invalid-node-percentage": {
			expectedToFail: true,
			config:         invalidNodePercentage,
			errorString:    "not in valid range [0, 100]",
		},
		"invalid-plugin-args": {
			expectedToFail: true,
			config:         invalidPluginArgs,
			errorString:    "has to match plugin args",
		},
		"duplicated-plugin-config": {
			expectedToFail: true,
			config:         duplicatedPluginConfig,
			errorString:    "Duplicate value: \"config\"",
		},
		"mismatch-queue-sort": {
			expectedToFail: true,
			config:         mismatchQueueSort,
			errorString:    "has to match for all profiles",
		},
		"valid-plugins": {
			expectedToFail: false,
			config:         validPlugins,
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

			if errs != nil && scenario.errorString != "" && !strings.Contains(errs.Error(), scenario.errorString) {
				t.Errorf("Unexpected error string\n want:\t%s\n got:\t%s", scenario.errorString, errs.Error())
			}
		})
	}
}

func TestValidateKubeSchedulerConfigurationV1(t *testing.T) {
	podInitialBackoffSeconds := int64(1)
	podMaxBackoffSeconds := int64(1)
	validConfig := &config.KubeSchedulerConfiguration{
		TypeMeta: metav1.TypeMeta{
			APIVersion: configv1.SchemeGroupVersion.String(),
		},
		Parallelism: 8,
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
		Profiles: []config.KubeSchedulerProfile{
			{
				SchedulerName:            "me",
				PercentageOfNodesToScore: pointer.Int32(35),
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
				SchedulerName:            "other",
				PercentageOfNodesToScore: pointer.Int32(35),
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

	enableContentProfilingSetWithoutEnableProfiling := validConfig.DeepCopy()
	enableContentProfilingSetWithoutEnableProfiling.EnableProfiling = false
	enableContentProfilingSetWithoutEnableProfiling.EnableContentionProfiling = true

	metricsBindAddrInvalid := validConfig.DeepCopy()
	metricsBindAddrInvalid.MetricsBindAddress = "0.0.0.0:9090"

	healthzBindAddrInvalid := validConfig.DeepCopy()
	healthzBindAddrInvalid.HealthzBindAddress = "0.0.0.0:9090"

	percentageOfNodesToScore101 := validConfig.DeepCopy()
	percentageOfNodesToScore101.PercentageOfNodesToScore = pointer.Int32(101)

	percentageOfNodesToScoreNegative := validConfig.DeepCopy()
	percentageOfNodesToScoreNegative.PercentageOfNodesToScore = pointer.Int32(-1)

	schedulerNameNotSet := validConfig.DeepCopy()
	schedulerNameNotSet.Profiles[1].SchedulerName = ""

	repeatedSchedulerName := validConfig.DeepCopy()
	repeatedSchedulerName.Profiles[0].SchedulerName = "other"

	profilePercentageOfNodesToScore101 := validConfig.DeepCopy()
	profilePercentageOfNodesToScore101.Profiles[1].PercentageOfNodesToScore = pointer.Int32(101)

	profilePercentageOfNodesToScoreNegative := validConfig.DeepCopy()
	profilePercentageOfNodesToScoreNegative.Profiles[1].PercentageOfNodesToScore = pointer.Int32(-1)

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
		Weight:         1,
	})

	validPlugins := validConfig.DeepCopy()
	validPlugins.Profiles[0].Plugins.Score.Enabled = append(validPlugins.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "PodTopologySpread", Weight: 2})

	invalidPlugins := validConfig.DeepCopy()
	invalidPlugins.Profiles[0].Plugins.Score.Enabled = append(invalidPlugins.Profiles[0].Plugins.Score.Enabled, config.Plugin{Name: "SelectorSpread"})

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
			errorString:    "should be an integer value greater than zero",
		},
		"bad-resource-name-not-set": {
			expectedToFail: true,
			config:         resourceNameNotSet,
			errorString:    "resourceName is required",
		},
		"bad-resource-namespace-not-set": {
			expectedToFail: true,
			config:         resourceNamespaceNotSet,
			errorString:    "resourceNamespace is required",
		},
		"non-empty-metrics-bind-addr": {
			expectedToFail: true,
			config:         metricsBindAddrInvalid,
			errorString:    "must be empty or with an explicit 0 port",
		},
		"non-empty-healthz-bind-addr": {
			expectedToFail: true,
			config:         healthzBindAddrInvalid,
			errorString:    "must be empty or with an explicit 0 port",
		},
		"bad-percentage-of-nodes-to-score": {
			expectedToFail: true,
			config:         percentageOfNodesToScore101,
			errorString:    "not in valid range [0-100]",
		},
		"negative-percentage-of-nodes-to-score": {
			expectedToFail: true,
			config:         percentageOfNodesToScoreNegative,
			errorString:    "not in valid range [0-100]",
		},
		"scheduler-name-not-set": {
			expectedToFail: true,
			config:         schedulerNameNotSet,
			errorString:    "Required value",
		},
		"repeated-scheduler-name": {
			expectedToFail: true,
			config:         repeatedSchedulerName,
			errorString:    "Duplicate value",
		},
		"greater-than-100-profile-percentage-of-nodes-to-score": {
			expectedToFail: true,
			config:         profilePercentageOfNodesToScore101,
			errorString:    "not in valid range [0-100]",
		},
		"negative-profile-percentage-of-nodes-to-score": {
			expectedToFail: true,
			config:         profilePercentageOfNodesToScoreNegative,
			errorString:    "not in valid range [0-100]",
		},
		"different-queue-sort": {
			expectedToFail: true,
			config:         differentQueueSort,
			errorString:    "has to match for all profiles",
		},
		"one-empty-queue-sort": {
			expectedToFail: true,
			config:         oneEmptyQueueSort,
			errorString:    "has to match for all profiles",
		},
		"extender-negative-weight": {
			expectedToFail: true,
			config:         extenderNegativeWeight,
			errorString:    "must have a positive weight applied to it",
		},
		"extender-duplicate-managed-resources": {
			expectedToFail: true,
			config:         extenderDuplicateManagedResource,
			errorString:    "duplicate extender managed resource name",
		},
		"extender-duplicate-bind": {
			expectedToFail: true,
			config:         extenderDuplicateBind,
			errorString:    "only one extender can implement bind",
		},
		"invalid-node-percentage": {
			expectedToFail: true,
			config:         invalidNodePercentage,
			errorString:    "not in valid range [0, 100]",
		},
		"invalid-plugin-args": {
			expectedToFail: true,
			config:         invalidPluginArgs,
			errorString:    "has to match plugin args",
		},
		"duplicated-plugin-config": {
			expectedToFail: true,
			config:         duplicatedPluginConfig,
			errorString:    "Duplicate value: \"config\"",
		},
		"mismatch-queue-sort": {
			expectedToFail: true,
			config:         mismatchQueueSort,
			errorString:    "has to match for all profiles",
		},
		"valid-plugins": {
			expectedToFail: false,
			config:         validPlugins,
		},
		"invalid-plugins": {
			expectedToFail: true,
			config:         invalidPlugins,
			errorString:    "\"SelectorSpread\": was invalid",
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

			if errs != nil && scenario.errorString != "" && !strings.Contains(errs.Error(), scenario.errorString) {
				t.Errorf("Unexpected error string\n want:\t%s\n got:\t%s", scenario.errorString, errs.Error())
			}
		})
	}
}
