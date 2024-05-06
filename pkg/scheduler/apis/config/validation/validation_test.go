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

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	componentbaseconfig "k8s.io/component-base/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	configv1 "k8s.io/kubernetes/pkg/scheduler/apis/config/v1"
	"k8s.io/utils/ptr"
)

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
			ResourceLock:      "leases",
			LeaderElect:       true,
			LeaseDuration:     metav1.Duration{Duration: 30 * time.Second},
			RenewDeadline:     metav1.Duration{Duration: 15 * time.Second},
			RetryPeriod:       metav1.Duration{Duration: 5 * time.Second},
			ResourceNamespace: "name",
			ResourceName:      "name",
		},
		PodInitialBackoffSeconds: podInitialBackoffSeconds,
		PodMaxBackoffSeconds:     podMaxBackoffSeconds,
		Profiles: []config.KubeSchedulerProfile{{
			SchedulerName:            "me",
			PercentageOfNodesToScore: ptr.To[int32](35),
			Plugins: &config.Plugins{
				QueueSort: config.PluginSet{
					Enabled: []config.Plugin{{Name: "CustomSort"}},
				},
				Score: config.PluginSet{
					Disabled: []config.Plugin{{Name: "*"}},
				},
			},
			PluginConfig: []config.PluginConfig{{
				Name: "DefaultPreemption",
				Args: &config.DefaultPreemptionArgs{MinCandidateNodesPercentage: 10, MinCandidateNodesAbsolute: 100},
			}},
		}, {
			SchedulerName:            "other",
			PercentageOfNodesToScore: ptr.To[int32](35),
			Plugins: &config.Plugins{
				QueueSort: config.PluginSet{
					Enabled: []config.Plugin{{Name: "CustomSort"}},
				},
				Bind: config.PluginSet{
					Enabled: []config.Plugin{{Name: "CustomBind"}},
				},
			},
		}},
		Extenders: []config.Extender{{
			PrioritizeVerb: "prioritize",
			Weight:         1,
		}},
	}

	invalidParallelismValue := validConfig.DeepCopy()
	invalidParallelismValue.Parallelism = 0

	resourceNameNotSet := validConfig.DeepCopy()
	resourceNameNotSet.LeaderElection.ResourceName = ""

	resourceNamespaceNotSet := validConfig.DeepCopy()
	resourceNamespaceNotSet.LeaderElection.ResourceNamespace = ""

	resourceLockNotLeases := validConfig.DeepCopy()
	resourceLockNotLeases.LeaderElection.ResourceLock = "configmap"

	enableContentProfilingSetWithoutEnableProfiling := validConfig.DeepCopy()
	enableContentProfilingSetWithoutEnableProfiling.EnableProfiling = false
	enableContentProfilingSetWithoutEnableProfiling.EnableContentionProfiling = true

	percentageOfNodesToScore101 := validConfig.DeepCopy()
	percentageOfNodesToScore101.PercentageOfNodesToScore = ptr.To[int32](101)

	percentageOfNodesToScoreNegative := validConfig.DeepCopy()
	percentageOfNodesToScoreNegative.PercentageOfNodesToScore = ptr.To[int32](-1)

	schedulerNameNotSet := validConfig.DeepCopy()
	schedulerNameNotSet.Profiles[1].SchedulerName = ""

	repeatedSchedulerName := validConfig.DeepCopy()
	repeatedSchedulerName.Profiles[0].SchedulerName = "other"

	profilePercentageOfNodesToScore101 := validConfig.DeepCopy()
	profilePercentageOfNodesToScore101.Profiles[1].PercentageOfNodesToScore = ptr.To[int32](101)

	profilePercentageOfNodesToScoreNegative := validConfig.DeepCopy()
	profilePercentageOfNodesToScoreNegative.Profiles[1].PercentageOfNodesToScore = ptr.To[int32](-1)

	differentQueueSort := validConfig.DeepCopy()
	differentQueueSort.Profiles[1].Plugins.QueueSort.Enabled[0].Name = "AnotherSort"

	oneEmptyQueueSort := validConfig.DeepCopy()
	oneEmptyQueueSort.Profiles[0].Plugins = nil

	extenderNegativeWeight := validConfig.DeepCopy()
	extenderNegativeWeight.Extenders[0].Weight = -1

	invalidNodePercentage := validConfig.DeepCopy()
	invalidNodePercentage.Profiles[0].PluginConfig = []config.PluginConfig{{
		Name: "DefaultPreemption",
		Args: &config.DefaultPreemptionArgs{MinCandidateNodesPercentage: 200, MinCandidateNodesAbsolute: 100},
	}}

	invalidPluginArgs := validConfig.DeepCopy()
	invalidPluginArgs.Profiles[0].PluginConfig = []config.PluginConfig{{
		Name: "DefaultPreemption",
		Args: &config.InterPodAffinityArgs{},
	}}

	duplicatedPluginConfig := validConfig.DeepCopy()
	duplicatedPluginConfig.Profiles[0].PluginConfig = []config.PluginConfig{{
		Name: "config",
	}, {
		Name: "config",
	}}

	mismatchQueueSort := validConfig.DeepCopy()
	mismatchQueueSort.Profiles = []config.KubeSchedulerProfile{{
		SchedulerName: "me",
		Plugins: &config.Plugins{
			QueueSort: config.PluginSet{
				Enabled: []config.Plugin{{Name: "PrioritySort"}},
			},
		},
		PluginConfig: []config.PluginConfig{{
			Name: "PrioritySort",
		}},
	}, {
		SchedulerName: "other",
		Plugins: &config.Plugins{
			QueueSort: config.PluginSet{
				Enabled: []config.Plugin{{Name: "CustomSort"}},
			},
		},
		PluginConfig: []config.PluginConfig{{
			Name: "CustomSort",
		}},
	}}

	extenderDuplicateManagedResource := validConfig.DeepCopy()
	extenderDuplicateManagedResource.Extenders[0].ManagedResources = []config.ExtenderManagedResource{
		{Name: "example.com/foo", IgnoredByScheduler: false},
		{Name: "example.com/foo", IgnoredByScheduler: false},
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
		config   *config.KubeSchedulerConfiguration
		wantErrs field.ErrorList
	}{
		"good": {
			config: validConfig,
		},
		"bad-parallelism-invalid-value": {
			config: invalidParallelismValue,
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "parallelism",
				},
			},
		},
		"bad-resource-name-not-set": {
			config: resourceNameNotSet,
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "leaderElection.resourceName",
				},
			},
		},
		"bad-resource-namespace-not-set": {
			config: resourceNamespaceNotSet,
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "leaderElection.resourceNamespace",
				},
			},
		},
		"bad-resource-lock-not-leases": {
			config: resourceLockNotLeases,
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "leaderElection.resourceLock",
				},
			},
		},
		"bad-percentage-of-nodes-to-score": {
			config: percentageOfNodesToScore101,
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "percentageOfNodesToScore",
				},
			},
		},
		"negative-percentage-of-nodes-to-score": {
			config: percentageOfNodesToScoreNegative,
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "percentageOfNodesToScore",
				},
			},
		},
		"scheduler-name-not-set": {
			config: schedulerNameNotSet,
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeRequired,
					Field: "profiles[1].schedulerName",
				},
			},
		},
		"repeated-scheduler-name": {
			config: repeatedSchedulerName,
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeDuplicate,
					Field: "profiles[1].schedulerName",
				},
			},
		},
		"greater-than-100-profile-percentage-of-nodes-to-score": {
			config: profilePercentageOfNodesToScore101,
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "profiles[1].percentageOfNodesToScore",
				},
			},
		},
		"negative-profile-percentage-of-nodes-to-score": {
			config: profilePercentageOfNodesToScoreNegative,
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "profiles[1].percentageOfNodesToScore",
				},
			},
		},
		"different-queue-sort": {
			config: differentQueueSort,
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "profiles[1].plugins.queueSort",
				},
			},
		},
		"one-empty-queue-sort": {
			config: oneEmptyQueueSort,
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "profiles[1].plugins.queueSort",
				},
			},
		},
		"extender-negative-weight": {
			config: extenderNegativeWeight,
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "extenders[0].weight",
				},
			},
		},
		"extender-duplicate-managed-resources": {
			config: extenderDuplicateManagedResource,
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "extenders[0].managedResources[1].name",
				},
			},
		},
		"extender-duplicate-bind": {
			config: extenderDuplicateBind,
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "extenders",
				},
			},
		},
		"invalid-node-percentage": {
			config: invalidNodePercentage,
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "profiles[0].pluginConfig[0].args.minCandidateNodesPercentage",
				},
			},
		},
		"invalid-plugin-args": {
			config: invalidPluginArgs,
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "profiles[0].pluginConfig[0].args",
				},
			},
		},
		"duplicated-plugin-config": {
			config: duplicatedPluginConfig,
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeDuplicate,
					Field: "profiles[0].pluginConfig[1]",
				},
			},
		},
		"mismatch-queue-sort": {
			config: mismatchQueueSort,
			wantErrs: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "profiles[1].plugins.queueSort",
				},
			},
		},
		"valid-plugins": {
			config: validPlugins,
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateKubeSchedulerConfiguration(scenario.config)
			diff := cmp.Diff(scenario.wantErrs.ToAggregate(), errs, ignoreBadValueDetail)
			if diff != "" {
				t.Errorf("KubeSchedulerConfiguration returned err (-want,+got):\n%s", diff)
			}
		})
	}
}
