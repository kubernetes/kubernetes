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
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	componentbaseconfig "k8s.io/component-base/config"
	componentbaseconfigv1alpha1 "k8s.io/component-base/config/v1alpha1"
	"k8s.io/kube-scheduler/config/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/utils/pointer"
)

func TestConvertKubeSchedulerConfiguration(t *testing.T) {
	cases := []struct {
		name string
		cfg  v1alpha1.KubeSchedulerConfiguration
		want config.KubeSchedulerConfiguration
	}{
		{
			name: "scheduler name",
			cfg: v1alpha1.KubeSchedulerConfiguration{
				SchedulerName: pointer.StringPtr("custom-name"),
			},
			want: config.KubeSchedulerConfiguration{
				Profiles: []config.KubeSchedulerProfile{
					{SchedulerName: "custom-name"},
				},
			},
		},
		{
			name: "plugins and plugin config",
			cfg: v1alpha1.KubeSchedulerConfiguration{
				Plugins: &v1alpha1.Plugins{
					QueueSort: &v1alpha1.PluginSet{
						Enabled: []v1alpha1.Plugin{
							{Name: "FooPlugin"},
						},
					},
				},
				PluginConfig: []v1alpha1.PluginConfig{
					{Name: "FooPlugin"},
				},
			},
			want: config.KubeSchedulerConfiguration{
				Profiles: []config.KubeSchedulerProfile{
					{
						Plugins: &config.Plugins{
							QueueSort: &config.PluginSet{
								Enabled: []config.Plugin{
									{Name: "FooPlugin"},
								},
							},
						},
						PluginConfig: []config.PluginConfig{
							{Name: "FooPlugin"},
						},
					},
				},
			},
		},
		{
			name: "custom hard pod affinity weight",
			cfg: v1alpha1.KubeSchedulerConfiguration{
				HardPodAffinitySymmetricWeight: pointer.Int32Ptr(3),
			},
			want: config.KubeSchedulerConfiguration{
				Profiles: []config.KubeSchedulerProfile{
					{
						PluginConfig: []config.PluginConfig{
							{
								Name: "InterPodAffinity",
								Args: runtime.Unknown{
									Raw: []byte(`{"hardPodAffinityWeight":3}`),
								},
							},
						},
					},
				},
			},
		},
		{
			name: "custom hard pod affinity weight and existing PluginConfig",
			cfg: v1alpha1.KubeSchedulerConfiguration{
				HardPodAffinitySymmetricWeight: pointer.Int32Ptr(3),
				PluginConfig: []v1alpha1.PluginConfig{
					{
						Name: "InterPodAffinity",
						Args: runtime.Unknown{
							Raw: []byte(`{"hardPodAffinityWeight":5}`),
						},
					},
				},
			},
			want: config.KubeSchedulerConfiguration{
				Profiles: []config.KubeSchedulerProfile{
					{
						PluginConfig: []config.PluginConfig{
							{
								Name: "InterPodAffinity",
								Args: runtime.Unknown{
									Raw: []byte(`{"hardPodAffinityWeight":5}`),
								},
							},
							{
								Name: "InterPodAffinity",
								Args: runtime.Unknown{
									Raw: []byte(`{"hardPodAffinityWeight":3}`),
								},
							},
						},
					},
				},
			},
		},
	}
	scheme := getScheme(t)
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var out config.KubeSchedulerConfiguration
			err := scheme.Convert(&tc.cfg, &out, nil)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tc.want, out); diff != "" {
				t.Errorf("unexpected conversion (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestV1alpha1ToConfigKubeSchedulerLeaderElectionConfiguration(t *testing.T) {
	configuration := &v1alpha1.KubeSchedulerLeaderElectionConfiguration{
		LockObjectName:      "name",
		LockObjectNamespace: "namespace",
		LeaderElectionConfiguration: componentbaseconfigv1alpha1.LeaderElectionConfiguration{
			ResourceName:      "name",
			ResourceNamespace: "namespace",
		},
	}
	emptyLockObjectNameConfig := configuration.DeepCopy()
	emptyLockObjectNameConfig.LockObjectName = ""

	emptyLockObjectNamespaceConfig := configuration.DeepCopy()
	emptyLockObjectNamespaceConfig.LockObjectNamespace = ""

	emptyResourceNameConfig := configuration.DeepCopy()
	emptyResourceNameConfig.ResourceName = ""

	emptyResourceNamespaceConfig := configuration.DeepCopy()
	emptyResourceNamespaceConfig.ResourceNamespace = ""

	differentNameConfig := configuration.DeepCopy()
	differentNameConfig.LockObjectName = "name1"

	differentNamespaceConfig := configuration.DeepCopy()
	differentNamespaceConfig.LockObjectNamespace = "namespace1"

	emptyconfig := &v1alpha1.KubeSchedulerLeaderElectionConfiguration{}

	scenarios := map[string]struct {
		expectedResourceNamespace string
		expectedResourceName      string
		expectedToFailed          bool
		config                    *v1alpha1.KubeSchedulerLeaderElectionConfiguration
	}{
		"both-set-same-name-and-namespace": {
			expectedResourceNamespace: "namespace",
			expectedResourceName:      "name",
			expectedToFailed:          false,
			config:                    configuration,
		},
		"not-set-lock-object-name": {
			expectedResourceNamespace: "namespace",
			expectedResourceName:      "name",
			expectedToFailed:          false,
			config:                    emptyLockObjectNameConfig,
		},
		"not-set-lock-object-namespace": {
			expectedResourceNamespace: "namespace",
			expectedResourceName:      "name",
			expectedToFailed:          false,
			config:                    emptyLockObjectNamespaceConfig,
		},
		"not-set-resource-name": {
			expectedResourceNamespace: "namespace",
			expectedResourceName:      "name",
			expectedToFailed:          false,
			config:                    emptyResourceNameConfig,
		},
		"not-set-resource-namespace": {
			expectedResourceNamespace: "namespace",
			expectedResourceName:      "name",
			expectedToFailed:          false,
			config:                    emptyResourceNamespaceConfig,
		},
		"set-different-name": {
			expectedResourceNamespace: "",
			expectedResourceName:      "",
			expectedToFailed:          true,
			config:                    differentNameConfig,
		},
		"set-different-namespace": {
			expectedResourceNamespace: "",
			expectedResourceName:      "",
			expectedToFailed:          true,
			config:                    differentNamespaceConfig,
		},
		"set-empty-name-and-namespace": {
			expectedResourceNamespace: "",
			expectedResourceName:      "",
			expectedToFailed:          false,
			config:                    emptyconfig,
		},
	}
	for name, scenario := range scenarios {
		out := &config.KubeSchedulerLeaderElectionConfiguration{}
		s := conversion.Scope(nil)
		err := Convert_v1alpha1_KubeSchedulerLeaderElectionConfiguration_To_config_KubeSchedulerLeaderElectionConfiguration(scenario.config, out, s)
		if err == nil && scenario.expectedToFailed {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if err == nil && !scenario.expectedToFailed {
			if out.ResourceName != scenario.expectedResourceName {
				t.Errorf("Unexpected success for scenario: %s, out.ResourceName: %s, expectedResourceName: %s", name, out.ResourceName, scenario.expectedResourceName)
			}
			if out.ResourceNamespace != scenario.expectedResourceNamespace {
				t.Errorf("Unexpected success for scenario: %s, out.ResourceNamespace: %s, expectedResourceNamespace: %s", name, out.ResourceNamespace, scenario.expectedResourceNamespace)
			}
		}
		if err != nil && !scenario.expectedToFailed {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, err)
		}
	}
}

func TestConfigToV1alpha1KubeSchedulerLeaderElectionConfiguration(t *testing.T) {
	configuration := &config.KubeSchedulerLeaderElectionConfiguration{
		LeaderElectionConfiguration: componentbaseconfig.LeaderElectionConfiguration{
			ResourceName:      "name",
			ResourceNamespace: "namespace",
		},
	}
	emptyconfig := &config.KubeSchedulerLeaderElectionConfiguration{}

	scenarios := map[string]struct {
		expectedResourceNamespace   string
		expectedResourceName        string
		expectedLockObjectNamespace string
		expectedLockObjectName      string
		expectedToFailed            bool
		config                      *config.KubeSchedulerLeaderElectionConfiguration
	}{
		"both-set-name-and-namespace": {
			expectedResourceNamespace:   "namespace",
			expectedResourceName:        "name",
			expectedLockObjectNamespace: "namespace",
			expectedLockObjectName:      "name",
			expectedToFailed:            false,
			config:                      configuration,
		},
		"set-empty-name-and-namespace": {
			expectedResourceNamespace:   "",
			expectedResourceName:        "",
			expectedLockObjectNamespace: "",
			expectedLockObjectName:      "",
			expectedToFailed:            false,
			config:                      emptyconfig,
		},
	}
	for name, scenario := range scenarios {
		out := &v1alpha1.KubeSchedulerLeaderElectionConfiguration{}
		s := conversion.Scope(nil)
		err := Convert_config_KubeSchedulerLeaderElectionConfiguration_To_v1alpha1_KubeSchedulerLeaderElectionConfiguration(scenario.config, out, s)
		if err == nil && scenario.expectedToFailed {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if err == nil && !scenario.expectedToFailed {
			if out.ResourceName != scenario.expectedResourceName {
				t.Errorf("Unexpected success for scenario: %s, out.ResourceName: %s, expectedResourceName: %s", name, out.ResourceName, scenario.expectedResourceName)
			}
			if out.LockObjectName != scenario.expectedLockObjectName {
				t.Errorf("Unexpected success for scenario: %s, out.LockObjectName: %s, expectedLockObjectName: %s", name, out.LockObjectName, scenario.expectedLockObjectName)
			}
			if out.ResourceNamespace != scenario.expectedResourceNamespace {
				t.Errorf("Unexpected success for scenario: %s, out.ResourceNamespace: %s, expectedResourceNamespace: %s", name, out.ResourceNamespace, scenario.expectedResourceNamespace)
			}
			if out.LockObjectNamespace != scenario.expectedLockObjectNamespace {
				t.Errorf("Unexpected success for scenario: %s, out.LockObjectNamespace: %s, expectedLockObjectNamespace: %s", name, out.LockObjectNamespace, scenario.expectedLockObjectNamespace)
			}
		}
		if err != nil && !scenario.expectedToFailed {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, err)
		}
	}
}

func TestConvertBetweenV1Alpha1PluginsAndConfigPlugins(t *testing.T) {
	// weight is assigned to score plugins
	weight := int32(10)
	// DummyWeight is a placeholder for the v1alpha1.plugins' weight will be filled with zero when
	// convert back from config.
	dummyWeight := int32(42)
	v1alpha1Plugins := v1alpha1.Plugins{
		QueueSort: &v1alpha1.PluginSet{
			Enabled: []v1alpha1.Plugin{
				{Name: "queuesort-plugin", Weight: &dummyWeight},
			},
			Disabled: []v1alpha1.Plugin{
				{Name: "disabled-queuesort-plugin", Weight: &dummyWeight},
			},
		},
		PreFilter: &v1alpha1.PluginSet{
			Enabled: []v1alpha1.Plugin{
				{Name: "prefilter-plugin", Weight: &dummyWeight},
			},
			Disabled: []v1alpha1.Plugin{
				{Name: "disabled-prefilter-plugin", Weight: &dummyWeight},
			},
		},
		Filter: &v1alpha1.PluginSet{
			Enabled: []v1alpha1.Plugin{
				{Name: "filter-plugin", Weight: &dummyWeight},
			},
			Disabled: []v1alpha1.Plugin{
				{Name: "disabled-filter-plugin", Weight: &dummyWeight},
			},
		},
		PostFilter: &v1alpha1.PluginSet{
			Enabled: []v1alpha1.Plugin{
				{Name: "postfilter-plugin", Weight: &dummyWeight},
			},
			Disabled: []v1alpha1.Plugin{
				{Name: "disabled-postfilter-plugin", Weight: &dummyWeight},
			},
		},
		Score: &v1alpha1.PluginSet{
			Enabled: []v1alpha1.Plugin{
				{Name: "score-plugin", Weight: &weight},
			},
			Disabled: []v1alpha1.Plugin{
				{Name: "disabled-score-plugin", Weight: &weight},
			},
		},
		Reserve: &v1alpha1.PluginSet{
			Enabled: []v1alpha1.Plugin{
				{Name: "reserve-plugin", Weight: &dummyWeight},
			},
			Disabled: []v1alpha1.Plugin{
				{Name: "disabled-reserve-plugin", Weight: &dummyWeight},
			},
		},
		Permit: &v1alpha1.PluginSet{
			Enabled: []v1alpha1.Plugin{
				{Name: "permit-plugin", Weight: &dummyWeight},
			},
			Disabled: []v1alpha1.Plugin{
				{Name: "disabled-permit-plugin", Weight: &dummyWeight},
			},
		},
		PreBind: &v1alpha1.PluginSet{
			Enabled: []v1alpha1.Plugin{
				{Name: "prebind-plugin", Weight: &dummyWeight},
			},
			Disabled: []v1alpha1.Plugin{
				{Name: "disabled-prebind-plugin", Weight: &dummyWeight},
			},
		},
		Bind: &v1alpha1.PluginSet{
			Enabled: []v1alpha1.Plugin{
				{Name: "bind-plugin", Weight: &dummyWeight},
			},
			Disabled: []v1alpha1.Plugin{
				{Name: "disabled-bind-plugin", Weight: &dummyWeight},
			},
		},
		PostBind: &v1alpha1.PluginSet{
			Enabled: []v1alpha1.Plugin{
				{Name: "postbind-plugin", Weight: &dummyWeight},
			},
			Disabled: []v1alpha1.Plugin{
				{Name: "disabled-postbind-plugin", Weight: &dummyWeight},
			},
		},
		Unreserve: &v1alpha1.PluginSet{
			Enabled: []v1alpha1.Plugin{
				{Name: "unreserve-plugin", Weight: &dummyWeight},
			},
			Disabled: []v1alpha1.Plugin{
				{Name: "disabled-unreserve-plugin", Weight: &dummyWeight},
			},
		},
	}
	configPlugins := config.Plugins{
		QueueSort: &config.PluginSet{
			Enabled: []config.Plugin{
				{Name: "queuesort-plugin", Weight: dummyWeight},
			},
			Disabled: []config.Plugin{
				{Name: "disabled-queuesort-plugin", Weight: dummyWeight},
			},
		},
		PreFilter: &config.PluginSet{
			Enabled: []config.Plugin{
				{Name: "prefilter-plugin", Weight: dummyWeight},
			},
			Disabled: []config.Plugin{
				{Name: "disabled-prefilter-plugin", Weight: dummyWeight},
			},
		},
		Filter: &config.PluginSet{
			Enabled: []config.Plugin{
				{Name: "filter-plugin", Weight: dummyWeight},
			},
			Disabled: []config.Plugin{
				{Name: "disabled-filter-plugin", Weight: dummyWeight},
			},
		},
		PreScore: &config.PluginSet{
			Enabled: []config.Plugin{
				{Name: "postfilter-plugin", Weight: dummyWeight},
			},
			Disabled: []config.Plugin{
				{Name: "disabled-postfilter-plugin", Weight: dummyWeight},
			},
		},
		Score: &config.PluginSet{
			Enabled: []config.Plugin{
				{Name: "score-plugin", Weight: weight},
			},
			Disabled: []config.Plugin{
				{Name: "disabled-score-plugin", Weight: weight},
			},
		},
		Reserve: &config.PluginSet{
			Enabled: []config.Plugin{
				{Name: "reserve-plugin", Weight: dummyWeight},
			},
			Disabled: []config.Plugin{
				{Name: "disabled-reserve-plugin", Weight: dummyWeight},
			},
		},
		Permit: &config.PluginSet{
			Enabled: []config.Plugin{
				{Name: "permit-plugin", Weight: dummyWeight},
			},
			Disabled: []config.Plugin{
				{Name: "disabled-permit-plugin", Weight: dummyWeight},
			},
		},
		PreBind: &config.PluginSet{
			Enabled: []config.Plugin{
				{Name: "prebind-plugin", Weight: dummyWeight},
			},
			Disabled: []config.Plugin{
				{Name: "disabled-prebind-plugin", Weight: dummyWeight},
			},
		},
		Bind: &config.PluginSet{
			Enabled: []config.Plugin{
				{Name: "bind-plugin", Weight: dummyWeight},
			},
			Disabled: []config.Plugin{
				{Name: "disabled-bind-plugin", Weight: dummyWeight},
			},
		},
		PostBind: &config.PluginSet{
			Enabled: []config.Plugin{
				{Name: "postbind-plugin", Weight: dummyWeight},
			},
			Disabled: []config.Plugin{
				{Name: "disabled-postbind-plugin", Weight: dummyWeight},
			},
		},
		Unreserve: &config.PluginSet{
			Enabled: []config.Plugin{
				{Name: "unreserve-plugin", Weight: dummyWeight},
			},
			Disabled: []config.Plugin{
				{Name: "disabled-unreserve-plugin", Weight: dummyWeight},
			},
		},
	}
	convertedConfigPlugins := config.Plugins{}
	convertedV1Alpha1Plugins := v1alpha1.Plugins{}
	scheme := getScheme(t)
	if err := scheme.Convert(&v1alpha1Plugins, &convertedConfigPlugins, nil); err != nil {
		t.Fatal(err)
	}
	if err := scheme.Convert(&configPlugins, &convertedV1Alpha1Plugins, nil); err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff(configPlugins, convertedConfigPlugins); diff != "" {
		t.Errorf("unexpected plugins diff (-want, +got): %s", diff)
	}
	if diff := cmp.Diff(v1alpha1Plugins, convertedV1Alpha1Plugins); diff != "" {
		t.Errorf("unexpected plugins diff (-want, +got): %s", diff)
	}
}

func getScheme(t *testing.T) *runtime.Scheme {
	scheme := runtime.NewScheme()
	if err := AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	return scheme
}
