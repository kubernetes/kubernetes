/*
Copyright 2021 The Kubernetes Authors.

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

package v1beta2

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/util/feature"
	componentbaseconfig "k8s.io/component-base/config/v1alpha1"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kube-scheduler/config/v1beta2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/utils/pointer"
)

var pluginConfigs = []v1beta2.PluginConfig{
	{
		Name: "DefaultPreemption",
		Args: runtime.RawExtension{
			Object: &v1beta2.DefaultPreemptionArgs{
				TypeMeta: metav1.TypeMeta{
					Kind:       "DefaultPreemptionArgs",
					APIVersion: "kubescheduler.config.k8s.io/v1beta2",
				},
				MinCandidateNodesPercentage: pointer.Int32(10),
				MinCandidateNodesAbsolute:   pointer.Int32(100),
			}},
	},
	{
		Name: "InterPodAffinity",
		Args: runtime.RawExtension{
			Object: &v1beta2.InterPodAffinityArgs{
				TypeMeta: metav1.TypeMeta{
					Kind:       "InterPodAffinityArgs",
					APIVersion: "kubescheduler.config.k8s.io/v1beta2",
				},
				HardPodAffinityWeight: pointer.Int32(1),
			}},
	},
	{
		Name: "NodeAffinity",
		Args: runtime.RawExtension{Object: &v1beta2.NodeAffinityArgs{
			TypeMeta: metav1.TypeMeta{
				Kind:       "NodeAffinityArgs",
				APIVersion: "kubescheduler.config.k8s.io/v1beta2",
			},
		}},
	},
	{
		Name: "NodeResourcesBalancedAllocation",
		Args: runtime.RawExtension{Object: &v1beta2.NodeResourcesBalancedAllocationArgs{
			TypeMeta: metav1.TypeMeta{
				Kind:       "NodeResourcesBalancedAllocationArgs",
				APIVersion: "kubescheduler.config.k8s.io/v1beta2",
			},
			Resources: []v1beta2.ResourceSpec{{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1}},
		}},
	},
	{
		Name: "NodeResourcesFit",
		Args: runtime.RawExtension{Object: &v1beta2.NodeResourcesFitArgs{
			TypeMeta: metav1.TypeMeta{
				Kind:       "NodeResourcesFitArgs",
				APIVersion: "kubescheduler.config.k8s.io/v1beta2",
			},
			ScoringStrategy: &v1beta2.ScoringStrategy{
				Type:      v1beta2.LeastAllocated,
				Resources: []v1beta2.ResourceSpec{{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1}},
			},
		}},
	},
	{
		Name: "PodTopologySpread",
		Args: runtime.RawExtension{Object: &v1beta2.PodTopologySpreadArgs{
			TypeMeta: metav1.TypeMeta{
				Kind:       "PodTopologySpreadArgs",
				APIVersion: "kubescheduler.config.k8s.io/v1beta2",
			},
			DefaultingType: v1beta2.SystemDefaulting,
		}},
	},
	{
		Name: "VolumeBinding",
		Args: runtime.RawExtension{Object: &v1beta2.VolumeBindingArgs{
			TypeMeta: metav1.TypeMeta{
				Kind:       "VolumeBindingArgs",
				APIVersion: "kubescheduler.config.k8s.io/v1beta2",
			},
			BindTimeoutSeconds: pointer.Int64(600),
		}},
	},
}

func TestSchedulerDefaults(t *testing.T) {
	enable := true
	tests := []struct {
		name     string
		config   *v1beta2.KubeSchedulerConfiguration
		expected *v1beta2.KubeSchedulerConfiguration
	}{
		{
			name:   "empty config",
			config: &v1beta2.KubeSchedulerConfiguration{},
			expected: &v1beta2.KubeSchedulerConfiguration{
				Parallelism: pointer.Int32(16),
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           &enable,
					EnableContentionProfiling: &enable,
				},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					LeaderElect:       pointer.Bool(true),
					LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
					RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
					RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
					ResourceLock:      "leases",
					ResourceNamespace: "kube-system",
					ResourceName:      "kube-scheduler",
				},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					QPS:         50,
					Burst:       100,
					ContentType: "application/vnd.kubernetes.protobuf",
				},
				PercentageOfNodesToScore: pointer.Int32(0),
				PodInitialBackoffSeconds: pointer.Int64(1),
				PodMaxBackoffSeconds:     pointer.Int64(10),
				Profiles: []v1beta2.KubeSchedulerProfile{
					{
						Plugins:       getDefaultPlugins(),
						PluginConfig:  pluginConfigs,
						SchedulerName: pointer.String("default-scheduler"),
					},
				},
			},
		},
		{
			name: "no scheduler name",
			config: &v1beta2.KubeSchedulerConfiguration{
				Profiles: []v1beta2.KubeSchedulerProfile{{}},
			},
			expected: &v1beta2.KubeSchedulerConfiguration{
				Parallelism: pointer.Int32(16),
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           &enable,
					EnableContentionProfiling: &enable,
				},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					LeaderElect:       pointer.Bool(true),
					LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
					RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
					RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
					ResourceLock:      "leases",
					ResourceNamespace: "kube-system",
					ResourceName:      "kube-scheduler",
				},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					QPS:         50,
					Burst:       100,
					ContentType: "application/vnd.kubernetes.protobuf",
				},
				PercentageOfNodesToScore: pointer.Int32(0),
				PodInitialBackoffSeconds: pointer.Int64(1),
				PodMaxBackoffSeconds:     pointer.Int64(10),
				Profiles: []v1beta2.KubeSchedulerProfile{
					{
						SchedulerName: pointer.String("default-scheduler"),
						Plugins:       getDefaultPlugins(),
						PluginConfig:  pluginConfigs},
				},
			},
		},
		{
			name: "two profiles",
			config: &v1beta2.KubeSchedulerConfiguration{
				Parallelism: pointer.Int32(16),
				Profiles: []v1beta2.KubeSchedulerProfile{
					{
						PluginConfig: []v1beta2.PluginConfig{
							{Name: "FooPlugin"},
						},
					},
					{
						SchedulerName: pointer.String("custom-scheduler"),
						Plugins: &v1beta2.Plugins{
							Bind: v1beta2.PluginSet{
								Enabled: []v1beta2.Plugin{
									{Name: "BarPlugin"},
								},
								Disabled: []v1beta2.Plugin{
									{Name: names.DefaultBinder},
								},
							},
						},
					},
				},
			},
			expected: &v1beta2.KubeSchedulerConfiguration{
				Parallelism: pointer.Int32(16),
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           &enable,
					EnableContentionProfiling: &enable,
				},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					LeaderElect:       pointer.Bool(true),
					LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
					RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
					RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
					ResourceLock:      "leases",
					ResourceNamespace: "kube-system",
					ResourceName:      "kube-scheduler",
				},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					QPS:         50,
					Burst:       100,
					ContentType: "application/vnd.kubernetes.protobuf",
				},
				PercentageOfNodesToScore: pointer.Int32(0),
				PodInitialBackoffSeconds: pointer.Int64(1),
				PodMaxBackoffSeconds:     pointer.Int64(10),
				Profiles: []v1beta2.KubeSchedulerProfile{
					{
						Plugins: getDefaultPlugins(),
						PluginConfig: []v1beta2.PluginConfig{
							{Name: "FooPlugin"},
							{
								Name: "DefaultPreemption",
								Args: runtime.RawExtension{
									Object: &v1beta2.DefaultPreemptionArgs{
										TypeMeta: metav1.TypeMeta{
											Kind:       "DefaultPreemptionArgs",
											APIVersion: "kubescheduler.config.k8s.io/v1beta2",
										},
										MinCandidateNodesPercentage: pointer.Int32(10),
										MinCandidateNodesAbsolute:   pointer.Int32(100),
									}},
							},
							{
								Name: "InterPodAffinity",
								Args: runtime.RawExtension{
									Object: &v1beta2.InterPodAffinityArgs{
										TypeMeta: metav1.TypeMeta{
											Kind:       "InterPodAffinityArgs",
											APIVersion: "kubescheduler.config.k8s.io/v1beta2",
										},
										HardPodAffinityWeight: pointer.Int32(1),
									}},
							},
							{
								Name: "NodeAffinity",
								Args: runtime.RawExtension{Object: &v1beta2.NodeAffinityArgs{
									TypeMeta: metav1.TypeMeta{
										Kind:       "NodeAffinityArgs",
										APIVersion: "kubescheduler.config.k8s.io/v1beta2",
									},
								}},
							},
							{
								Name: "NodeResourcesBalancedAllocation",
								Args: runtime.RawExtension{Object: &v1beta2.NodeResourcesBalancedAllocationArgs{
									TypeMeta: metav1.TypeMeta{
										Kind:       "NodeResourcesBalancedAllocationArgs",
										APIVersion: "kubescheduler.config.k8s.io/v1beta2",
									},
									Resources: []v1beta2.ResourceSpec{{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1}},
								}},
							},
							{
								Name: "NodeResourcesFit",
								Args: runtime.RawExtension{Object: &v1beta2.NodeResourcesFitArgs{
									TypeMeta: metav1.TypeMeta{
										Kind:       "NodeResourcesFitArgs",
										APIVersion: "kubescheduler.config.k8s.io/v1beta2",
									},
									ScoringStrategy: &v1beta2.ScoringStrategy{
										Type:      v1beta2.LeastAllocated,
										Resources: []v1beta2.ResourceSpec{{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1}},
									},
								}},
							},
							{
								Name: "PodTopologySpread",
								Args: runtime.RawExtension{Object: &v1beta2.PodTopologySpreadArgs{
									TypeMeta: metav1.TypeMeta{
										Kind:       "PodTopologySpreadArgs",
										APIVersion: "kubescheduler.config.k8s.io/v1beta2",
									},
									DefaultingType: v1beta2.SystemDefaulting,
								}},
							},
							{
								Name: "VolumeBinding",
								Args: runtime.RawExtension{Object: &v1beta2.VolumeBindingArgs{
									TypeMeta: metav1.TypeMeta{
										Kind:       "VolumeBindingArgs",
										APIVersion: "kubescheduler.config.k8s.io/v1beta2",
									},
									BindTimeoutSeconds: pointer.Int64(600),
								}},
							},
						},
					},
					{
						SchedulerName: pointer.String("custom-scheduler"),
						Plugins: &v1beta2.Plugins{
							QueueSort: v1beta2.PluginSet{
								Enabled: []v1beta2.Plugin{
									{Name: names.PrioritySort},
								},
							},
							PreFilter: v1beta2.PluginSet{
								Enabled: []v1beta2.Plugin{
									{Name: names.NodeResourcesFit},
									{Name: names.NodePorts},
									{Name: names.VolumeRestrictions},
									{Name: names.PodTopologySpread},
									{Name: names.InterPodAffinity},
									{Name: names.VolumeBinding},
									{Name: names.NodeAffinity},
								},
							},
							Filter: v1beta2.PluginSet{
								Enabled: []v1beta2.Plugin{
									{Name: names.NodeUnschedulable},
									{Name: names.NodeName},
									{Name: names.TaintToleration},
									{Name: names.NodeAffinity},
									{Name: names.NodePorts},
									{Name: names.NodeResourcesFit},
									{Name: names.VolumeRestrictions},
									{Name: names.EBSLimits},
									{Name: names.GCEPDLimits},
									{Name: names.NodeVolumeLimits},
									{Name: names.AzureDiskLimits},
									{Name: names.VolumeBinding},
									{Name: names.VolumeZone},
									{Name: names.PodTopologySpread},
									{Name: names.InterPodAffinity},
								},
							},
							PostFilter: v1beta2.PluginSet{
								Enabled: []v1beta2.Plugin{
									{Name: names.DefaultPreemption},
								},
							},
							PreScore: v1beta2.PluginSet{
								Enabled: []v1beta2.Plugin{
									{Name: names.InterPodAffinity},
									{Name: names.PodTopologySpread},
									{Name: names.TaintToleration},
									{Name: names.NodeAffinity},
								},
							},
							Score: v1beta2.PluginSet{
								Enabled: []v1beta2.Plugin{
									{Name: names.NodeResourcesBalancedAllocation, Weight: pointer.Int32(1)},
									{Name: names.ImageLocality, Weight: pointer.Int32(1)},
									{Name: names.InterPodAffinity, Weight: pointer.Int32(1)},
									{Name: names.NodeResourcesFit, Weight: pointer.Int32(1)},
									{Name: names.NodeAffinity, Weight: pointer.Int32(1)},
									{Name: names.PodTopologySpread, Weight: pointer.Int32(2)},
									{Name: names.TaintToleration, Weight: pointer.Int32(1)},
								},
							},
							Reserve: v1beta2.PluginSet{
								Enabled: []v1beta2.Plugin{
									{Name: names.VolumeBinding},
								},
							},
							PreBind: v1beta2.PluginSet{
								Enabled: []v1beta2.Plugin{
									{Name: names.VolumeBinding},
								},
							},
							Bind: v1beta2.PluginSet{
								Enabled: []v1beta2.Plugin{
									{Name: "BarPlugin"},
								},
							},
						},
						PluginConfig: pluginConfigs,
					},
				},
			},
		},
		{
			name: "set non default parallelism",
			config: &v1beta2.KubeSchedulerConfiguration{
				Parallelism: pointer.Int32(8),
			},
			expected: &v1beta2.KubeSchedulerConfiguration{
				Parallelism: pointer.Int32(8),
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           &enable,
					EnableContentionProfiling: &enable,
				},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					LeaderElect:       pointer.Bool(true),
					LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
					RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
					RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
					ResourceLock:      "leases",
					ResourceNamespace: "kube-system",
					ResourceName:      "kube-scheduler",
				},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					QPS:         50,
					Burst:       100,
					ContentType: "application/vnd.kubernetes.protobuf",
				},
				PercentageOfNodesToScore: pointer.Int32(0),
				PodInitialBackoffSeconds: pointer.Int64(1),
				PodMaxBackoffSeconds:     pointer.Int64(10),
				Profiles: []v1beta2.KubeSchedulerProfile{
					{
						Plugins:       getDefaultPlugins(),
						PluginConfig:  pluginConfigs,
						SchedulerName: pointer.String("default-scheduler"),
					},
				},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			SetDefaults_KubeSchedulerConfiguration(tc.config)
			if diff := cmp.Diff(tc.expected, tc.config); diff != "" {
				t.Errorf("Got unexpected defaults (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestPluginArgsDefaults(t *testing.T) {
	tests := []struct {
		name     string
		features map[featuregate.Feature]bool
		in       runtime.Object
		want     runtime.Object
	}{
		{
			name: "DefaultPreemptionArgs empty",
			in:   &v1beta2.DefaultPreemptionArgs{},
			want: &v1beta2.DefaultPreemptionArgs{
				MinCandidateNodesPercentage: pointer.Int32(10),
				MinCandidateNodesAbsolute:   pointer.Int32(100),
			},
		},
		{
			name: "DefaultPreemptionArgs with value",
			in: &v1beta2.DefaultPreemptionArgs{
				MinCandidateNodesPercentage: pointer.Int32(50),
			},
			want: &v1beta2.DefaultPreemptionArgs{
				MinCandidateNodesPercentage: pointer.Int32(50),
				MinCandidateNodesAbsolute:   pointer.Int32(100),
			},
		},
		{
			name: "InterPodAffinityArgs empty",
			in:   &v1beta2.InterPodAffinityArgs{},
			want: &v1beta2.InterPodAffinityArgs{
				HardPodAffinityWeight: pointer.Int32(1),
			},
		},
		{
			name: "InterPodAffinityArgs explicit 0",
			in: &v1beta2.InterPodAffinityArgs{
				HardPodAffinityWeight: pointer.Int32(0),
			},
			want: &v1beta2.InterPodAffinityArgs{
				HardPodAffinityWeight: pointer.Int32(0),
			},
		},
		{
			name: "InterPodAffinityArgs with value",
			in: &v1beta2.InterPodAffinityArgs{
				HardPodAffinityWeight: pointer.Int32(5),
			},
			want: &v1beta2.InterPodAffinityArgs{
				HardPodAffinityWeight: pointer.Int32(5),
			},
		},
		{
			name: "NodeResourcesBalancedAllocationArgs resources empty",
			in:   &v1beta2.NodeResourcesBalancedAllocationArgs{},
			want: &v1beta2.NodeResourcesBalancedAllocationArgs{
				Resources: []v1beta2.ResourceSpec{
					{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1},
				},
			},
		},
		{
			name: "NodeResourcesBalancedAllocationArgs with scalar resource",
			in: &v1beta2.NodeResourcesBalancedAllocationArgs{
				Resources: []v1beta2.ResourceSpec{
					{Name: "scalar.io/scalar1", Weight: 1},
				},
			},
			want: &v1beta2.NodeResourcesBalancedAllocationArgs{
				Resources: []v1beta2.ResourceSpec{
					{Name: "scalar.io/scalar1", Weight: 1},
				},
			},
		},
		{
			name: "NodeResourcesBalancedAllocationArgs with mixed resources",
			in: &v1beta2.NodeResourcesBalancedAllocationArgs{
				Resources: []v1beta2.ResourceSpec{
					{Name: string(v1.ResourceCPU), Weight: 1},
					{Name: "scalar.io/scalar1", Weight: 1},
				},
			},
			want: &v1beta2.NodeResourcesBalancedAllocationArgs{
				Resources: []v1beta2.ResourceSpec{
					{Name: string(v1.ResourceCPU), Weight: 1},
					{Name: "scalar.io/scalar1", Weight: 1},
				},
			},
		},
		{
			name: "NodeResourcesBalancedAllocationArgs have resource no weight",
			in: &v1beta2.NodeResourcesBalancedAllocationArgs{
				Resources: []v1beta2.ResourceSpec{
					{Name: string(v1.ResourceCPU)},
					{Name: "scalar.io/scalar0"},
					{Name: "scalar.io/scalar1", Weight: 1},
				},
			},
			want: &v1beta2.NodeResourcesBalancedAllocationArgs{
				Resources: []v1beta2.ResourceSpec{
					{Name: string(v1.ResourceCPU), Weight: 1},
					{Name: "scalar.io/scalar0", Weight: 1},
					{Name: "scalar.io/scalar1", Weight: 1},
				},
			},
		},
		{
			name: "PodTopologySpreadArgs resources empty",
			in:   &v1beta2.PodTopologySpreadArgs{},
			want: &v1beta2.PodTopologySpreadArgs{
				DefaultingType: v1beta2.SystemDefaulting,
			},
		},
		{
			name: "PodTopologySpreadArgs resources with value",
			in: &v1beta2.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						TopologyKey:       "planet",
						WhenUnsatisfiable: v1.DoNotSchedule,
						MaxSkew:           2,
					},
				},
			},
			want: &v1beta2.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						TopologyKey:       "planet",
						WhenUnsatisfiable: v1.DoNotSchedule,
						MaxSkew:           2,
					},
				},
				DefaultingType: v1beta2.SystemDefaulting,
			},
		},
		{
			name: "NodeResourcesFitArgs not set",
			in:   &v1beta2.NodeResourcesFitArgs{},
			want: &v1beta2.NodeResourcesFitArgs{
				ScoringStrategy: &v1beta2.ScoringStrategy{
					Type:      v1beta2.LeastAllocated,
					Resources: defaultResourceSpec,
				},
			},
		},
		{
			name: "NodeResourcesFitArgs Resources empty",
			in: &v1beta2.NodeResourcesFitArgs{
				ScoringStrategy: &v1beta2.ScoringStrategy{
					Type: v1beta2.MostAllocated,
				},
			},
			want: &v1beta2.NodeResourcesFitArgs{
				ScoringStrategy: &v1beta2.ScoringStrategy{
					Type:      v1beta2.MostAllocated,
					Resources: defaultResourceSpec,
				},
			},
		},
		{
			name: "VolumeBindingArgs empty, VolumeCapacityPriority disabled",
			features: map[featuregate.Feature]bool{
				features.VolumeCapacityPriority: false,
			},
			in: &v1beta2.VolumeBindingArgs{},
			want: &v1beta2.VolumeBindingArgs{
				BindTimeoutSeconds: pointer.Int64(600),
			},
		},
		{
			name: "VolumeBindingArgs empty, VolumeCapacityPriority enabled",
			features: map[featuregate.Feature]bool{
				features.VolumeCapacityPriority: true,
			},
			in: &v1beta2.VolumeBindingArgs{},
			want: &v1beta2.VolumeBindingArgs{
				BindTimeoutSeconds: pointer.Int64(600),
				Shape: []v1beta2.UtilizationShapePoint{
					{Utilization: 0, Score: 0},
					{Utilization: 100, Score: 10},
				},
			},
		},
	}
	for _, tc := range tests {
		scheme := runtime.NewScheme()
		utilruntime.Must(AddToScheme(scheme))
		t.Run(tc.name, func(t *testing.T) {
			for k, v := range tc.features {
				defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, k, v)()
			}
			scheme.Default(tc.in)
			if diff := cmp.Diff(tc.in, tc.want); diff != "" {
				t.Errorf("Got unexpected defaults (-want, +got):\n%s", diff)
			}
		})
	}
}
