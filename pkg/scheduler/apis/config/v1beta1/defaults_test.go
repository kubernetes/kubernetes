/*
Copyright 2020 The Kubernetes Authors.

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

package v1beta1

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
	"k8s.io/kube-scheduler/config/v1beta1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/utils/pointer"
)

var pluginConfigs = []v1beta1.PluginConfig{
	{
		Name: "DefaultPreemption",
		Args: runtime.RawExtension{
			Object: &v1beta1.DefaultPreemptionArgs{
				TypeMeta: metav1.TypeMeta{
					Kind:       "DefaultPreemptionArgs",
					APIVersion: "kubescheduler.config.k8s.io/v1beta1",
				},
				MinCandidateNodesPercentage: pointer.Int32Ptr(10),
				MinCandidateNodesAbsolute:   pointer.Int32Ptr(100),
			}},
	},
	{
		Name: "InterPodAffinity",
		Args: runtime.RawExtension{
			Object: &v1beta1.InterPodAffinityArgs{
				TypeMeta: metav1.TypeMeta{
					Kind:       "InterPodAffinityArgs",
					APIVersion: "kubescheduler.config.k8s.io/v1beta1",
				},
				HardPodAffinityWeight: pointer.Int32Ptr(1),
			}},
	},
	{
		Name: "NodeAffinity",
		Args: runtime.RawExtension{Object: &v1beta1.NodeAffinityArgs{
			TypeMeta: metav1.TypeMeta{
				Kind:       "NodeAffinityArgs",
				APIVersion: "kubescheduler.config.k8s.io/v1beta1",
			},
		}},
	},
	{
		Name: "NodeResourcesBalancedAllocation",
		Args: runtime.RawExtension{Object: &v1beta1.NodeResourcesBalancedAllocationArgs{
			TypeMeta: metav1.TypeMeta{
				Kind:       "NodeResourcesBalancedAllocationArgs",
				APIVersion: "kubescheduler.config.k8s.io/v1beta1",
			},
			Resources: []v1beta1.ResourceSpec{{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1}},
		}},
	},
	{
		Name: "NodeResourcesFit",
		Args: runtime.RawExtension{Object: &v1beta1.NodeResourcesFitArgs{
			TypeMeta: metav1.TypeMeta{
				Kind:       "NodeResourcesFitArgs",
				APIVersion: "kubescheduler.config.k8s.io/v1beta1",
			},
			ScoringStrategy: &v1beta1.ScoringStrategy{
				Type:      v1beta1.LeastAllocated,
				Resources: []v1beta1.ResourceSpec{{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1}},
			},
		}},
	},
	{
		Name: "NodeResourcesLeastAllocated",
		Args: runtime.RawExtension{Object: &v1beta1.NodeResourcesLeastAllocatedArgs{
			TypeMeta: metav1.TypeMeta{
				Kind:       "NodeResourcesLeastAllocatedArgs",
				APIVersion: "kubescheduler.config.k8s.io/v1beta1",
			},
			Resources: []v1beta1.ResourceSpec{{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1}},
		}},
	},
	{
		Name: "PodTopologySpread",
		Args: runtime.RawExtension{Object: &v1beta1.PodTopologySpreadArgs{
			TypeMeta: metav1.TypeMeta{
				Kind:       "PodTopologySpreadArgs",
				APIVersion: "kubescheduler.config.k8s.io/v1beta1",
			},
			DefaultingType: v1beta1.SystemDefaulting,
		}},
	},
	{
		Name: "VolumeBinding",
		Args: runtime.RawExtension{Object: &v1beta1.VolumeBindingArgs{
			TypeMeta: metav1.TypeMeta{
				Kind:       "VolumeBindingArgs",
				APIVersion: "kubescheduler.config.k8s.io/v1beta1",
			},
			BindTimeoutSeconds: pointer.Int64Ptr(600),
		}},
	},
}

func TestSchedulerDefaults(t *testing.T) {
	enable := true
	tests := []struct {
		name     string
		config   *v1beta1.KubeSchedulerConfiguration
		expected *v1beta1.KubeSchedulerConfiguration
	}{
		{
			name:   "empty config",
			config: &v1beta1.KubeSchedulerConfiguration{},
			expected: &v1beta1.KubeSchedulerConfiguration{
				Parallelism:        pointer.Int32Ptr(16),
				HealthzBindAddress: pointer.StringPtr("0.0.0.0:10251"),
				MetricsBindAddress: pointer.StringPtr("0.0.0.0:10251"),
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           &enable,
					EnableContentionProfiling: &enable,
				},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					LeaderElect:       pointer.BoolPtr(true),
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
				PercentageOfNodesToScore: pointer.Int32Ptr(0),
				PodInitialBackoffSeconds: pointer.Int64Ptr(1),
				PodMaxBackoffSeconds:     pointer.Int64Ptr(10),
				Profiles: []v1beta1.KubeSchedulerProfile{
					{
						SchedulerName: pointer.StringPtr("default-scheduler"),
						Plugins:       getDefaultPlugins(),
						PluginConfig:  pluginConfigs,
					},
				},
			},
		},
		{
			name: "no scheduler name",
			config: &v1beta1.KubeSchedulerConfiguration{
				Profiles: []v1beta1.KubeSchedulerProfile{{}},
			},
			expected: &v1beta1.KubeSchedulerConfiguration{
				Parallelism:        pointer.Int32Ptr(16),
				HealthzBindAddress: pointer.StringPtr("0.0.0.0:10251"),
				MetricsBindAddress: pointer.StringPtr("0.0.0.0:10251"),
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           &enable,
					EnableContentionProfiling: &enable,
				},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					LeaderElect:       pointer.BoolPtr(true),
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
				PercentageOfNodesToScore: pointer.Int32Ptr(0),
				PodInitialBackoffSeconds: pointer.Int64Ptr(1),
				PodMaxBackoffSeconds:     pointer.Int64Ptr(10),
				Profiles: []v1beta1.KubeSchedulerProfile{
					{
						SchedulerName: pointer.StringPtr("default-scheduler"),
						Plugins:       getDefaultPlugins(),
						PluginConfig:  pluginConfigs,
					},
				},
			},
		},
		{
			name: "two profiles",
			config: &v1beta1.KubeSchedulerConfiguration{
				Parallelism: pointer.Int32Ptr(16),
				Profiles: []v1beta1.KubeSchedulerProfile{
					{
						PluginConfig: []v1beta1.PluginConfig{
							{Name: "FooPlugin"},
						},
					},
					{
						SchedulerName: pointer.StringPtr("custom-scheduler"),
						Plugins: &v1beta1.Plugins{
							Bind: &v1beta1.PluginSet{
								Enabled: []v1beta1.Plugin{
									{Name: "BarPlugin"},
								},
							},
						},
					},
				},
			},
			expected: &v1beta1.KubeSchedulerConfiguration{
				Parallelism:        pointer.Int32Ptr(16),
				HealthzBindAddress: pointer.StringPtr("0.0.0.0:10251"),
				MetricsBindAddress: pointer.StringPtr("0.0.0.0:10251"),
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           &enable,
					EnableContentionProfiling: &enable,
				},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					LeaderElect:       pointer.BoolPtr(true),
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
				PercentageOfNodesToScore: pointer.Int32Ptr(0),
				PodInitialBackoffSeconds: pointer.Int64Ptr(1),
				PodMaxBackoffSeconds:     pointer.Int64Ptr(10),
				Profiles: []v1beta1.KubeSchedulerProfile{
					{
						Plugins: getDefaultPlugins(),
						PluginConfig: []v1beta1.PluginConfig{
							{Name: "FooPlugin"},
							{
								Name: "DefaultPreemption",
								Args: runtime.RawExtension{
									Object: &v1beta1.DefaultPreemptionArgs{
										TypeMeta: metav1.TypeMeta{
											Kind:       "DefaultPreemptionArgs",
											APIVersion: "kubescheduler.config.k8s.io/v1beta1",
										},
										MinCandidateNodesPercentage: pointer.Int32Ptr(10),
										MinCandidateNodesAbsolute:   pointer.Int32Ptr(100),
									}},
							},
							{
								Name: "InterPodAffinity",
								Args: runtime.RawExtension{
									Object: &v1beta1.InterPodAffinityArgs{
										TypeMeta: metav1.TypeMeta{
											Kind:       "InterPodAffinityArgs",
											APIVersion: "kubescheduler.config.k8s.io/v1beta1",
										},
										HardPodAffinityWeight: pointer.Int32Ptr(1),
									}},
							},
							{
								Name: "NodeAffinity",
								Args: runtime.RawExtension{Object: &v1beta1.NodeAffinityArgs{
									TypeMeta: metav1.TypeMeta{
										Kind:       "NodeAffinityArgs",
										APIVersion: "kubescheduler.config.k8s.io/v1beta1",
									},
								}},
							},
							{
								Name: "NodeResourcesBalancedAllocation",
								Args: runtime.RawExtension{Object: &v1beta1.NodeResourcesBalancedAllocationArgs{
									TypeMeta: metav1.TypeMeta{
										Kind:       "NodeResourcesBalancedAllocationArgs",
										APIVersion: "kubescheduler.config.k8s.io/v1beta1",
									},
									Resources: []v1beta1.ResourceSpec{{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1}},
								}},
							},
							{
								Name: "NodeResourcesFit",
								Args: runtime.RawExtension{Object: &v1beta1.NodeResourcesFitArgs{
									TypeMeta: metav1.TypeMeta{
										Kind:       "NodeResourcesFitArgs",
										APIVersion: "kubescheduler.config.k8s.io/v1beta1",
									},
									ScoringStrategy: &v1beta1.ScoringStrategy{
										Type:      v1beta1.LeastAllocated,
										Resources: []v1beta1.ResourceSpec{{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1}},
									},
								}},
							},
							{
								Name: "NodeResourcesLeastAllocated",
								Args: runtime.RawExtension{Object: &v1beta1.NodeResourcesLeastAllocatedArgs{
									TypeMeta: metav1.TypeMeta{
										Kind:       "NodeResourcesLeastAllocatedArgs",
										APIVersion: "kubescheduler.config.k8s.io/v1beta1",
									},
									Resources: []v1beta1.ResourceSpec{{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1}},
								}},
							},
							{
								Name: "PodTopologySpread",
								Args: runtime.RawExtension{Object: &v1beta1.PodTopologySpreadArgs{
									TypeMeta: metav1.TypeMeta{
										Kind:       "PodTopologySpreadArgs",
										APIVersion: "kubescheduler.config.k8s.io/v1beta1",
									},
									DefaultingType: v1beta1.SystemDefaulting,
								}},
							},
							{
								Name: "VolumeBinding",
								Args: runtime.RawExtension{Object: &v1beta1.VolumeBindingArgs{
									TypeMeta: metav1.TypeMeta{
										Kind:       "VolumeBindingArgs",
										APIVersion: "kubescheduler.config.k8s.io/v1beta1",
									},
									BindTimeoutSeconds: pointer.Int64Ptr(600),
								}},
							},
						},
					},
					{
						SchedulerName: pointer.StringPtr("custom-scheduler"),
						Plugins: &v1beta1.Plugins{
							QueueSort: &v1beta1.PluginSet{
								Enabled: []v1beta1.Plugin{
									{Name: names.PrioritySort},
								},
							},
							PreFilter: &v1beta1.PluginSet{
								Enabled: []v1beta1.Plugin{
									{Name: names.NodeResourcesFit},
									{Name: names.NodePorts},
									{Name: names.VolumeRestrictions},
									{Name: names.PodTopologySpread},
									{Name: names.InterPodAffinity},
									{Name: names.VolumeBinding},
									{Name: names.NodeAffinity},
								},
							},
							Filter: &v1beta1.PluginSet{
								Enabled: []v1beta1.Plugin{
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
							PostFilter: &v1beta1.PluginSet{
								Enabled: []v1beta1.Plugin{
									{Name: names.DefaultPreemption},
								},
							},
							PreScore: &v1beta1.PluginSet{
								Enabled: []v1beta1.Plugin{
									{Name: names.InterPodAffinity},
									{Name: names.PodTopologySpread},
									{Name: names.TaintToleration},
									{Name: names.NodeAffinity},
								},
							},
							Score: &v1beta1.PluginSet{
								Enabled: []v1beta1.Plugin{
									{Name: names.NodeResourcesBalancedAllocation, Weight: pointer.Int32Ptr(1)},
									{Name: names.ImageLocality, Weight: pointer.Int32Ptr(1)},
									{Name: names.InterPodAffinity, Weight: pointer.Int32Ptr(1)},
									{Name: names.NodeResourcesLeastAllocated, Weight: pointer.Int32Ptr(1)},
									{Name: names.NodeAffinity, Weight: pointer.Int32Ptr(1)},
									{Name: names.NodePreferAvoidPods, Weight: pointer.Int32Ptr(10000)},
									{Name: names.PodTopologySpread, Weight: pointer.Int32Ptr(2)},
									{Name: names.TaintToleration, Weight: pointer.Int32Ptr(1)},
								},
							},
							Reserve: &v1beta1.PluginSet{
								Enabled: []v1beta1.Plugin{
									{Name: names.VolumeBinding},
								},
							},
							PreBind: &v1beta1.PluginSet{
								Enabled: []v1beta1.Plugin{
									{Name: names.VolumeBinding},
								},
							},
							Bind: &v1beta1.PluginSet{
								Enabled: []v1beta1.Plugin{
									{Name: names.DefaultBinder},
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
			name: "metrics and healthz address with no port",
			config: &v1beta1.KubeSchedulerConfiguration{
				Parallelism:        pointer.Int32Ptr(16),
				MetricsBindAddress: pointer.StringPtr("1.2.3.4"),
				HealthzBindAddress: pointer.StringPtr("1.2.3.4"),
			},
			expected: &v1beta1.KubeSchedulerConfiguration{
				Parallelism:        pointer.Int32Ptr(16),
				HealthzBindAddress: pointer.StringPtr("1.2.3.4:10251"),
				MetricsBindAddress: pointer.StringPtr("1.2.3.4:10251"),
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           &enable,
					EnableContentionProfiling: &enable,
				},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					LeaderElect:       pointer.BoolPtr(true),
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
				PercentageOfNodesToScore: pointer.Int32Ptr(0),
				PodInitialBackoffSeconds: pointer.Int64Ptr(1),
				PodMaxBackoffSeconds:     pointer.Int64Ptr(10),
				Profiles: []v1beta1.KubeSchedulerProfile{
					{
						SchedulerName: pointer.StringPtr("default-scheduler"),
						Plugins:       getDefaultPlugins(),
						PluginConfig:  pluginConfigs,
					},
				},
			},
		},
		{
			name: "metrics and healthz port with no address",
			config: &v1beta1.KubeSchedulerConfiguration{
				MetricsBindAddress: pointer.StringPtr(":12345"),
				HealthzBindAddress: pointer.StringPtr(":12345"),
			},
			expected: &v1beta1.KubeSchedulerConfiguration{
				Parallelism:        pointer.Int32Ptr(16),
				HealthzBindAddress: pointer.StringPtr("0.0.0.0:12345"),
				MetricsBindAddress: pointer.StringPtr("0.0.0.0:12345"),
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           &enable,
					EnableContentionProfiling: &enable,
				},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					LeaderElect:       pointer.BoolPtr(true),
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
				PercentageOfNodesToScore: pointer.Int32Ptr(0),
				PodInitialBackoffSeconds: pointer.Int64Ptr(1),
				PodMaxBackoffSeconds:     pointer.Int64Ptr(10),
				Profiles: []v1beta1.KubeSchedulerProfile{
					{
						SchedulerName: pointer.StringPtr("default-scheduler"),
						Plugins:       getDefaultPlugins(),
						PluginConfig:  pluginConfigs,
					},
				},
			},
		},
		{
			name: "set non default parallelism",
			config: &v1beta1.KubeSchedulerConfiguration{
				Parallelism: pointer.Int32Ptr(8),
			},
			expected: &v1beta1.KubeSchedulerConfiguration{
				Parallelism:        pointer.Int32Ptr(8),
				HealthzBindAddress: pointer.StringPtr("0.0.0.0:10251"),
				MetricsBindAddress: pointer.StringPtr("0.0.0.0:10251"),
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           &enable,
					EnableContentionProfiling: &enable,
				},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					LeaderElect:       pointer.BoolPtr(true),
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
				PercentageOfNodesToScore: pointer.Int32Ptr(0),
				PodInitialBackoffSeconds: pointer.Int64Ptr(1),
				PodMaxBackoffSeconds:     pointer.Int64Ptr(10),
				Profiles: []v1beta1.KubeSchedulerProfile{
					{
						SchedulerName: pointer.StringPtr("default-scheduler"),
						Plugins:       getDefaultPlugins(),
						PluginConfig:  pluginConfigs,
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
			in:   &v1beta1.DefaultPreemptionArgs{},
			want: &v1beta1.DefaultPreemptionArgs{
				MinCandidateNodesPercentage: pointer.Int32Ptr(10),
				MinCandidateNodesAbsolute:   pointer.Int32Ptr(100),
			},
		},
		{
			name: "DefaultPreemptionArgs with value",
			in: &v1beta1.DefaultPreemptionArgs{
				MinCandidateNodesPercentage: pointer.Int32Ptr(50),
			},
			want: &v1beta1.DefaultPreemptionArgs{
				MinCandidateNodesPercentage: pointer.Int32Ptr(50),
				MinCandidateNodesAbsolute:   pointer.Int32Ptr(100),
			},
		},
		{
			name: "InterPodAffinityArgs empty",
			in:   &v1beta1.InterPodAffinityArgs{},
			want: &v1beta1.InterPodAffinityArgs{
				HardPodAffinityWeight: pointer.Int32Ptr(1),
			},
		},
		{
			name: "InterPodAffinityArgs explicit 0",
			in: &v1beta1.InterPodAffinityArgs{
				HardPodAffinityWeight: pointer.Int32Ptr(0),
			},
			want: &v1beta1.InterPodAffinityArgs{
				HardPodAffinityWeight: pointer.Int32Ptr(0),
			},
		},
		{
			name: "InterPodAffinityArgs with value",
			in: &v1beta1.InterPodAffinityArgs{
				HardPodAffinityWeight: pointer.Int32Ptr(5),
			},
			want: &v1beta1.InterPodAffinityArgs{
				HardPodAffinityWeight: pointer.Int32Ptr(5),
			},
		},
		{
			name: "NodeResourcesLeastAllocatedArgs resources empty",
			in:   &v1beta1.NodeResourcesLeastAllocatedArgs{},
			want: &v1beta1.NodeResourcesLeastAllocatedArgs{
				Resources: []v1beta1.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
				},
			},
		},
		{
			name: "NodeResourcesLeastAllocatedArgs resources with value",
			in: &v1beta1.NodeResourcesLeastAllocatedArgs{
				Resources: []v1beta1.ResourceSpec{
					{Name: "resource", Weight: 2},
				},
			},
			want: &v1beta1.NodeResourcesLeastAllocatedArgs{
				Resources: []v1beta1.ResourceSpec{
					{Name: "resource", Weight: 2},
				},
			},
		},
		{
			name: "NodeResourcesMostAllocatedArgs resources empty",
			in:   &v1beta1.NodeResourcesMostAllocatedArgs{},
			want: &v1beta1.NodeResourcesMostAllocatedArgs{
				Resources: []v1beta1.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
				},
			},
		},
		{
			name: "NodeResourcesMostAllocatedArgs resources with value",
			in: &v1beta1.NodeResourcesMostAllocatedArgs{
				Resources: []v1beta1.ResourceSpec{
					{Name: "resource", Weight: 2},
				},
			},
			want: &v1beta1.NodeResourcesMostAllocatedArgs{
				Resources: []v1beta1.ResourceSpec{
					{Name: "resource", Weight: 2},
				},
			},
		},
		{
			name: "NodeResourcesBalancedAllocationArgs resources empty",
			in:   &v1beta1.NodeResourcesBalancedAllocationArgs{},
			want: &v1beta1.NodeResourcesBalancedAllocationArgs{
				Resources: []v1beta1.ResourceSpec{
					{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1},
				},
			},
		},
		{
			name: "NodeResourcesBalancedAllocationArgs with scalar resource",
			in: &v1beta1.NodeResourcesBalancedAllocationArgs{
				Resources: []v1beta1.ResourceSpec{
					{Name: "scalar.io/scalar1", Weight: 1},
				},
			},
			want: &v1beta1.NodeResourcesBalancedAllocationArgs{
				Resources: []v1beta1.ResourceSpec{
					{Name: "scalar.io/scalar1", Weight: 1},
				},
			},
		},
		{
			name: "NodeResourcesBalancedAllocationArgs with mixed resources",
			in: &v1beta1.NodeResourcesBalancedAllocationArgs{
				Resources: []v1beta1.ResourceSpec{
					{Name: string(v1.ResourceCPU), Weight: 1},
					{Name: "scalar.io/scalar1", Weight: 1},
				},
			},
			want: &v1beta1.NodeResourcesBalancedAllocationArgs{
				Resources: []v1beta1.ResourceSpec{
					{Name: string(v1.ResourceCPU), Weight: 1},
					{Name: "scalar.io/scalar1", Weight: 1},
				},
			},
		},
		{
			name: "NodeResourcesBalancedAllocationArgs have resource no weight",
			in: &v1beta1.NodeResourcesBalancedAllocationArgs{
				Resources: []v1beta1.ResourceSpec{
					{Name: string(v1.ResourceCPU)},
					{Name: "scalar.io/scalar0"},
					{Name: "scalar.io/scalar1", Weight: 1},
				},
			},
			want: &v1beta1.NodeResourcesBalancedAllocationArgs{
				Resources: []v1beta1.ResourceSpec{
					{Name: string(v1.ResourceCPU), Weight: 1},
					{Name: "scalar.io/scalar0", Weight: 1},
					{Name: "scalar.io/scalar1", Weight: 1},
				},
			},
		},
		{
			name: "PodTopologySpreadArgs resources empty",
			in:   &v1beta1.PodTopologySpreadArgs{},
			want: &v1beta1.PodTopologySpreadArgs{
				DefaultingType: v1beta1.SystemDefaulting,
			},
		},
		{
			name: "PodTopologySpreadArgs resources with value",
			in: &v1beta1.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						TopologyKey:       "planet",
						WhenUnsatisfiable: v1.DoNotSchedule,
						MaxSkew:           2,
					},
				},
			},
			want: &v1beta1.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						TopologyKey:       "planet",
						WhenUnsatisfiable: v1.DoNotSchedule,
						MaxSkew:           2,
					},
				},
				// TODO(#94008): Make SystemDefaulting in v1beta2.
				DefaultingType: v1beta1.ListDefaulting,
			},
		},
		{
			name: "PodTopologySpreadArgs empty, DefaultPodTopologySpread feature disabled",
			features: map[featuregate.Feature]bool{
				features.DefaultPodTopologySpread: false,
			},
			in: &v1beta1.PodTopologySpreadArgs{},
			want: &v1beta1.PodTopologySpreadArgs{
				DefaultingType: v1beta1.ListDefaulting,
			},
		},
		{
			name: "NodeResourcesFitArgs not set",
			in:   &v1beta1.NodeResourcesFitArgs{},
			want: &v1beta1.NodeResourcesFitArgs{
				ScoringStrategy: &v1beta1.ScoringStrategy{
					Type:      v1beta1.LeastAllocated,
					Resources: defaultResourceSpec,
				},
			},
		},
		{
			name: "NodeResourcesFitArgs Resources empty",
			in: &v1beta1.NodeResourcesFitArgs{
				ScoringStrategy: &v1beta1.ScoringStrategy{
					Type: v1beta1.MostAllocated,
				},
			},
			want: &v1beta1.NodeResourcesFitArgs{
				ScoringStrategy: &v1beta1.ScoringStrategy{
					Type:      v1beta1.MostAllocated,
					Resources: defaultResourceSpec,
				},
			},
		},
		{
			name: "VolumeBindingArgs empty, VolumeCapacityPriority disabled",
			features: map[featuregate.Feature]bool{
				features.VolumeCapacityPriority: false,
			},
			in: &v1beta1.VolumeBindingArgs{},
			want: &v1beta1.VolumeBindingArgs{
				BindTimeoutSeconds: pointer.Int64Ptr(600),
			},
		},
		{
			name: "VolumeBindingArgs empty, VolumeCapacityPriority enabled",
			features: map[featuregate.Feature]bool{
				features.VolumeCapacityPriority: true,
			},
			in: &v1beta1.VolumeBindingArgs{},
			want: &v1beta1.VolumeBindingArgs{
				BindTimeoutSeconds: pointer.Int64Ptr(600),
				Shape: []v1beta1.UtilizationShapePoint{
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
