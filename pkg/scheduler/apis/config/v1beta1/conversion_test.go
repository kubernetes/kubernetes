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

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kube-scheduler/config/v1beta1"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/utils/pointer"
)

func TestV1beta1ToConfigKubeSchedulerConfigurationConversion(t *testing.T) {
	cases := []struct {
		name   string
		config v1beta1.KubeSchedulerConfiguration
		want   config.KubeSchedulerConfiguration
	}{
		{
			name:   "default conversion v1beta1 to config",
			config: v1beta1.KubeSchedulerConfiguration{},
			want:   config.KubeSchedulerConfiguration{AlgorithmSource: config.SchedulerAlgorithmSource{Provider: pointer.StringPtr(v1beta1.SchedulerDefaultProviderName)}},
		},
		{
			name: "Not qualified plugins conversion v1beta1 to config",
			config: v1beta1.KubeSchedulerConfiguration{
				Profiles: []v1beta1.KubeSchedulerProfile{
					{
						PluginConfig: []v1beta1.PluginConfig{
							{
								Name: "InterPodAffinity",
								Args: runtime.RawExtension{
									Object: &v1beta1.InterPodAffinityArgs{
										HardPodAffinityWeight: pointer.Int32Ptr(5),
									},
								},
							},
							{
								Name: "VolumeBinding",
								Args: runtime.RawExtension{
									Object: &v1beta1.VolumeBindingArgs{
										BindTimeoutSeconds: pointer.Int64Ptr(300),
									},
								},
							},
							{
								Name: "RequestedToCapacityRatio",
								Args: runtime.RawExtension{
									Object: &v1beta1.RequestedToCapacityRatioArgs{
										Shape: []v1beta1.UtilizationShapePoint{
											{Utilization: 1, Score: 2},
										},
										Resources: []v1beta1.ResourceSpec{
											{Name: "cpu", Weight: 2},
										},
									},
								},
							},
							{
								Name: "NodeResourcesLeastAllocated",
								Args: runtime.RawExtension{
									Object: &v1beta1.NodeResourcesLeastAllocatedArgs{
										Resources: []v1beta1.ResourceSpec{
											{Name: "mem", Weight: 2},
										},
									},
								},
							},
							{
								Name: "PodTopologySpread",
								Args: runtime.RawExtension{
									Object: &v1beta1.PodTopologySpreadArgs{
										DefaultConstraints: []corev1.TopologySpreadConstraint{},
									},
								},
							},
							{
								Name: "OutOfTreePlugin",
								Args: runtime.RawExtension{
									Raw: []byte(`{"foo":"bar"}`),
								},
							},
						},
					},
				},
			},
			want: config.KubeSchedulerConfiguration{
				AlgorithmSource: config.SchedulerAlgorithmSource{Provider: pointer.StringPtr(v1beta1.SchedulerDefaultProviderName)},
				Profiles: []config.KubeSchedulerProfile{
					{
						PluginConfig: []config.PluginConfig{
							{
								Name: "plugin.kubescheduler.k8s.io/InterPodAffinity",
								Args: &config.InterPodAffinityArgs{
									HardPodAffinityWeight: 5,
								},
							},
							{
								Name: "plugin.kubescheduler.k8s.io/VolumeBinding",
								Args: &config.VolumeBindingArgs{
									BindTimeoutSeconds: 300,
								},
							},
							{
								Name: "plugin.kubescheduler.k8s.io/RequestedToCapacityRatio",
								Args: &config.RequestedToCapacityRatioArgs{
									Shape: []config.UtilizationShapePoint{
										{Utilization: 1, Score: 2},
									},
									Resources: []config.ResourceSpec{
										{Name: "cpu", Weight: 2},
									},
								},
							},
							{
								Name: "plugin.kubescheduler.k8s.io/NodeResourcesLeastAllocated",
								Args: &config.NodeResourcesLeastAllocatedArgs{
									Resources: []config.ResourceSpec{
										{Name: "mem", Weight: 2},
									},
								},
							},
							{
								Name: "plugin.kubescheduler.k8s.io/PodTopologySpread",
								Args: &config.PodTopologySpreadArgs{
									DefaultConstraints: []corev1.TopologySpreadConstraint{},
									DefaultingType:     config.SystemDefaulting,
								},
							},
							{
								Name: "OutOfTreePlugin",
								Args: &runtime.Unknown{
									Raw:         []byte(`{"foo":"bar"}`),
									ContentType: "application/json",
								},
							},
						},
					},
				},
			},
		},
		{
			name: "Qualified plugins conversion v1beta1 to config",
			config: v1beta1.KubeSchedulerConfiguration{
				Profiles: []v1beta1.KubeSchedulerProfile{
					{
						PluginConfig: []v1beta1.PluginConfig{
							{
								Name: "plugin.kubescheduler.k8s.io/InterPodAffinity",
								Args: runtime.RawExtension{
									Object: &v1beta1.InterPodAffinityArgs{
										HardPodAffinityWeight: pointer.Int32Ptr(5),
									},
								},
							},
							{
								Name: "plugin.kubescheduler.k8s.io/VolumeBinding",
								Args: runtime.RawExtension{
									Object: &v1beta1.VolumeBindingArgs{
										BindTimeoutSeconds: pointer.Int64Ptr(300),
									},
								},
							},
							{
								Name: "plugin.kubescheduler.k8s.io/RequestedToCapacityRatio",
								Args: runtime.RawExtension{
									Object: &v1beta1.RequestedToCapacityRatioArgs{
										Shape: []v1beta1.UtilizationShapePoint{
											{Utilization: 1, Score: 2},
										},
										Resources: []v1beta1.ResourceSpec{
											{Name: "cpu", Weight: 2},
										},
									},
								},
							},
							{
								Name: "plugin.kubescheduler.k8s.io/NodeResourcesLeastAllocated",
								Args: runtime.RawExtension{
									Object: &v1beta1.NodeResourcesLeastAllocatedArgs{
										Resources: []v1beta1.ResourceSpec{
											{Name: "mem", Weight: 2},
										},
									},
								},
							},
							{
								Name: "plugin.kubescheduler.k8s.io/PodTopologySpread",
								Args: runtime.RawExtension{
									Object: &v1beta1.PodTopologySpreadArgs{
										DefaultConstraints: []corev1.TopologySpreadConstraint{},
									},
								},
							},
							{
								Name: "OutOfTreePlugin",
								Args: runtime.RawExtension{
									Raw: []byte(`{"foo":"bar"}`),
								},
							},
						},
					},
				},
			},
			want: config.KubeSchedulerConfiguration{
				AlgorithmSource: config.SchedulerAlgorithmSource{Provider: pointer.StringPtr(v1beta1.SchedulerDefaultProviderName)},
				Profiles: []config.KubeSchedulerProfile{
					{
						PluginConfig: []config.PluginConfig{
							{
								Name: "plugin.kubescheduler.k8s.io/InterPodAffinity",
								Args: &config.InterPodAffinityArgs{
									HardPodAffinityWeight: 5,
								},
							},
							{
								Name: "plugin.kubescheduler.k8s.io/VolumeBinding",
								Args: &config.VolumeBindingArgs{
									BindTimeoutSeconds: 300,
								},
							},
							{
								Name: "plugin.kubescheduler.k8s.io/RequestedToCapacityRatio",
								Args: &config.RequestedToCapacityRatioArgs{
									Shape: []config.UtilizationShapePoint{
										{Utilization: 1, Score: 2},
									},
									Resources: []config.ResourceSpec{
										{Name: "cpu", Weight: 2},
									},
								},
							},
							{
								Name: "plugin.kubescheduler.k8s.io/NodeResourcesLeastAllocated",
								Args: &config.NodeResourcesLeastAllocatedArgs{
									Resources: []config.ResourceSpec{
										{Name: "mem", Weight: 2},
									},
								},
							},
							{
								Name: "plugin.kubescheduler.k8s.io/PodTopologySpread",
								Args: &config.PodTopologySpreadArgs{
									DefaultConstraints: []corev1.TopologySpreadConstraint{},
									DefaultingType:     config.SystemDefaulting,
								},
							},
							{
								Name: "OutOfTreePlugin",
								Args: &runtime.Unknown{
									Raw:         []byte(`{"foo":"bar"}`),
									ContentType: "application/json",
								},
							},
						},
					},
				},
			},
		},
	}

	scheme := runtime.NewScheme()
	if err := AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got config.KubeSchedulerConfiguration
			if err := scheme.Convert(&tc.config, &got, nil); err != nil {
				t.Errorf("failed to convert: %+v", err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("unexpected conversion (-want, +got):\n%s", diff)
			}
		})
	}
}
