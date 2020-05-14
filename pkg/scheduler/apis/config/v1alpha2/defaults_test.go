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

package v1alpha2

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	componentbaseconfig "k8s.io/component-base/config/v1alpha1"
	"k8s.io/kube-scheduler/config/v1alpha2"
	"k8s.io/utils/pointer"
)

func TestSchedulerDefaults(t *testing.T) {
	enable := true
	tests := []struct {
		name     string
		config   *v1alpha2.KubeSchedulerConfiguration
		expected *v1alpha2.KubeSchedulerConfiguration
	}{
		{
			name:   "empty config",
			config: &v1alpha2.KubeSchedulerConfiguration{},
			expected: &v1alpha2.KubeSchedulerConfiguration{
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
					ResourceLock:      "endpointsleases",
					ResourceNamespace: "kube-system",
					ResourceName:      "kube-scheduler",
				},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					QPS:         50,
					Burst:       100,
					ContentType: "application/vnd.kubernetes.protobuf",
				},
				DisablePreemption:        pointer.BoolPtr(false),
				PercentageOfNodesToScore: pointer.Int32Ptr(0),
				BindTimeoutSeconds:       pointer.Int64Ptr(600),
				PodInitialBackoffSeconds: pointer.Int64Ptr(1),
				PodMaxBackoffSeconds:     pointer.Int64Ptr(10),
				Profiles: []v1alpha2.KubeSchedulerProfile{
					{SchedulerName: pointer.StringPtr("default-scheduler")},
				},
			},
		},
		{
			name: "no scheduler name",
			config: &v1alpha2.KubeSchedulerConfiguration{
				Profiles: []v1alpha2.KubeSchedulerProfile{
					{
						PluginConfig: []v1alpha2.PluginConfig{
							{Name: "FooPlugin"},
						},
					},
				},
			},
			expected: &v1alpha2.KubeSchedulerConfiguration{
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
					ResourceLock:      "endpointsleases",
					ResourceNamespace: "kube-system",
					ResourceName:      "kube-scheduler",
				},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					QPS:         50,
					Burst:       100,
					ContentType: "application/vnd.kubernetes.protobuf",
				},
				DisablePreemption:        pointer.BoolPtr(false),
				PercentageOfNodesToScore: pointer.Int32Ptr(0),
				BindTimeoutSeconds:       pointer.Int64Ptr(600),
				PodInitialBackoffSeconds: pointer.Int64Ptr(1),
				PodMaxBackoffSeconds:     pointer.Int64Ptr(10),
				Profiles: []v1alpha2.KubeSchedulerProfile{
					{
						SchedulerName: pointer.StringPtr("default-scheduler"),
						PluginConfig: []v1alpha2.PluginConfig{
							{Name: "FooPlugin"},
						},
					},
				},
			},
		},
		{
			name: "two profiles",
			config: &v1alpha2.KubeSchedulerConfiguration{
				Profiles: []v1alpha2.KubeSchedulerProfile{
					{
						PluginConfig: []v1alpha2.PluginConfig{
							{Name: "FooPlugin"},
						},
					},
					{
						SchedulerName: pointer.StringPtr("custom-scheduler"),
						Plugins: &v1alpha2.Plugins{
							Bind: &v1alpha2.PluginSet{
								Enabled: []v1alpha2.Plugin{
									{Name: "BarPlugin"},
								},
							},
						},
					},
				},
			},
			expected: &v1alpha2.KubeSchedulerConfiguration{
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
					ResourceLock:      "endpointsleases",
					ResourceNamespace: "kube-system",
					ResourceName:      "kube-scheduler",
				},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					QPS:         50,
					Burst:       100,
					ContentType: "application/vnd.kubernetes.protobuf",
				},
				DisablePreemption:        pointer.BoolPtr(false),
				PercentageOfNodesToScore: pointer.Int32Ptr(0),
				BindTimeoutSeconds:       pointer.Int64Ptr(600),
				PodInitialBackoffSeconds: pointer.Int64Ptr(1),
				PodMaxBackoffSeconds:     pointer.Int64Ptr(10),
				Profiles: []v1alpha2.KubeSchedulerProfile{
					{
						PluginConfig: []v1alpha2.PluginConfig{
							{Name: "FooPlugin"},
						},
					},
					{
						SchedulerName: pointer.StringPtr("custom-scheduler"),
						Plugins: &v1alpha2.Plugins{
							Bind: &v1alpha2.PluginSet{
								Enabled: []v1alpha2.Plugin{
									{Name: "BarPlugin"},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "metrics and healthz address with no port",
			config: &v1alpha2.KubeSchedulerConfiguration{
				MetricsBindAddress: pointer.StringPtr("1.2.3.4"),
				HealthzBindAddress: pointer.StringPtr("1.2.3.4"),
			},
			expected: &v1alpha2.KubeSchedulerConfiguration{
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
					ResourceLock:      "endpointsleases",
					ResourceNamespace: "kube-system",
					ResourceName:      "kube-scheduler",
				},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					QPS:         50,
					Burst:       100,
					ContentType: "application/vnd.kubernetes.protobuf",
				},
				DisablePreemption:        pointer.BoolPtr(false),
				PercentageOfNodesToScore: pointer.Int32Ptr(0),
				BindTimeoutSeconds:       pointer.Int64Ptr(600),
				PodInitialBackoffSeconds: pointer.Int64Ptr(1),
				PodMaxBackoffSeconds:     pointer.Int64Ptr(10),
				Profiles: []v1alpha2.KubeSchedulerProfile{
					{SchedulerName: pointer.StringPtr("default-scheduler")},
				},
			},
		},
		{
			name: "metrics and healthz port with no address",
			config: &v1alpha2.KubeSchedulerConfiguration{
				MetricsBindAddress: pointer.StringPtr(":12345"),
				HealthzBindAddress: pointer.StringPtr(":12345"),
			},
			expected: &v1alpha2.KubeSchedulerConfiguration{
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
					ResourceLock:      "endpointsleases",
					ResourceNamespace: "kube-system",
					ResourceName:      "kube-scheduler",
				},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					QPS:         50,
					Burst:       100,
					ContentType: "application/vnd.kubernetes.protobuf",
				},
				DisablePreemption:        pointer.BoolPtr(false),
				PercentageOfNodesToScore: pointer.Int32Ptr(0),
				BindTimeoutSeconds:       pointer.Int64Ptr(600),
				PodInitialBackoffSeconds: pointer.Int64Ptr(1),
				PodMaxBackoffSeconds:     pointer.Int64Ptr(10),
				Profiles: []v1alpha2.KubeSchedulerProfile{
					{SchedulerName: pointer.StringPtr("default-scheduler")},
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
		name string
		in   runtime.Object
		want runtime.Object
	}{
		{
			name: "InterPodAffinityArgs empty",
			in:   &v1alpha2.InterPodAffinityArgs{},
			want: &v1alpha2.InterPodAffinityArgs{
				HardPodAffinityWeight: pointer.Int32Ptr(1),
			},
		},
		{
			name: "InterPodAffinityArgs explicit 0",
			in: &v1alpha2.InterPodAffinityArgs{
				HardPodAffinityWeight: pointer.Int32Ptr(0),
			},
			want: &v1alpha2.InterPodAffinityArgs{
				HardPodAffinityWeight: pointer.Int32Ptr(0),
			},
		},
		{
			name: "InterPodAffinityArgs with value",
			in: &v1alpha2.InterPodAffinityArgs{
				HardPodAffinityWeight: pointer.Int32Ptr(5),
			},
			want: &v1alpha2.InterPodAffinityArgs{
				HardPodAffinityWeight: pointer.Int32Ptr(5),
			},
		},
		{
			name: "NodeResourcesLeastAllocatedArgs resources empty",
			in:   &v1alpha2.NodeResourcesLeastAllocatedArgs{},
			want: &v1alpha2.NodeResourcesLeastAllocatedArgs{
				Resources: []v1alpha2.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
				},
			},
		},
		{
			name: "NodeResourcesLeastAllocatedArgs resources with value",
			in: &v1alpha2.NodeResourcesLeastAllocatedArgs{
				Resources: []v1alpha2.ResourceSpec{
					{Name: "resource", Weight: 2},
				},
			},
			want: &v1alpha2.NodeResourcesLeastAllocatedArgs{
				Resources: []v1alpha2.ResourceSpec{
					{Name: "resource", Weight: 2},
				},
			},
		},
		{
			name: "NodeResourcesMostAllocatedArgs resources empty",
			in:   &v1alpha2.NodeResourcesMostAllocatedArgs{},
			want: &v1alpha2.NodeResourcesMostAllocatedArgs{
				Resources: []v1alpha2.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
				},
			},
		},
		{
			name: "NodeResourcesMostAllocatedArgs resources with value",
			in: &v1alpha2.NodeResourcesMostAllocatedArgs{
				Resources: []v1alpha2.ResourceSpec{
					{Name: "resource", Weight: 2},
				},
			},
			want: &v1alpha2.NodeResourcesMostAllocatedArgs{
				Resources: []v1alpha2.ResourceSpec{
					{Name: "resource", Weight: 2},
				},
			},
		},
		{
			name: "NodeResourcesMostAllocatedArgs resources empty",
			in:   &v1alpha2.NodeResourcesMostAllocatedArgs{},
			want: &v1alpha2.NodeResourcesMostAllocatedArgs{
				Resources: []v1alpha2.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
				},
			},
		},
		{
			name: "NodeResourcesMostAllocatedArgs resources with value",
			in: &v1alpha2.NodeResourcesMostAllocatedArgs{
				Resources: []v1alpha2.ResourceSpec{
					{Name: "resource", Weight: 2},
				},
			},
			want: &v1alpha2.NodeResourcesMostAllocatedArgs{
				Resources: []v1alpha2.ResourceSpec{
					{Name: "resource", Weight: 2},
				},
			},
		},
	}
	for _, tc := range tests {
		scheme := runtime.NewScheme()
		utilruntime.Must(AddToScheme(scheme))
		t.Run(tc.name, func(t *testing.T) {
			scheme.Default(tc.in)
			if diff := cmp.Diff(tc.in, tc.want); diff != "" {
				t.Errorf("Got unexpected defaults (-want, +got):\n%s", diff)
			}
		})
	}
}
