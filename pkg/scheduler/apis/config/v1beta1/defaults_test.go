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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	componentbaseconfig "k8s.io/component-base/config/v1alpha1"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kube-scheduler/config/v1beta1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/pointer"
)

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
				PodInitialBackoffSeconds: pointer.Int64Ptr(1),
				PodMaxBackoffSeconds:     pointer.Int64Ptr(10),
				Profiles: []v1beta1.KubeSchedulerProfile{
					{SchedulerName: pointer.StringPtr("default-scheduler")},
				},
			},
		},
		{
			name: "no scheduler name",
			config: &v1beta1.KubeSchedulerConfiguration{
				Profiles: []v1beta1.KubeSchedulerProfile{
					{
						PluginConfig: []v1beta1.PluginConfig{
							{Name: "FooPlugin"},
						},
					},
				},
			},
			expected: &v1beta1.KubeSchedulerConfiguration{
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
				PodInitialBackoffSeconds: pointer.Int64Ptr(1),
				PodMaxBackoffSeconds:     pointer.Int64Ptr(10),
				Profiles: []v1beta1.KubeSchedulerProfile{
					{
						SchedulerName: pointer.StringPtr("default-scheduler"),
						PluginConfig: []v1beta1.PluginConfig{
							{Name: "FooPlugin"},
						},
					},
				},
			},
		},
		{
			name: "two profiles",
			config: &v1beta1.KubeSchedulerConfiguration{
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
				PodInitialBackoffSeconds: pointer.Int64Ptr(1),
				PodMaxBackoffSeconds:     pointer.Int64Ptr(10),
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
		},
		{
			name: "metrics and healthz address with no port",
			config: &v1beta1.KubeSchedulerConfiguration{
				MetricsBindAddress: pointer.StringPtr("1.2.3.4"),
				HealthzBindAddress: pointer.StringPtr("1.2.3.4"),
			},
			expected: &v1beta1.KubeSchedulerConfiguration{
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
				PodInitialBackoffSeconds: pointer.Int64Ptr(1),
				PodMaxBackoffSeconds:     pointer.Int64Ptr(10),
				Profiles: []v1beta1.KubeSchedulerProfile{
					{SchedulerName: pointer.StringPtr("default-scheduler")},
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
				PodInitialBackoffSeconds: pointer.Int64Ptr(1),
				PodMaxBackoffSeconds:     pointer.Int64Ptr(10),
				Profiles: []v1beta1.KubeSchedulerProfile{
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
		name    string
		feature featuregate.Feature
		in      runtime.Object
		want    runtime.Object
	}{
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
			name: "PodTopologySpreadArgs resources empty",
			in:   &v1beta1.PodTopologySpreadArgs{},
			want: &v1beta1.PodTopologySpreadArgs{},
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
			},
		},
		{
			name:    "PodTopologySpreadArgs resources empty, NewPodTopologySpread feature enabled",
			feature: features.DefaultPodTopologySpread,
			in:      &v1beta1.PodTopologySpreadArgs{},
			want: &v1beta1.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						TopologyKey:       v1.LabelHostname,
						WhenUnsatisfiable: v1.ScheduleAnyway,
						MaxSkew:           3,
					},
					{
						TopologyKey:       v1.LabelZoneFailureDomainStable,
						WhenUnsatisfiable: v1.ScheduleAnyway,
						MaxSkew:           5,
					},
				},
			},
		},
	}
	for _, tc := range tests {
		scheme := runtime.NewScheme()
		utilruntime.Must(AddToScheme(scheme))
		t.Run(tc.name, func(t *testing.T) {
			if tc.feature != "" {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, tc.feature, true)()
			}
			scheme.Default(tc.in)
			if diff := cmp.Diff(tc.in, tc.want); diff != "" {
				t.Errorf("Got unexpected defaults (-want, +got):\n%s", diff)
			}
		})
	}
}
