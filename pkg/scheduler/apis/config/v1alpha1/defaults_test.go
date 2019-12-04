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

package v1alpha1

import (
	"reflect"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	componentbaseconfig "k8s.io/component-base/config/v1alpha1"
	kubeschedulerconfigv1alpha1 "k8s.io/kube-scheduler/config/v1alpha1"
	"k8s.io/utils/pointer"
)

func TestSchedulerDefaults(t *testing.T) {
	enable := true
	tests := []struct {
		name     string
		config   *kubeschedulerconfigv1alpha1.KubeSchedulerConfiguration
		expected *kubeschedulerconfigv1alpha1.KubeSchedulerConfiguration
	}{
		{
			name:   "empty config",
			config: &kubeschedulerconfigv1alpha1.KubeSchedulerConfiguration{},
			expected: &kubeschedulerconfigv1alpha1.KubeSchedulerConfiguration{
				SchedulerName:                  pointer.StringPtr("default-scheduler"),
				AlgorithmSource:                kubeschedulerconfigv1alpha1.SchedulerAlgorithmSource{Provider: pointer.StringPtr("DefaultProvider")},
				HardPodAffinitySymmetricWeight: pointer.Int32Ptr(1),
				HealthzBindAddress:             pointer.StringPtr("0.0.0.0:10251"),
				MetricsBindAddress:             pointer.StringPtr("0.0.0.0:10251"),
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           &enable,
					EnableContentionProfiling: &enable,
				},
				LeaderElection: kubeschedulerconfigv1alpha1.KubeSchedulerLeaderElectionConfiguration{
					LeaderElectionConfiguration: componentbaseconfig.LeaderElectionConfiguration{
						LeaderElect:       pointer.BoolPtr(true),
						LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
						RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
						RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
						ResourceLock:      "endpointsleases",
						ResourceNamespace: "",
						ResourceName:      "",
					},
					LockObjectName:      "kube-scheduler",
					LockObjectNamespace: "kube-system",
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
				Plugins:                  nil,
			},
		},
		{
			name: "metrics and healthz address with no port",
			config: &kubeschedulerconfigv1alpha1.KubeSchedulerConfiguration{
				MetricsBindAddress: pointer.StringPtr("1.2.3.4"),
				HealthzBindAddress: pointer.StringPtr("1.2.3.4"),
			},
			expected: &kubeschedulerconfigv1alpha1.KubeSchedulerConfiguration{
				SchedulerName:                  pointer.StringPtr("default-scheduler"),
				AlgorithmSource:                kubeschedulerconfigv1alpha1.SchedulerAlgorithmSource{Provider: pointer.StringPtr("DefaultProvider")},
				HardPodAffinitySymmetricWeight: pointer.Int32Ptr(1),
				HealthzBindAddress:             pointer.StringPtr("1.2.3.4:10251"),
				MetricsBindAddress:             pointer.StringPtr("1.2.3.4:10251"),
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           &enable,
					EnableContentionProfiling: &enable,
				},
				LeaderElection: kubeschedulerconfigv1alpha1.KubeSchedulerLeaderElectionConfiguration{
					LeaderElectionConfiguration: componentbaseconfig.LeaderElectionConfiguration{
						LeaderElect:       pointer.BoolPtr(true),
						LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
						RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
						RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
						ResourceLock:      "endpointsleases",
						ResourceNamespace: "",
						ResourceName:      "",
					},
					LockObjectName:      "kube-scheduler",
					LockObjectNamespace: "kube-system",
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
				Plugins:                  nil,
			},
		},
		{
			name: "metrics and healthz port with no address",
			config: &kubeschedulerconfigv1alpha1.KubeSchedulerConfiguration{
				MetricsBindAddress: pointer.StringPtr(":12345"),
				HealthzBindAddress: pointer.StringPtr(":12345"),
			},
			expected: &kubeschedulerconfigv1alpha1.KubeSchedulerConfiguration{
				SchedulerName:                  pointer.StringPtr("default-scheduler"),
				AlgorithmSource:                kubeschedulerconfigv1alpha1.SchedulerAlgorithmSource{Provider: pointer.StringPtr("DefaultProvider")},
				HardPodAffinitySymmetricWeight: pointer.Int32Ptr(1),
				HealthzBindAddress:             pointer.StringPtr("0.0.0.0:12345"),
				MetricsBindAddress:             pointer.StringPtr("0.0.0.0:12345"),
				DebuggingConfiguration: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           &enable,
					EnableContentionProfiling: &enable,
				},
				LeaderElection: kubeschedulerconfigv1alpha1.KubeSchedulerLeaderElectionConfiguration{
					LeaderElectionConfiguration: componentbaseconfig.LeaderElectionConfiguration{
						LeaderElect:       pointer.BoolPtr(true),
						LeaseDuration:     metav1.Duration{Duration: 15 * time.Second},
						RenewDeadline:     metav1.Duration{Duration: 10 * time.Second},
						RetryPeriod:       metav1.Duration{Duration: 2 * time.Second},
						ResourceLock:      "endpointsleases",
						ResourceNamespace: "",
						ResourceName:      "",
					},
					LockObjectName:      "kube-scheduler",
					LockObjectNamespace: "kube-system",
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
				Plugins:                  nil,
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			SetDefaults_KubeSchedulerConfiguration(tc.config)
			if !reflect.DeepEqual(tc.expected, tc.config) {
				t.Errorf("Expected:\n%#v\n\nGot:\n%#v", tc.expected, tc.config)
			}
		})
	}
}
