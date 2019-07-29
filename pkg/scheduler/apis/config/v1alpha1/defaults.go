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
	"net"
	"strconv"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	componentbaseconfigv1alpha1 "k8s.io/component-base/config/v1alpha1"
	kubeschedulerconfigv1alpha1 "k8s.io/kube-scheduler/config/v1alpha1"

	// this package shouldn't really depend on other k8s.io/kubernetes code
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/master/ports"
)

// When the --failure-domains scheduler flag is not specified,
// DefaultFailureDomains defines the set of label keys used when TopologyKey is empty in PreferredDuringScheduling anti-affinity.
var defaultFailureDomains string = v1.LabelHostname + "," + v1.LabelZoneFailureDomain + "," + v1.LabelZoneRegion

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

// SetDefaults_KubeSchedulerConfiguration sets additional defaults
func SetDefaults_KubeSchedulerConfiguration(obj *kubeschedulerconfigv1alpha1.KubeSchedulerConfiguration) {
	if len(obj.SchedulerName) == 0 {
		obj.SchedulerName = api.DefaultSchedulerName
	}

	if obj.HardPodAffinitySymmetricWeight == 0 {
		obj.HardPodAffinitySymmetricWeight = api.DefaultHardPodAffinitySymmetricWeight
	}

	if obj.AlgorithmSource.Policy == nil &&
		(obj.AlgorithmSource.Provider == nil || len(*obj.AlgorithmSource.Provider) == 0) {
		val := kubeschedulerconfigv1alpha1.SchedulerDefaultProviderName
		obj.AlgorithmSource.Provider = &val
	}

	if policy := obj.AlgorithmSource.Policy; policy != nil {
		if policy.ConfigMap != nil && len(policy.ConfigMap.Namespace) == 0 {
			obj.AlgorithmSource.Policy.ConfigMap.Namespace = api.NamespaceSystem
		}
	}

	if host, port, err := net.SplitHostPort(obj.HealthzBindAddress); err == nil {
		if len(host) == 0 {
			host = "0.0.0.0"
		}
		obj.HealthzBindAddress = net.JoinHostPort(host, port)
	} else {
		obj.HealthzBindAddress = net.JoinHostPort("0.0.0.0", strconv.Itoa(ports.InsecureSchedulerPort))
	}

	if host, port, err := net.SplitHostPort(obj.MetricsBindAddress); err == nil {
		if len(host) == 0 {
			host = "0.0.0.0"
		}
		obj.MetricsBindAddress = net.JoinHostPort(host, port)
	} else {
		obj.MetricsBindAddress = net.JoinHostPort("0.0.0.0", strconv.Itoa(ports.InsecureSchedulerPort))
	}

	if len(obj.LeaderElection.LockObjectNamespace) == 0 && len(obj.LeaderElection.ResourceNamespace) == 0 {
		obj.LeaderElection.LockObjectNamespace = kubeschedulerconfigv1alpha1.SchedulerDefaultLockObjectNamespace
	}
	if len(obj.LeaderElection.LockObjectName) == 0 && len(obj.LeaderElection.ResourceName) == 0 {
		obj.LeaderElection.LockObjectName = kubeschedulerconfigv1alpha1.SchedulerDefaultLockObjectName
	}

	if len(obj.ClientConnection.ContentType) == 0 {
		obj.ClientConnection.ContentType = "application/vnd.kubernetes.protobuf"
	}
	// Scheduler has an opinion about QPS/Burst, setting specific defaults for itself, instead of generic settings.
	if obj.ClientConnection.QPS == 0.0 {
		obj.ClientConnection.QPS = 50.0
	}
	if obj.ClientConnection.Burst == 0 {
		obj.ClientConnection.Burst = 100
	}

	// Use the default LeaderElectionConfiguration options
	componentbaseconfigv1alpha1.RecommendedDefaultLeaderElectionConfiguration(&obj.LeaderElection.LeaderElectionConfiguration)

	if obj.BindTimeoutSeconds == nil {
		defaultBindTimeoutSeconds := int64(600)
		obj.BindTimeoutSeconds = &defaultBindTimeoutSeconds
	}
}
