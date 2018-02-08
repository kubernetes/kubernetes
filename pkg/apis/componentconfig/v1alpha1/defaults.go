/*
Copyright 2015 The Kubernetes Authors.

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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kruntime "k8s.io/apimachinery/pkg/runtime"
	api "k8s.io/kubernetes/pkg/apis/core"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/pkg/master/ports"
)

func addDefaultingFuncs(scheme *kruntime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_KubeSchedulerConfiguration(obj *KubeSchedulerConfiguration) {
	if len(obj.SchedulerName) == 0 {
		obj.SchedulerName = api.DefaultSchedulerName
	}

	if obj.HardPodAffinitySymmetricWeight == 0 {
		obj.HardPodAffinitySymmetricWeight = api.DefaultHardPodAffinitySymmetricWeight
	}

	if obj.AlgorithmSource.Policy == nil &&
		(obj.AlgorithmSource.Provider == nil || len(*obj.AlgorithmSource.Provider) == 0) {
		val := SchedulerDefaultProviderName
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
		obj.HealthzBindAddress = net.JoinHostPort("0.0.0.0", strconv.Itoa(ports.SchedulerPort))
	}

	if host, port, err := net.SplitHostPort(obj.MetricsBindAddress); err == nil {
		if len(host) == 0 {
			host = "0.0.0.0"
		}
		obj.MetricsBindAddress = net.JoinHostPort(host, port)
	} else {
		obj.MetricsBindAddress = net.JoinHostPort("0.0.0.0", strconv.Itoa(ports.SchedulerPort))
	}

	if len(obj.ClientConnection.ContentType) == 0 {
		obj.ClientConnection.ContentType = "application/vnd.kubernetes.protobuf"
	}
	if obj.ClientConnection.QPS == 0.0 {
		obj.ClientConnection.QPS = 50.0
	}
	if obj.ClientConnection.Burst == 0 {
		obj.ClientConnection.Burst = 100
	}

	if len(obj.LeaderElection.LockObjectNamespace) == 0 {
		obj.LeaderElection.LockObjectNamespace = SchedulerDefaultLockObjectNamespace
	}
	if len(obj.LeaderElection.LockObjectName) == 0 {
		obj.LeaderElection.LockObjectName = SchedulerDefaultLockObjectName
	}

	if len(obj.FailureDomains) == 0 {
		obj.FailureDomains = kubeletapis.DefaultFailureDomains
	}
}

func SetDefaults_LeaderElectionConfiguration(obj *LeaderElectionConfiguration) {
	zero := metav1.Duration{}
	if obj.LeaseDuration == zero {
		obj.LeaseDuration = metav1.Duration{Duration: 15 * time.Second}
	}
	if obj.RenewDeadline == zero {
		obj.RenewDeadline = metav1.Duration{Duration: 10 * time.Second}
	}
	if obj.RetryPeriod == zero {
		obj.RetryPeriod = metav1.Duration{Duration: 2 * time.Second}
	}
	if obj.ResourceLock == "" {
		// obj.ResourceLock = rl.EndpointsResourceLock
		obj.ResourceLock = "endpoints"
	}
}
