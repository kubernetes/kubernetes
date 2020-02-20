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

	"k8s.io/apimachinery/pkg/runtime"
	componentbaseconfigv1alpha1 "k8s.io/component-base/config/v1alpha1"
	kubeschedulerconfigv1alpha1 "k8s.io/kube-scheduler/config/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"

	// this package shouldn't really depend on other k8s.io/kubernetes code
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/master/ports"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

// SetDefaults_KubeSchedulerConfiguration sets additional defaults
func SetDefaults_KubeSchedulerConfiguration(obj *kubeschedulerconfigv1alpha1.KubeSchedulerConfiguration) {
	if obj.SchedulerName == nil {
		val := api.DefaultSchedulerName
		obj.SchedulerName = &val
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

	// For Healthz and Metrics bind addresses, we want to check:
	// 1. If the value is nil, default to 0.0.0.0 and default scheduler port
	// 2. If there is a value set, attempt to split it. If it's just a port (ie, ":1234"), default to 0.0.0.0 with that port
	// 3. If splitting the value fails, check if the value is even a valid IP. If so, use that with the default port.
	// Otherwise use the default bind address
	defaultBindAddress := net.JoinHostPort("0.0.0.0", strconv.Itoa(ports.InsecureSchedulerPort))
	if obj.HealthzBindAddress == nil {
		obj.HealthzBindAddress = &defaultBindAddress
	} else {
		if host, port, err := net.SplitHostPort(*obj.HealthzBindAddress); err == nil {
			if len(host) == 0 {
				host = "0.0.0.0"
			}
			hostPort := net.JoinHostPort(host, port)
			obj.HealthzBindAddress = &hostPort
		} else {
			// Something went wrong splitting the host/port, could just be a missing port so check if the
			// existing value is a valid IP address. If so, use that with the default scheduler port
			if host := net.ParseIP(*obj.HealthzBindAddress); host != nil {
				hostPort := net.JoinHostPort(*obj.HealthzBindAddress, strconv.Itoa(ports.InsecureSchedulerPort))
				obj.HealthzBindAddress = &hostPort
			} else {
				// TODO: in v1beta1 we should let this error instead of stomping with a default value
				obj.HealthzBindAddress = &defaultBindAddress
			}
		}
	}

	if obj.MetricsBindAddress == nil {
		obj.MetricsBindAddress = &defaultBindAddress
	} else {
		if host, port, err := net.SplitHostPort(*obj.MetricsBindAddress); err == nil {
			if len(host) == 0 {
				host = "0.0.0.0"
			}
			hostPort := net.JoinHostPort(host, port)
			obj.MetricsBindAddress = &hostPort
		} else {
			// Something went wrong splitting the host/port, could just be a missing port so check if the
			// existing value is a valid IP address. If so, use that with the default scheduler port
			if host := net.ParseIP(*obj.MetricsBindAddress); host != nil {
				hostPort := net.JoinHostPort(*obj.MetricsBindAddress, strconv.Itoa(ports.InsecureSchedulerPort))
				obj.MetricsBindAddress = &hostPort
			} else {
				// TODO: in v1beta1 we should let this error instead of stomping with a default value
				obj.MetricsBindAddress = &defaultBindAddress
			}
		}
	}

	if obj.DisablePreemption == nil {
		disablePreemption := false
		obj.DisablePreemption = &disablePreemption
	}

	if obj.PercentageOfNodesToScore == nil {
		percentageOfNodesToScore := int32(config.DefaultPercentageOfNodesToScore)
		obj.PercentageOfNodesToScore = &percentageOfNodesToScore
	}

	if len(obj.LeaderElection.ResourceLock) == 0 {
		obj.LeaderElection.ResourceLock = "endpointsleases"
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
		val := int64(600)
		obj.BindTimeoutSeconds = &val
	}

	if obj.PodInitialBackoffSeconds == nil {
		val := int64(1)
		obj.PodInitialBackoffSeconds = &val
	}

	if obj.PodMaxBackoffSeconds == nil {
		val := int64(10)
		obj.PodMaxBackoffSeconds = &val
	}

	// Enable profiling by default in the scheduler
	if obj.EnableProfiling == nil {
		enableProfiling := true
		obj.EnableProfiling = &enableProfiling
	}

	// Enable contention profiling by default if profiling is enabled
	if *obj.EnableProfiling && obj.EnableContentionProfiling == nil {
		enableContentionProfiling := true
		obj.EnableContentionProfiling = &enableContentionProfiling
	}
}
