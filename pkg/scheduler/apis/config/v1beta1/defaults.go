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
	"net"
	"strconv"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/util/feature"
	componentbaseconfigv1alpha1 "k8s.io/component-base/config/v1alpha1"
	"k8s.io/kube-scheduler/config/v1beta1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/utils/pointer"

	// this package shouldn't really depend on other k8s.io/kubernetes code
	api "k8s.io/kubernetes/pkg/apis/core"
)

var defaultResourceSpec = []v1beta1.ResourceSpec{
	{Name: string(corev1.ResourceCPU), Weight: 1},
	{Name: string(corev1.ResourceMemory), Weight: 1},
}

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

// SetDefaults_KubeSchedulerConfiguration sets additional defaults
func SetDefaults_KubeSchedulerConfiguration(obj *v1beta1.KubeSchedulerConfiguration) {
	if len(obj.Profiles) == 0 {
		obj.Profiles = append(obj.Profiles, v1beta1.KubeSchedulerProfile{})
	}
	// Only apply a default scheduler name when there is a single profile.
	// Validation will ensure that every profile has a non-empty unique name.
	if len(obj.Profiles) == 1 && obj.Profiles[0].SchedulerName == nil {
		obj.Profiles[0].SchedulerName = pointer.StringPtr(api.DefaultSchedulerName)
	}

	// For Healthz and Metrics bind addresses, we want to check:
	// 1. If the value is nil, default to 0.0.0.0 and default scheduler port
	// 2. If there is a value set, attempt to split it. If it's just a port (ie, ":1234"), default to 0.0.0.0 with that port
	// 3. If splitting the value fails, check if the value is even a valid IP. If so, use that with the default port.
	// Otherwise use the default bind address
	defaultBindAddress := net.JoinHostPort("0.0.0.0", strconv.Itoa(config.DefaultInsecureSchedulerPort))
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
				hostPort := net.JoinHostPort(*obj.HealthzBindAddress, strconv.Itoa(config.DefaultInsecureSchedulerPort))
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
				hostPort := net.JoinHostPort(*obj.MetricsBindAddress, strconv.Itoa(config.DefaultInsecureSchedulerPort))
				obj.MetricsBindAddress = &hostPort
			} else {
				// TODO: in v1beta1 we should let this error instead of stomping with a default value
				obj.MetricsBindAddress = &defaultBindAddress
			}
		}
	}

	if obj.PercentageOfNodesToScore == nil {
		percentageOfNodesToScore := int32(config.DefaultPercentageOfNodesToScore)
		obj.PercentageOfNodesToScore = &percentageOfNodesToScore
	}

	if len(obj.LeaderElection.ResourceLock) == 0 {
		obj.LeaderElection.ResourceLock = "endpointsleases"
	}
	if len(obj.LeaderElection.ResourceNamespace) == 0 {
		obj.LeaderElection.ResourceNamespace = v1beta1.SchedulerDefaultLockObjectNamespace
	}
	if len(obj.LeaderElection.ResourceName) == 0 {
		obj.LeaderElection.ResourceName = v1beta1.SchedulerDefaultLockObjectName
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
	componentbaseconfigv1alpha1.RecommendedDefaultLeaderElectionConfiguration(&obj.LeaderElection)

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

func SetDefaults_InterPodAffinityArgs(obj *v1beta1.InterPodAffinityArgs) {
	// Note that an object is created manually in cmd/kube-scheduler/app/options/deprecated.go
	// DeprecatedOptions#ApplyTo.
	// Update that object if a new default field is added here.
	if obj.HardPodAffinityWeight == nil {
		obj.HardPodAffinityWeight = pointer.Int32Ptr(1)
	}
}

func SetDefaults_NodeResourcesLeastAllocatedArgs(obj *v1beta1.NodeResourcesLeastAllocatedArgs) {
	if len(obj.Resources) == 0 {
		// If no resources specified, used the default set.
		obj.Resources = append(obj.Resources, defaultResourceSpec...)
	}
}

func SetDefaults_NodeResourcesMostAllocatedArgs(obj *v1beta1.NodeResourcesMostAllocatedArgs) {
	if len(obj.Resources) == 0 {
		// If no resources specified, used the default set.
		obj.Resources = append(obj.Resources, defaultResourceSpec...)
	}
}

func SetDefaults_RequestedToCapacityRatioArgs(obj *v1beta1.RequestedToCapacityRatioArgs) {
	if len(obj.Resources) == 0 {
		// If no resources specified, used the default set.
		obj.Resources = append(obj.Resources, defaultResourceSpec...)
	}
}

func SetDefaults_VolumeBindingArgs(obj *v1beta1.VolumeBindingArgs) {
	if obj.BindTimeoutSeconds == nil {
		obj.BindTimeoutSeconds = pointer.Int64Ptr(600)
	}
}

func SetDefaults_PodTopologySpreadArgs(obj *v1beta1.PodTopologySpreadArgs) {
	if !feature.DefaultFeatureGate.Enabled(features.DefaultPodTopologySpread) {
		// When feature is disabled, the default spreading is done by legacy
		// SelectorSpread plugin.
		return
	}
	if obj.DefaultConstraints == nil {
		obj.DefaultConstraints = []corev1.TopologySpreadConstraint{
			{
				TopologyKey:       corev1.LabelHostname,
				WhenUnsatisfiable: corev1.ScheduleAnyway,
				MaxSkew:           3,
			},
			{
				TopologyKey:       corev1.LabelZoneFailureDomainStable,
				WhenUnsatisfiable: corev1.ScheduleAnyway,
				MaxSkew:           5,
			},
		}
	}
}
