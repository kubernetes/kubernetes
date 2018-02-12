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
	utilpointer "k8s.io/kubernetes/pkg/util/pointer"
)

func addDefaultingFuncs(scheme *kruntime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_KubeControllerManagerConfiguration(obj *KubeControllerManagerConfiguration) {
	zero := metav1.Duration{}
	if len(obj.Controllers) == 0 {
		obj.Controllers = []string{"*"}
	}
	// Port
	if obj.Address == "" {
		obj.Address = "0.0.0.0"
	}
	if obj.ConcurrentEndpointSyncs == 0 {
		obj.ConcurrentEndpointSyncs = 5
	}
	if obj.ConcurrentServiceSyncs == 0 {
		obj.ConcurrentServiceSyncs = 1
	}
	if obj.ConcurrentRCSyncs == 0 {
		obj.ConcurrentRCSyncs = 5
	}
	if obj.ConcurrentRSSyncs == 0 {
		obj.ConcurrentRSSyncs = 5
	}
	if obj.ConcurrentDaemonSetSyncs == 0 {
		obj.ConcurrentDaemonSetSyncs = 2
	}
	if obj.ConcurrentJobSyncs == 0 {
		obj.ConcurrentJobSyncs = 5
	}
	if obj.ConcurrentResourceQuotaSyncs == 0 {
		obj.ConcurrentResourceQuotaSyncs = 5
	}
	if obj.ConcurrentDeploymentSyncs == 0 {
		obj.ConcurrentDeploymentSyncs = 5
	}
	if obj.ConcurrentNamespaceSyncs == 0 {
		obj.ConcurrentNamespaceSyncs = 10
	}
	if obj.ConcurrentSATokenSyncs == 0 {
		obj.ConcurrentSATokenSyncs = 5
	}
	if obj.RouteReconciliationPeriod == zero {
		obj.RouteReconciliationPeriod = metav1.Duration{Duration: 10 * time.Second}
	}
	if obj.ResourceQuotaSyncPeriod == zero {
		obj.ResourceQuotaSyncPeriod = metav1.Duration{Duration: 5 * time.Minute}
	}
	if obj.NamespaceSyncPeriod == zero {
		obj.NamespaceSyncPeriod = metav1.Duration{Duration: 5 * time.Minute}
	}
	if obj.PVClaimBinderSyncPeriod == zero {
		obj.PVClaimBinderSyncPeriod = metav1.Duration{Duration: 15 * time.Second}
	}
	if obj.HorizontalPodAutoscalerSyncPeriod == zero {
		obj.HorizontalPodAutoscalerSyncPeriod = metav1.Duration{Duration: 30 * time.Second}
	}
	if obj.HorizontalPodAutoscalerUpscaleForbiddenWindow == zero {
		obj.HorizontalPodAutoscalerUpscaleForbiddenWindow = metav1.Duration{Duration: 3 * time.Minute}
	}
	if obj.HorizontalPodAutoscalerDownscaleForbiddenWindow == zero {
		obj.HorizontalPodAutoscalerDownscaleForbiddenWindow = metav1.Duration{Duration: 5 * time.Minute}
	}
	if obj.HorizontalPodAutoscalerTolerance == 0 {
		obj.HorizontalPodAutoscalerTolerance = 0.1
	}
	if obj.DeploymentControllerSyncPeriod == zero {
		obj.DeploymentControllerSyncPeriod = metav1.Duration{Duration: 30 * time.Second}
	}
	if obj.MinResyncPeriod == zero {
		obj.MinResyncPeriod = metav1.Duration{Duration: 12 * time.Hour}
	}
	if obj.RegisterRetryCount == 0 {
		obj.RegisterRetryCount = 10
	}
	if obj.PodEvictionTimeout == zero {
		obj.PodEvictionTimeout = metav1.Duration{Duration: 5 * time.Minute}
	}
	if obj.NodeMonitorGracePeriod == zero {
		obj.NodeMonitorGracePeriod = metav1.Duration{Duration: 40 * time.Second}
	}
	if obj.NodeStartupGracePeriod == zero {
		obj.NodeStartupGracePeriod = metav1.Duration{Duration: 60 * time.Second}
	}
	if obj.NodeMonitorPeriod == zero {
		obj.NodeMonitorPeriod = metav1.Duration{Duration: 5 * time.Second}
	}
	if obj.ClusterName == "" {
		obj.ClusterName = "kubernetes"
	}
	if obj.NodeCIDRMaskSize == 0 {
		obj.NodeCIDRMaskSize = 24
	}
	if obj.ConfigureCloudRoutes == nil {
		obj.ConfigureCloudRoutes = utilpointer.BoolPtr(true)
	}
	if obj.TerminatedPodGCThreshold == 0 {
		obj.TerminatedPodGCThreshold = 12500
	}
	if obj.ContentType == "" {
		obj.ContentType = "application/vnd.kubernetes.protobuf"
	}
	if obj.KubeAPIQPS == 0 {
		obj.KubeAPIQPS = 20.0
	}
	if obj.KubeAPIBurst == 0 {
		obj.KubeAPIBurst = 30
	}
	if obj.ControllerStartInterval == zero {
		obj.ControllerStartInterval = metav1.Duration{Duration: 0 * time.Second}
	}
	if obj.EnableGarbageCollector == nil {
		obj.EnableGarbageCollector = utilpointer.BoolPtr(true)
	}
	if obj.ConcurrentGCSyncs == 0 {
		obj.ConcurrentGCSyncs = 20
	}
	if obj.ClusterSigningCertFile == "" {
		obj.ClusterSigningCertFile = "/etc/kubernetes/ca/ca.pem"
	}
	if obj.ClusterSigningKeyFile == "" {
		obj.ClusterSigningKeyFile = "/etc/kubernetes/ca/ca.key"
	}
	if obj.ClusterSigningDuration == zero {
		obj.ClusterSigningDuration = metav1.Duration{Duration: 365 * 24 * time.Hour}
	}
	if obj.ReconcilerSyncLoopPeriod == zero {
		obj.ReconcilerSyncLoopPeriod = metav1.Duration{Duration: 60 * time.Second}
	}
	if obj.EnableTaintManager == nil {
		obj.EnableTaintManager = utilpointer.BoolPtr(true)
	}
	if obj.HorizontalPodAutoscalerUseRESTClients == nil {
		obj.HorizontalPodAutoscalerUseRESTClients = utilpointer.BoolPtr(true)
	}
}

func SetDefaults_PersistentVolumeRecyclerConfiguration(obj *PersistentVolumeRecyclerConfiguration) {
	if obj.MaximumRetry == 0 {
		obj.MaximumRetry = 3
	}
	if obj.MinimumTimeoutNFS == 0 {
		obj.MinimumTimeoutNFS = 300
	}
	if obj.IncrementTimeoutNFS == 0 {
		obj.IncrementTimeoutNFS = 30
	}
	if obj.MinimumTimeoutHostPath == 0 {
		obj.MinimumTimeoutHostPath = 60
	}
	if obj.IncrementTimeoutHostPath == 0 {
		obj.IncrementTimeoutHostPath = 30
	}
}

func SetDefaults_VolumeConfiguration(obj *VolumeConfiguration) {
	if obj.EnableHostPathProvisioning == nil {
		obj.EnableHostPathProvisioning = utilpointer.BoolPtr(false)
	}
	if obj.EnableDynamicProvisioning == nil {
		obj.EnableDynamicProvisioning = utilpointer.BoolPtr(true)
	}
	if obj.FlexVolumePluginDir == "" {
		obj.FlexVolumePluginDir = "/usr/libexec/kubernetes/kubelet-plugins/volume/exec/"
	}
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
	if obj.LeaderElect == nil {
		obj.LeaderElect = utilpointer.BoolPtr(true)
	}
}
