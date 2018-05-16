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

func SetDefaults_CloudControllerManagerConfiguration(obj *CloudControllerManagerConfiguration) {
	zero := metav1.Duration{}
	if obj.ServiceController.ConcurrentServiceSyncs == 0 {
		obj.ServiceController.ConcurrentServiceSyncs = 1
	}
	if obj.NodeStatusUpdateFrequency == zero {
		obj.NodeStatusUpdateFrequency = metav1.Duration{Duration: 5 * time.Minute}
	}
}

func SetDefaults_KubeControllerManagerConfiguration(obj *KubeControllerManagerConfiguration) {
	zero := metav1.Duration{}
	if len(obj.Controllers) == 0 {
		obj.Controllers = []string{"*"}
	}
	if obj.EndPointController.ConcurrentEndpointSyncs == 0 {
		obj.EndPointController.ConcurrentEndpointSyncs = 5
	}
	if obj.ServiceController.ConcurrentServiceSyncs == 0 {
		obj.ServiceController.ConcurrentServiceSyncs = 1
	}
	if obj.ReplicationController.ConcurrentRCSyncs == 0 {
		obj.ReplicationController.ConcurrentRCSyncs = 5
	}
	if obj.ReplicaSetController.ConcurrentRSSyncs == 0 {
		obj.ReplicaSetController.ConcurrentRSSyncs = 5
	}
	if obj.DaemonSetController.ConcurrentDaemonSetSyncs == 0 {
		obj.DaemonSetController.ConcurrentDaemonSetSyncs = 2
	}
	if obj.JobController.ConcurrentJobSyncs == 0 {
		obj.JobController.ConcurrentJobSyncs = 5
	}
	if obj.ResourceQuotaController.ConcurrentResourceQuotaSyncs == 0 {
		obj.ResourceQuotaController.ConcurrentResourceQuotaSyncs = 5
	}
	if obj.DeploymentController.ConcurrentDeploymentSyncs == 0 {
		obj.DeploymentController.ConcurrentDeploymentSyncs = 5
	}
	if obj.NamespaceController.ConcurrentNamespaceSyncs == 0 {
		obj.NamespaceController.ConcurrentNamespaceSyncs = 10
	}
	if obj.SAController.ConcurrentSATokenSyncs == 0 {
		obj.SAController.ConcurrentSATokenSyncs = 5
	}
	if obj.ResourceQuotaController.ResourceQuotaSyncPeriod == zero {
		obj.ResourceQuotaController.ResourceQuotaSyncPeriod = metav1.Duration{Duration: 5 * time.Minute}
	}
	if obj.NamespaceController.NamespaceSyncPeriod == zero {
		obj.NamespaceController.NamespaceSyncPeriod = metav1.Duration{Duration: 5 * time.Minute}
	}
	if obj.PersistentVolumeBinderController.PVClaimBinderSyncPeriod == zero {
		obj.PersistentVolumeBinderController.PVClaimBinderSyncPeriod = metav1.Duration{Duration: 15 * time.Second}
	}
	if obj.HPAController.HorizontalPodAutoscalerSyncPeriod == zero {
		obj.HPAController.HorizontalPodAutoscalerSyncPeriod = metav1.Duration{Duration: 30 * time.Second}
	}
	if obj.HPAController.HorizontalPodAutoscalerUpscaleForbiddenWindow == zero {
		obj.HPAController.HorizontalPodAutoscalerUpscaleForbiddenWindow = metav1.Duration{Duration: 3 * time.Minute}
	}
	if obj.HPAController.HorizontalPodAutoscalerDownscaleForbiddenWindow == zero {
		obj.HPAController.HorizontalPodAutoscalerDownscaleForbiddenWindow = metav1.Duration{Duration: 5 * time.Minute}
	}
	if obj.HPAController.HorizontalPodAutoscalerTolerance == 0 {
		obj.HPAController.HorizontalPodAutoscalerTolerance = 0.1
	}
	if obj.DeploymentController.DeploymentControllerSyncPeriod == zero {
		obj.DeploymentController.DeploymentControllerSyncPeriod = metav1.Duration{Duration: 30 * time.Second}
	}
	if obj.DeprecatedController.RegisterRetryCount == 0 {
		obj.DeprecatedController.RegisterRetryCount = 10
	}
	if obj.NodeLifecycleController.PodEvictionTimeout == zero {
		obj.NodeLifecycleController.PodEvictionTimeout = metav1.Duration{Duration: 5 * time.Minute}
	}
	if obj.NodeLifecycleController.NodeMonitorGracePeriod == zero {
		obj.NodeLifecycleController.NodeMonitorGracePeriod = metav1.Duration{Duration: 40 * time.Second}
	}
	if obj.NodeLifecycleController.NodeStartupGracePeriod == zero {
		obj.NodeLifecycleController.NodeStartupGracePeriod = metav1.Duration{Duration: 60 * time.Second}
	}
	if obj.NodeIpamController.NodeCIDRMaskSize == 0 {
		obj.NodeIpamController.NodeCIDRMaskSize = 24
	}
	if obj.PodGCController.TerminatedPodGCThreshold == 0 {
		obj.PodGCController.TerminatedPodGCThreshold = 12500
	}
	if obj.GarbageCollectorController.EnableGarbageCollector == nil {
		obj.GarbageCollectorController.EnableGarbageCollector = utilpointer.BoolPtr(true)
	}
	if obj.GarbageCollectorController.ConcurrentGCSyncs == 0 {
		obj.GarbageCollectorController.ConcurrentGCSyncs = 20
	}
	if obj.CSRSigningController.ClusterSigningCertFile == "" {
		obj.CSRSigningController.ClusterSigningCertFile = "/etc/kubernetes/ca/ca.pem"
	}
	if obj.CSRSigningController.ClusterSigningKeyFile == "" {
		obj.CSRSigningController.ClusterSigningKeyFile = "/etc/kubernetes/ca/ca.key"
	}
	if obj.CSRSigningController.ClusterSigningDuration == zero {
		obj.CSRSigningController.ClusterSigningDuration = metav1.Duration{Duration: 365 * 24 * time.Hour}
	}
	if obj.AttachDetachController.ReconcilerSyncLoopPeriod == zero {
		obj.AttachDetachController.ReconcilerSyncLoopPeriod = metav1.Duration{Duration: 60 * time.Second}
	}
	if obj.NodeLifecycleController.EnableTaintManager == nil {
		obj.NodeLifecycleController.EnableTaintManager = utilpointer.BoolPtr(true)
	}
	if obj.HPAController.HorizontalPodAutoscalerUseRESTClients == nil {
		obj.HPAController.HorizontalPodAutoscalerUseRESTClients = utilpointer.BoolPtr(true)
	}
}

func SetDefaults_GenericComponentConfiguration(obj *GenericComponentConfiguration) {
	zero := metav1.Duration{}
	if obj.MinResyncPeriod == zero {
		obj.MinResyncPeriod = metav1.Duration{Duration: 12 * time.Hour}
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
}

func SetDefaults_KubeCloudSharedConfiguration(obj *KubeCloudSharedConfiguration) {
	zero := metav1.Duration{}
	// Port
	if obj.Address == "" {
		obj.Address = "0.0.0.0"
	}
	if obj.RouteReconciliationPeriod == zero {
		obj.RouteReconciliationPeriod = metav1.Duration{Duration: 10 * time.Second}
	}
	if obj.NodeMonitorPeriod == zero {
		obj.NodeMonitorPeriod = metav1.Duration{Duration: 5 * time.Second}
	}
	if obj.ClusterName == "" {
		obj.ClusterName = "kubernetes"
	}
	if obj.ConfigureCloudRoutes == nil {
		obj.ConfigureCloudRoutes = utilpointer.BoolPtr(true)
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
