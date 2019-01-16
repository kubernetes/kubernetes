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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kruntime "k8s.io/apimachinery/pkg/runtime"
	componentbaseconfigv1alpha1 "k8s.io/component-base/config/v1alpha1"
	kubectrlmgrconfigv1alpha1 "k8s.io/kube-controller-manager/config/v1alpha1"
	utilpointer "k8s.io/utils/pointer"
)

func addDefaultingFuncs(scheme *kruntime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_KubeControllerManagerConfiguration(obj *kubectrlmgrconfigv1alpha1.KubeControllerManagerConfiguration) {
	zero := metav1.Duration{}
	if obj.AttachDetachController.ReconcilerSyncLoopPeriod == zero {
		obj.AttachDetachController.ReconcilerSyncLoopPeriod = metav1.Duration{Duration: 60 * time.Second}
	}
	if obj.DeprecatedController.RegisterRetryCount == 0 {
		obj.DeprecatedController.RegisterRetryCount = 10
	}
	if obj.NodeIPAMController.NodeCIDRMaskSize == 0 {
		obj.NodeIPAMController.NodeCIDRMaskSize = 24
	}
	if obj.PersistentVolumeBinderController.PVClaimBinderSyncPeriod == zero {
		obj.PersistentVolumeBinderController.PVClaimBinderSyncPeriod = metav1.Duration{Duration: 15 * time.Second}
	}
	if obj.SAController.ConcurrentSATokenSyncs == 0 {
		obj.SAController.ConcurrentSATokenSyncs = 5
	}
	if obj.TTLAfterFinishedController.ConcurrentTTLSyncs <= 0 {
		obj.TTLAfterFinishedController.ConcurrentTTLSyncs = 5
	}

	// These defaults override the recommended defaults from the componentbaseconfigv1alpha1 package that are applied automatically
	// These client-connection defaults are specific to the kube-controller-manager
	if obj.Generic.ClientConnection.QPS == 0.0 {
		obj.Generic.ClientConnection.QPS = 20.0
	}
	if obj.Generic.ClientConnection.Burst == 0 {
		obj.Generic.ClientConnection.Burst = 30
	}

	// Use the default RecommendedDefaultGenericControllerManagerConfiguration options
	RecommendedDefaultGenericControllerManagerConfiguration(&obj.Generic)
}

func RecommendedDefaultGenericControllerManagerConfiguration(obj *kubectrlmgrconfigv1alpha1.GenericControllerManagerConfiguration) {
	zero := metav1.Duration{}
	if obj.Address == "" {
		obj.Address = "0.0.0.0"
	}
	if obj.MinResyncPeriod == zero {
		obj.MinResyncPeriod = metav1.Duration{Duration: 12 * time.Hour}
	}
	if obj.ControllerStartInterval == zero {
		obj.ControllerStartInterval = metav1.Duration{Duration: 0 * time.Second}
	}
	if len(obj.Controllers) == 0 {
		obj.Controllers = []string{"*"}
	}

	// Use the default ClientConnectionConfiguration and LeaderElectionConfiguration options
	componentbaseconfigv1alpha1.RecommendedDefaultClientConnectionConfiguration(&obj.ClientConnection)
	componentbaseconfigv1alpha1.RecommendedDefaultLeaderElectionConfiguration(&obj.LeaderElection)
}

func SetDefaults_KubeCloudSharedConfiguration(obj *kubectrlmgrconfigv1alpha1.KubeCloudSharedConfiguration) {
	zero := metav1.Duration{}
	if obj.NodeMonitorPeriod == zero {
		obj.NodeMonitorPeriod = metav1.Duration{Duration: 5 * time.Second}
	}
	if obj.ClusterName == "" {
		obj.ClusterName = "kubernetes"
	}
	if obj.ConfigureCloudRoutes == nil {
		obj.ConfigureCloudRoutes = utilpointer.BoolPtr(true)
	}
	if obj.RouteReconciliationPeriod == zero {
		obj.RouteReconciliationPeriod = metav1.Duration{Duration: 10 * time.Second}
	}
}

func SetDefaults_ServiceControllerConfiguration(obj *kubectrlmgrconfigv1alpha1.ServiceControllerConfiguration) {
	if obj.ConcurrentServiceSyncs == 0 {
		obj.ConcurrentServiceSyncs = 1
	}
}

func SetDefaults_CSRSigningControllerConfiguration(obj *kubectrlmgrconfigv1alpha1.CSRSigningControllerConfiguration) {
	zero := metav1.Duration{}
	if obj.ClusterSigningCertFile == "" {
		obj.ClusterSigningCertFile = "/etc/kubernetes/ca/ca.pem"
	}
	if obj.ClusterSigningKeyFile == "" {
		obj.ClusterSigningKeyFile = "/etc/kubernetes/ca/ca.key"
	}
	if obj.ClusterSigningDuration == zero {
		obj.ClusterSigningDuration = metav1.Duration{Duration: 365 * 24 * time.Hour}
	}
}

func SetDefaults_DeploymentControllerConfiguration(obj *kubectrlmgrconfigv1alpha1.DeploymentControllerConfiguration) {
	zero := metav1.Duration{}
	if obj.ConcurrentDeploymentSyncs == 0 {
		obj.ConcurrentDeploymentSyncs = 5
	}
	if obj.DeploymentControllerSyncPeriod == zero {
		obj.DeploymentControllerSyncPeriod = metav1.Duration{Duration: 30 * time.Second}
	}
}

func SetDefaults_DaemonSetControllerConfiguration(obj *kubectrlmgrconfigv1alpha1.DaemonSetControllerConfiguration) {
	if obj.ConcurrentDaemonSetSyncs == 0 {
		obj.ConcurrentDaemonSetSyncs = 2
	}
}

func SetDefaults_EndpointControllerConfiguration(obj *kubectrlmgrconfigv1alpha1.EndpointControllerConfiguration) {
	if obj.ConcurrentEndpointSyncs == 0 {
		obj.ConcurrentEndpointSyncs = 5
	}
}

func SetDefaults_GarbageCollectorControllerConfiguration(obj *kubectrlmgrconfigv1alpha1.GarbageCollectorControllerConfiguration) {
	if obj.EnableGarbageCollector == nil {
		obj.EnableGarbageCollector = utilpointer.BoolPtr(true)
	}
	if obj.ConcurrentGCSyncs == 0 {
		obj.ConcurrentGCSyncs = 20
	}
}

func SetDefaults_HPAControllerConfiguration(obj *kubectrlmgrconfigv1alpha1.HPAControllerConfiguration) {
	zero := metav1.Duration{}
	if obj.HorizontalPodAutoscalerUseRESTClients == nil {
		obj.HorizontalPodAutoscalerUseRESTClients = utilpointer.BoolPtr(true)
	}
	if obj.HorizontalPodAutoscalerSyncPeriod == zero {
		obj.HorizontalPodAutoscalerSyncPeriod = metav1.Duration{Duration: 15 * time.Second}
	}
	if obj.HorizontalPodAutoscalerUpscaleForbiddenWindow == zero {
		obj.HorizontalPodAutoscalerUpscaleForbiddenWindow = metav1.Duration{Duration: 3 * time.Minute}
	}
	if obj.HorizontalPodAutoscalerDownscaleStabilizationWindow == zero {
		obj.HorizontalPodAutoscalerDownscaleStabilizationWindow = metav1.Duration{Duration: 5 * time.Minute}
	}
	if obj.HorizontalPodAutoscalerCPUInitializationPeriod == zero {
		obj.HorizontalPodAutoscalerCPUInitializationPeriod = metav1.Duration{Duration: 5 * time.Minute}
	}
	if obj.HorizontalPodAutoscalerInitialReadinessDelay == zero {
		obj.HorizontalPodAutoscalerInitialReadinessDelay = metav1.Duration{Duration: 30 * time.Second}
	}
	if obj.HorizontalPodAutoscalerDownscaleForbiddenWindow == zero {
		obj.HorizontalPodAutoscalerDownscaleForbiddenWindow = metav1.Duration{Duration: 5 * time.Minute}
	}
	if obj.HorizontalPodAutoscalerTolerance == 0 {
		obj.HorizontalPodAutoscalerTolerance = 0.1
	}
}

func SetDefaults_JobControllerConfiguration(obj *kubectrlmgrconfigv1alpha1.JobControllerConfiguration) {
	if obj.ConcurrentJobSyncs == 0 {
		obj.ConcurrentJobSyncs = 5
	}
}

func SetDefaults_NamespaceControllerConfiguration(obj *kubectrlmgrconfigv1alpha1.NamespaceControllerConfiguration) {
	zero := metav1.Duration{}
	if obj.ConcurrentNamespaceSyncs == 0 {
		obj.ConcurrentNamespaceSyncs = 10
	}
	if obj.NamespaceSyncPeriod == zero {
		obj.NamespaceSyncPeriod = metav1.Duration{Duration: 5 * time.Minute}
	}
}

func SetDefaults_NodeLifecycleControllerConfiguration(obj *kubectrlmgrconfigv1alpha1.NodeLifecycleControllerConfiguration) {
	zero := metav1.Duration{}
	if obj.PodEvictionTimeout == zero {
		obj.PodEvictionTimeout = metav1.Duration{Duration: 5 * time.Minute}
	}
	if obj.NodeMonitorGracePeriod == zero {
		obj.NodeMonitorGracePeriod = metav1.Duration{Duration: 40 * time.Second}
	}
	if obj.NodeStartupGracePeriod == zero {
		obj.NodeStartupGracePeriod = metav1.Duration{Duration: 60 * time.Second}
	}
	if obj.EnableTaintManager == nil {
		obj.EnableTaintManager = utilpointer.BoolPtr(true)
	}
}

func SetDefaults_PodGCControllerConfiguration(obj *kubectrlmgrconfigv1alpha1.PodGCControllerConfiguration) {
	if obj.TerminatedPodGCThreshold == 0 {
		obj.TerminatedPodGCThreshold = 12500
	}
}

func SetDefaults_ReplicaSetControllerConfiguration(obj *kubectrlmgrconfigv1alpha1.ReplicaSetControllerConfiguration) {
	if obj.ConcurrentRSSyncs == 0 {
		obj.ConcurrentRSSyncs = 5
	}
}

func SetDefaults_ReplicationControllerConfiguration(obj *kubectrlmgrconfigv1alpha1.ReplicationControllerConfiguration) {
	if obj.ConcurrentRCSyncs == 0 {
		obj.ConcurrentRCSyncs = 5
	}
}

func SetDefaults_ResourceQuotaControllerConfiguration(obj *kubectrlmgrconfigv1alpha1.ResourceQuotaControllerConfiguration) {
	zero := metav1.Duration{}
	if obj.ConcurrentResourceQuotaSyncs == 0 {
		obj.ConcurrentResourceQuotaSyncs = 5
	}
	if obj.ResourceQuotaSyncPeriod == zero {
		obj.ResourceQuotaSyncPeriod = metav1.Duration{Duration: 5 * time.Minute}
	}
}

func SetDefaults_PersistentVolumeRecyclerConfiguration(obj *kubectrlmgrconfigv1alpha1.PersistentVolumeRecyclerConfiguration) {
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

func SetDefaults_VolumeConfiguration(obj *kubectrlmgrconfigv1alpha1.VolumeConfiguration) {
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
