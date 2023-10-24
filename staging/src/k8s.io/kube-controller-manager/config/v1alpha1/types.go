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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	cpconfigv1alpha1 "k8s.io/cloud-provider/config/v1alpha1"
	serviceconfigv1alpha1 "k8s.io/cloud-provider/controllers/service/config/v1alpha1"
	cmconfigv1alpha1 "k8s.io/controller-manager/config/v1alpha1"
)

// PersistentVolumeRecyclerConfiguration contains elements describing persistent volume plugins.
type PersistentVolumeRecyclerConfiguration struct {
	// maximumRetry is number of retries the PV recycler will execute on failure to recycle
	// PV.
	MaximumRetry int32
	// minimumTimeoutNFS is the minimum ActiveDeadlineSeconds to use for an NFS Recycler
	// pod.
	MinimumTimeoutNFS int32
	// podTemplateFilePathNFS is the file path to a pod definition used as a template for
	// NFS persistent volume recycling
	PodTemplateFilePathNFS string
	// incrementTimeoutNFS is the increment of time added per Gi to ActiveDeadlineSeconds
	// for an NFS scrubber pod.
	IncrementTimeoutNFS int32
	// podTemplateFilePathHostPath is the file path to a pod definition used as a template for
	// HostPath persistent volume recycling. This is for development and testing only and
	// will not work in a multi-node cluster.
	PodTemplateFilePathHostPath string
	// minimumTimeoutHostPath is the minimum ActiveDeadlineSeconds to use for a HostPath
	// Recycler pod.  This is for development and testing only and will not work in a multi-node
	// cluster.
	MinimumTimeoutHostPath int32
	// incrementTimeoutHostPath is the increment of time added per Gi to ActiveDeadlineSeconds
	// for a HostPath scrubber pod.  This is for development and testing only and will not work
	// in a multi-node cluster.
	IncrementTimeoutHostPath int32
}

// VolumeConfiguration contains *all* enumerated flags meant to configure all volume
// plugins. From this config, the controller-manager binary will create many instances of
// volume.VolumeConfig, each containing only the configuration needed for that plugin which
// are then passed to the appropriate plugin. The ControllerManager binary is the only part
// of the code which knows what plugins are supported and which flags correspond to each plugin.
type VolumeConfiguration struct {
	// enableHostPathProvisioning enables HostPath PV provisioning when running without a
	// cloud provider. This allows testing and development of provisioning features. HostPath
	// provisioning is not supported in any way, won't work in a multi-node cluster, and
	// should not be used for anything other than testing or development.
	EnableHostPathProvisioning *bool
	// enableDynamicProvisioning enables the provisioning of volumes when running within an environment
	// that supports dynamic provisioning. Defaults to true.
	EnableDynamicProvisioning *bool
	// persistentVolumeRecyclerConfiguration holds configuration for persistent volume plugins.
	PersistentVolumeRecyclerConfiguration PersistentVolumeRecyclerConfiguration
	// volumePluginDir is the full path of the directory in which the flex
	// volume plugin should search for additional third party volume plugins
	FlexVolumePluginDir string
}

// GroupResource describes an group resource.
type GroupResource struct {
	// group is the group portion of the GroupResource.
	Group string
	// resource is the resource portion of the GroupResource.
	Resource string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// KubeControllerManagerConfiguration contains elements describing kube-controller manager.
type KubeControllerManagerConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// Generic holds configuration for a generic controller-manager
	Generic cmconfigv1alpha1.GenericControllerManagerConfiguration
	// KubeCloudSharedConfiguration holds configuration for shared related features
	// both in cloud controller manager and kube-controller manager.
	KubeCloudShared cpconfigv1alpha1.KubeCloudSharedConfiguration

	// AttachDetachControllerConfiguration holds configuration for
	// AttachDetachController related features.
	AttachDetachController AttachDetachControllerConfiguration
	// CSRSigningControllerConfiguration holds configuration for
	// CSRSigningController related features.
	CSRSigningController CSRSigningControllerConfiguration
	// DaemonSetControllerConfiguration holds configuration for DaemonSetController
	// related features.
	DaemonSetController DaemonSetControllerConfiguration
	// DeploymentControllerConfiguration holds configuration for
	// DeploymentController related features.
	DeploymentController DeploymentControllerConfiguration
	// StatefulSetControllerConfiguration holds configuration for
	// StatefulSetController related features.
	StatefulSetController StatefulSetControllerConfiguration
	// DeprecatedControllerConfiguration holds configuration for some deprecated
	// features.
	DeprecatedController DeprecatedControllerConfiguration
	// EndpointControllerConfiguration holds configuration for EndpointController
	// related features.
	EndpointController EndpointControllerConfiguration
	// EndpointSliceControllerConfiguration holds configuration for
	// EndpointSliceController related features.
	EndpointSliceController EndpointSliceControllerConfiguration
	// EndpointSliceMirroringControllerConfiguration holds configuration for
	// EndpointSliceMirroringController related features.
	EndpointSliceMirroringController EndpointSliceMirroringControllerConfiguration
	// EphemeralVolumeControllerConfiguration holds configuration for EphemeralVolumeController
	// related features.
	EphemeralVolumeController EphemeralVolumeControllerConfiguration
	// GarbageCollectorControllerConfiguration holds configuration for
	// GarbageCollectorController related features.
	GarbageCollectorController GarbageCollectorControllerConfiguration
	// HPAControllerConfiguration holds configuration for HPAController related features.
	HPAController HPAControllerConfiguration
	// JobControllerConfiguration holds configuration for JobController related features.
	JobController JobControllerConfiguration
	// CronJobControllerConfiguration holds configuration for CronJobController related features.
	CronJobController CronJobControllerConfiguration
	// LegacySATokenCleanerConfiguration holds configuration for LegacySATokenCleaner related features.
	LegacySATokenCleaner LegacySATokenCleanerConfiguration
	// NamespaceControllerConfiguration holds configuration for NamespaceController
	// related features.
	NamespaceController NamespaceControllerConfiguration
	// NodeIPAMControllerConfiguration holds configuration for NodeIPAMController
	// related features.
	NodeIPAMController NodeIPAMControllerConfiguration
	// NodeLifecycleControllerConfiguration holds configuration for
	// NodeLifecycleController related features.
	NodeLifecycleController NodeLifecycleControllerConfiguration
	// PersistentVolumeBinderControllerConfiguration holds configuration for
	// PersistentVolumeBinderController related features.
	PersistentVolumeBinderController PersistentVolumeBinderControllerConfiguration
	// PodGCControllerConfiguration holds configuration for PodGCController
	// related features.
	PodGCController PodGCControllerConfiguration
	// ReplicaSetControllerConfiguration holds configuration for ReplicaSet related features.
	ReplicaSetController ReplicaSetControllerConfiguration
	// ReplicationControllerConfiguration holds configuration for
	// ReplicationController related features.
	ReplicationController ReplicationControllerConfiguration
	// ResourceQuotaControllerConfiguration holds configuration for
	// ResourceQuotaController related features.
	ResourceQuotaController ResourceQuotaControllerConfiguration
	// SAControllerConfiguration holds configuration for ServiceAccountController
	// related features.
	SAController SAControllerConfiguration
	// ServiceControllerConfiguration holds configuration for ServiceController
	// related features.
	ServiceController serviceconfigv1alpha1.ServiceControllerConfiguration
	// TTLAfterFinishedControllerConfiguration holds configuration for
	// TTLAfterFinishedController related features.
	TTLAfterFinishedController TTLAfterFinishedControllerConfiguration
	// ValidatingAdmissionPolicyStatusControllerConfiguration holds configuration for
	// ValidatingAdmissionPolicyStatusController related features.
	ValidatingAdmissionPolicyStatusController ValidatingAdmissionPolicyStatusControllerConfiguration
}

// AttachDetachControllerConfiguration contains elements describing AttachDetachController.
type AttachDetachControllerConfiguration struct {
	// Reconciler runs a periodic loop to reconcile the desired state of the with
	// the actual state of the world by triggering attach detach operations.
	// This flag enables or disables reconcile.  Is false by default, and thus enabled.
	DisableAttachDetachReconcilerSync bool
	// ReconcilerSyncLoopPeriod is the amount of time the reconciler sync states loop
	// wait between successive executions. Is set to 60 sec by default.
	ReconcilerSyncLoopPeriod metav1.Duration
	// DisableForceDetachOnTimeout disables force detach when the maximum unmount
	// time is exceeded. Is false by default, and thus force detach on unmount is
	// enabled.
	DisableForceDetachOnTimeout bool `json:"disableForceDetachOnTimeout"`
}

// CSRSigningControllerConfiguration contains elements describing CSRSigningController.
type CSRSigningControllerConfiguration struct {
	// clusterSigningCertFile is the filename containing a PEM-encoded
	// X509 CA certificate used to issue cluster-scoped certificates
	ClusterSigningCertFile string
	// clusterSigningCertFile is the filename containing a PEM-encoded
	// RSA or ECDSA private key used to issue cluster-scoped certificates
	ClusterSigningKeyFile string

	// kubeletServingSignerConfiguration holds the certificate and key used to issue certificates for the kubernetes.io/kubelet-serving signer
	KubeletServingSignerConfiguration CSRSigningConfiguration
	// kubeletClientSignerConfiguration holds the certificate and key used to issue certificates for the kubernetes.io/kube-apiserver-client-kubelet
	KubeletClientSignerConfiguration CSRSigningConfiguration
	// kubeAPIServerClientSignerConfiguration holds the certificate and key used to issue certificates for the kubernetes.io/kube-apiserver-client
	KubeAPIServerClientSignerConfiguration CSRSigningConfiguration
	// legacyUnknownSignerConfiguration holds the certificate and key used to issue certificates for the kubernetes.io/legacy-unknown
	LegacyUnknownSignerConfiguration CSRSigningConfiguration

	// clusterSigningDuration is the max length of duration signed certificates will be given.
	// Individual CSRs may request shorter certs by setting spec.expirationSeconds.
	ClusterSigningDuration metav1.Duration
}

// CSRSigningConfiguration holds information about a particular CSR signer
type CSRSigningConfiguration struct {
	// certFile is the filename containing a PEM-encoded
	// X509 CA certificate used to issue certificates
	CertFile string
	// keyFile is the filename containing a PEM-encoded
	// RSA or ECDSA private key used to issue certificates
	KeyFile string
}

// DaemonSetControllerConfiguration contains elements describing DaemonSetController.
type DaemonSetControllerConfiguration struct {
	// concurrentDaemonSetSyncs is the number of daemonset objects that are
	// allowed to sync concurrently. Larger number = more responsive daemonset,
	// but more CPU (and network) load.
	ConcurrentDaemonSetSyncs int32
}

// DeploymentControllerConfiguration contains elements describing DeploymentController.
type DeploymentControllerConfiguration struct {
	// concurrentDeploymentSyncs is the number of deployment objects that are
	// allowed to sync concurrently. Larger number = more responsive deployments,
	// but more CPU (and network) load.
	ConcurrentDeploymentSyncs int32
}

// StatefulSetControllerConfiguration contains elements describing StatefulSetController.
type StatefulSetControllerConfiguration struct {
	// concurrentStatefulSetSyncs is the number of statefulset objects that are
	// allowed to sync concurrently. Larger number = more responsive statefulsets,
	// but more CPU (and network) load.
	ConcurrentStatefulSetSyncs int32
}

// DeprecatedControllerConfiguration contains elements be deprecated.
type DeprecatedControllerConfiguration struct {
}

// EndpointControllerConfiguration contains elements describing EndpointController.
type EndpointControllerConfiguration struct {
	// concurrentEndpointSyncs is the number of endpoint syncing operations
	// that will be done concurrently. Larger number = faster endpoint updating,
	// but more CPU (and network) load.
	ConcurrentEndpointSyncs int32

	// EndpointUpdatesBatchPeriod describes the length of endpoint updates batching period.
	// Processing of pod changes will be delayed by this duration to join them with potential
	// upcoming updates and reduce the overall number of endpoints updates.
	EndpointUpdatesBatchPeriod metav1.Duration
}

// EndpointSliceControllerConfiguration contains elements describing
// EndpointSliceController.
type EndpointSliceControllerConfiguration struct {
	// concurrentServiceEndpointSyncs is the number of service endpoint syncing
	// operations that will be done concurrently. Larger number = faster
	// endpoint slice updating, but more CPU (and network) load.
	ConcurrentServiceEndpointSyncs int32

	// maxEndpointsPerSlice is the maximum number of endpoints that will be
	// added to an EndpointSlice. More endpoints per slice will result in fewer
	// and larger endpoint slices, but larger resources.
	MaxEndpointsPerSlice int32

	// EndpointUpdatesBatchPeriod describes the length of endpoint updates batching period.
	// Processing of pod changes will be delayed by this duration to join them with potential
	// upcoming updates and reduce the overall number of endpoints updates.
	EndpointUpdatesBatchPeriod metav1.Duration
}

// EndpointSliceMirroringControllerConfiguration contains elements describing
// EndpointSliceMirroringController.
type EndpointSliceMirroringControllerConfiguration struct {
	// mirroringConcurrentServiceEndpointSyncs is the number of service endpoint
	// syncing operations that will be done concurrently. Larger number = faster
	// endpoint slice updating, but more CPU (and network) load.
	MirroringConcurrentServiceEndpointSyncs int32

	// mirroringMaxEndpointsPerSubset is the maximum number of endpoints that
	// will be mirrored to an EndpointSlice for an EndpointSubset.
	MirroringMaxEndpointsPerSubset int32

	// mirroringEndpointUpdatesBatchPeriod can be used to batch EndpointSlice
	// updates. All updates triggered by EndpointSlice changes will be delayed
	// by up to 'mirroringEndpointUpdatesBatchPeriod'. If other addresses in the
	// same Endpoints resource change in that period, they will be batched to a
	// single EndpointSlice update. Default 0 value means that each Endpoints
	// update triggers an EndpointSlice update.
	MirroringEndpointUpdatesBatchPeriod metav1.Duration
}

// EphemeralVolumeControllerConfiguration contains elements describing EphemeralVolumeController.
type EphemeralVolumeControllerConfiguration struct {
	// ConcurrentEphemeralVolumeSyncseSyncs is the number of ephemeral volume syncing operations
	// that will be done concurrently. Larger number = faster ephemeral volume updating,
	// but more CPU (and network) load.
	ConcurrentEphemeralVolumeSyncs int32
}

// GarbageCollectorControllerConfiguration contains elements describing GarbageCollectorController.
type GarbageCollectorControllerConfiguration struct {
	// enables the generic garbage collector. MUST be synced with the
	// corresponding flag of the kube-apiserver. WARNING: the generic garbage
	// collector is an alpha feature.
	EnableGarbageCollector *bool
	// concurrentGCSyncs is the number of garbage collector workers that are
	// allowed to sync concurrently.
	ConcurrentGCSyncs int32
	// gcIgnoredResources is the list of GroupResources that garbage collection should ignore.
	GCIgnoredResources []GroupResource
}

// HPAControllerConfiguration contains elements describing HPAController.
type HPAControllerConfiguration struct {
	// ConcurrentHorizontalPodAutoscalerSyncs is the number of HPA objects that are allowed to sync concurrently.
	// Larger number = more responsive HPA processing, but more CPU (and network) load.
	ConcurrentHorizontalPodAutoscalerSyncs int32
	// HorizontalPodAutoscalerSyncPeriod is the period for syncing the number of
	// pods in horizontal pod autoscaler.
	HorizontalPodAutoscalerSyncPeriod metav1.Duration
	// HorizontalPodAutoscalerDowncaleStabilizationWindow is a period for which autoscaler will look
	// backwards and not scale down below any recommendation it made during that period.
	HorizontalPodAutoscalerDownscaleStabilizationWindow metav1.Duration
	// HorizontalPodAutoscalerTolerance is the tolerance for when
	// resource usage suggests upscaling/downscaling
	HorizontalPodAutoscalerTolerance float64
	// HorizontalPodAutoscalerCPUInitializationPeriod is the period after pod start when CPU samples
	// might be skipped.
	HorizontalPodAutoscalerCPUInitializationPeriod metav1.Duration
	// HorizontalPodAutoscalerInitialReadinessDelay is period after pod start during which readiness
	// changes are treated as readiness being set for the first time. The only effect of this is that
	// HPA will disregard CPU samples from unready pods that had last readiness change during that
	// period.
	HorizontalPodAutoscalerInitialReadinessDelay metav1.Duration
}

// JobControllerConfiguration contains elements describing JobController.
type JobControllerConfiguration struct {
	// concurrentJobSyncs is the number of job objects that are
	// allowed to sync concurrently. Larger number = more responsive jobs,
	// but more CPU (and network) load.
	ConcurrentJobSyncs int32
}

// CronJobControllerConfiguration contains elements describing CrongJob2Controller.
type CronJobControllerConfiguration struct {
	// concurrentCronJobSyncs is the number of job objects that are
	// allowed to sync concurrently. Larger number = more responsive jobs,
	// but more CPU (and network) load.
	ConcurrentCronJobSyncs int32
}

// LegacySATokenCleanerConfiguration contains elements describing LegacySATokenCleaner
type LegacySATokenCleanerConfiguration struct {
	// CleanUpPeriod is the period of time since the last usage of an
	// auto-generated service account token before it can be deleted.
	CleanUpPeriod metav1.Duration
}

// NamespaceControllerConfiguration contains elements describing NamespaceController.
type NamespaceControllerConfiguration struct {
	// namespaceSyncPeriod is the period for syncing namespace life-cycle
	// updates.
	NamespaceSyncPeriod metav1.Duration
	// concurrentNamespaceSyncs is the number of namespace objects that are
	// allowed to sync concurrently.
	ConcurrentNamespaceSyncs int32
}

// NodeIPAMControllerConfiguration contains elements describing NodeIpamController.
type NodeIPAMControllerConfiguration struct {
	// serviceCIDR is CIDR Range for Services in cluster.
	ServiceCIDR string
	// secondaryServiceCIDR is CIDR Range for Services in cluster. This is used in dual stack clusters. SecondaryServiceCIDR must be of different IP family than ServiceCIDR
	SecondaryServiceCIDR string
	// NodeCIDRMaskSize is the mask size for node cidr in cluster.
	NodeCIDRMaskSize int32
	// NodeCIDRMaskSizeIPv4 is the mask size for node cidr in dual-stack cluster.
	NodeCIDRMaskSizeIPv4 int32
	// NodeCIDRMaskSizeIPv6 is the mask size for node cidr in dual-stack cluster.
	NodeCIDRMaskSizeIPv6 int32
}

// NodeLifecycleControllerConfiguration contains elements describing NodeLifecycleController.
type NodeLifecycleControllerConfiguration struct {
	// nodeEvictionRate is the number of nodes per second on which pods are deleted in case of node failure when a zone is healthy
	NodeEvictionRate float32
	// secondaryNodeEvictionRate is the number of nodes per second on which pods are deleted in case of node failure when a zone is unhealthy
	SecondaryNodeEvictionRate float32
	// nodeStartupGracePeriod is the amount of time which we allow starting a node to
	// be unresponsive before marking it unhealthy.
	NodeStartupGracePeriod metav1.Duration
	// nodeMontiorGracePeriod is the amount of time which we allow a running node to be
	// unresponsive before marking it unhealthy. Must be N times more than kubelet's
	// nodeStatusUpdateFrequency, where N means number of retries allowed for kubelet
	// to post node status.
	NodeMonitorGracePeriod metav1.Duration
	// podEvictionTimeout is the grace period for deleting pods on failed nodes.
	PodEvictionTimeout metav1.Duration
	// secondaryNodeEvictionRate is implicitly overridden to 0 for clusters smaller than or equal to largeClusterSizeThreshold
	LargeClusterSizeThreshold int32
	// Zone is treated as unhealthy in nodeEvictionRate and secondaryNodeEvictionRate when at least
	// unhealthyZoneThreshold (no less than 3) of Nodes in the zone are NotReady
	UnhealthyZoneThreshold float32
}

// PersistentVolumeBinderControllerConfiguration contains elements describing
// PersistentVolumeBinderController.
type PersistentVolumeBinderControllerConfiguration struct {
	// pvClaimBinderSyncPeriod is the period for syncing persistent volumes
	// and persistent volume claims.
	PVClaimBinderSyncPeriod metav1.Duration
	// volumeConfiguration holds configuration for volume related features.
	VolumeConfiguration VolumeConfiguration
}

// PodGCControllerConfiguration contains elements describing PodGCController.
type PodGCControllerConfiguration struct {
	// terminatedPodGCThreshold is the number of terminated pods that can exist
	// before the terminated pod garbage collector starts deleting terminated pods.
	// If <= 0, the terminated pod garbage collector is disabled.
	TerminatedPodGCThreshold int32
}

// ReplicaSetControllerConfiguration contains elements describing ReplicaSetController.
type ReplicaSetControllerConfiguration struct {
	// concurrentRSSyncs is the number of replica sets that are  allowed to sync
	// concurrently. Larger number = more responsive replica  management, but more
	// CPU (and network) load.
	ConcurrentRSSyncs int32
}

// ReplicationControllerConfiguration contains elements describing ReplicationController.
type ReplicationControllerConfiguration struct {
	// concurrentRCSyncs is the number of replication controllers that are
	// allowed to sync concurrently. Larger number = more responsive replica
	// management, but more CPU (and network) load.
	ConcurrentRCSyncs int32
}

// ResourceQuotaControllerConfiguration contains elements describing ResourceQuotaController.
type ResourceQuotaControllerConfiguration struct {
	// resourceQuotaSyncPeriod is the period for syncing quota usage status
	// in the system.
	ResourceQuotaSyncPeriod metav1.Duration
	// concurrentResourceQuotaSyncs is the number of resource quotas that are
	// allowed to sync concurrently. Larger number = more responsive quota
	// management, but more CPU (and network) load.
	ConcurrentResourceQuotaSyncs int32
}

// SAControllerConfiguration contains elements describing ServiceAccountController.
type SAControllerConfiguration struct {
	// serviceAccountKeyFile is the filename containing a PEM-encoded private RSA key
	// used to sign service account tokens.
	ServiceAccountKeyFile string
	// concurrentSATokenSyncs is the number of service account token syncing operations
	// that will be done concurrently.
	ConcurrentSATokenSyncs int32
	// rootCAFile is the root certificate authority will be included in service
	// account's token secret. This must be a valid PEM-encoded CA bundle.
	RootCAFile string
}

// TTLAfterFinishedControllerConfiguration contains elements describing TTLAfterFinishedController.
type TTLAfterFinishedControllerConfiguration struct {
	// concurrentTTLSyncs is the number of TTL-after-finished collector workers that are
	// allowed to sync concurrently.
	ConcurrentTTLSyncs int32
}

// ValidatingAdmissionPolicyStatusControllerConfiguration contains elements describing ValidatingAdmissionPolicyStatusController.
type ValidatingAdmissionPolicyStatusControllerConfiguration struct {
	// ConcurrentPolicySyncs is the number of policy objects that are
	// allowed to sync concurrently. Larger number = quicker type checking,
	// but more CPU (and network) load.
	// The default value is 5.
	ConcurrentPolicySyncs int32
}

// LeaderElectionControllerConfiguration contains elements describing LeaderElectionController.
type LeaderElectionControllerConfiguration struct {
	// LeaderElectionConcurrentLeaseSyncs is the number of lease objects that are
	// allowed to sync concurrently. Larger number = quicker type checking,
	// but more CPU (and network) load.
	// The default value is 5.
	LeaderElectionConcurrentLeaseSyncs int32
}
