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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ClientConnectionConfiguration contains details for constructing a client.
type ClientConnectionConfiguration struct {
	// kubeConfigFile is the path to a kubeconfig file.
	KubeConfigFile string `json:"kubeconfig"`
	// acceptContentTypes defines the Accept header sent by clients when connecting to a server, overriding the
	// default value of 'application/json'. This field will control all connections to the server used by a particular
	// client.
	AcceptContentTypes string `json:"acceptContentTypes"`
	// contentType is the content type used when sending data to the server from this client.
	ContentType string `json:"contentType"`
	// cps controls the number of queries per second allowed for this connection.
	QPS float32 `json:"qps"`
	// burst allows extra queries to accumulate when a client is exceeding its rate.
	Burst int32 `json:"burst"`
}

// SchedulerPolicySource configures a means to obtain a scheduler Policy. One
// source field must be specified, and source fields are mutually exclusive.
type SchedulerPolicySource struct {
	// File is a file policy source.
	File *SchedulerPolicyFileSource `json:"file,omitempty"`
	// ConfigMap is a config map policy source.
	ConfigMap *SchedulerPolicyConfigMapSource `json:"configMap,omitempty"`
}

// SchedulerPolicyFileSource is a policy serialized to disk and accessed via
// path.
type SchedulerPolicyFileSource struct {
	// Path is the location of a serialized policy.
	Path string `json:"path"`
}

// SchedulerPolicyConfigMapSource is a policy serialized into a config map value
// under the SchedulerPolicyConfigMapKey key.
type SchedulerPolicyConfigMapSource struct {
	// Namespace is the namespace of the policy config map.
	Namespace string `json:"namespace"`
	// Name is the name of hte policy config map.
	Name string `json:"name"`
}

// SchedulerAlgorithmSource is the source of a scheduler algorithm. One source
// field must be specified, and source fields are mutually exclusive.
type SchedulerAlgorithmSource struct {
	// Policy is a policy based algorithm source.
	Policy *SchedulerPolicySource `json:"policy,omitempty"`
	// Provider is the name of a scheduling algorithm provider to use.
	Provider *string `json:"provider,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type KubeSchedulerConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// SchedulerName is name of the scheduler, used to select which pods
	// will be processed by this scheduler, based on pod's "spec.SchedulerName".
	SchedulerName string `json:"schedulerName"`
	// AlgorithmSource specifies the scheduler algorithm source.
	AlgorithmSource SchedulerAlgorithmSource `json:"algorithmSource"`
	// RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule
	// corresponding to every RequiredDuringScheduling affinity rule.
	// HardPodAffinitySymmetricWeight represents the weight of implicit PreferredDuringScheduling affinity rule, in the range 0-100.
	HardPodAffinitySymmetricWeight int32 `json:"hardPodAffinitySymmetricWeight"`

	// LeaderElection defines the configuration of leader election client.
	LeaderElection KubeSchedulerLeaderElectionConfiguration `json:"leaderElection"`

	// ClientConnection specifies the kubeconfig file and client connection
	// settings for the proxy server to use when communicating with the apiserver.
	ClientConnection ClientConnectionConfiguration `json:"clientConnection"`
	// HealthzBindAddress is the IP address and port for the health check server to serve on,
	// defaulting to 0.0.0.0:10251
	HealthzBindAddress string `json:"healthzBindAddress"`
	// MetricsBindAddress is the IP address and port for the metrics server to
	// serve on, defaulting to 0.0.0.0:10251.
	MetricsBindAddress string `json:"metricsBindAddress"`
	// EnableProfiling enables profiling via web interface on /debug/pprof
	// handler. Profiling handlers will be handled by metrics server.
	EnableProfiling bool `json:"enableProfiling"`
	// EnableContentionProfiling enables lock contention profiling, if
	// EnableProfiling is true.
	EnableContentionProfiling bool `json:"enableContentionProfiling"`

	// Indicate the "all topologies" set for empty topologyKey when it's used for PreferredDuringScheduling pod anti-affinity.
	FailureDomains string `json:"failureDomains"`

	// DisablePreemption disables the pod preemption feature.
	DisablePreemption bool `json:"disablePreemption"`
}

// LeaderElectionConfiguration defines the configuration of leader election
// clients for components that can run with leader election enabled.
type LeaderElectionConfiguration struct {
	// leaderElect enables a leader election client to gain leadership
	// before executing the main loop. Enable this when running replicated
	// components for high availability.
	LeaderElect *bool `json:"leaderElect"`
	// leaseDuration is the duration that non-leader candidates will wait
	// after observing a leadership renewal until attempting to acquire
	// leadership of a led but unrenewed leader slot. This is effectively the
	// maximum duration that a leader can be stopped before it is replaced
	// by another candidate. This is only applicable if leader election is
	// enabled.
	LeaseDuration metav1.Duration `json:"leaseDuration"`
	// renewDeadline is the interval between attempts by the acting master to
	// renew a leadership slot before it stops leading. This must be less
	// than or equal to the lease duration. This is only applicable if leader
	// election is enabled.
	RenewDeadline metav1.Duration `json:"renewDeadline"`
	// retryPeriod is the duration the clients should wait between attempting
	// acquisition and renewal of a leadership. This is only applicable if
	// leader election is enabled.
	RetryPeriod metav1.Duration `json:"retryPeriod"`
	// resourceLock indicates the resource object type that will be used to lock
	// during leader election cycles.
	ResourceLock string `json:"resourceLock"`
}

// KubeSchedulerLeaderElectionConfiguration expands LeaderElectionConfiguration
// to include scheduler specific configuration.
type KubeSchedulerLeaderElectionConfiguration struct {
	LeaderElectionConfiguration `json:",inline"`
	// LockObjectNamespace defines the namespace of the lock object
	LockObjectNamespace string `json:"lockObjectNamespace"`
	// LockObjectName defines the lock object name
	LockObjectName string `json:"lockObjectName"`
}

type PersistentVolumeRecyclerConfiguration struct {
	// maximumRetry is number of retries the PV recycler will execute on failure to recycle
	// PV.
	MaximumRetry int32 `json:"maximumRetry"`
	// minimumTimeoutNFS is the minimum ActiveDeadlineSeconds to use for an NFS Recycler
	// pod.
	MinimumTimeoutNFS int32 `json:"minimumTimeoutNFS"`
	// podTemplateFilePathNFS is the file path to a pod definition used as a template for
	// NFS persistent volume recycling
	PodTemplateFilePathNFS string `json:"podTemplateFilePathNFS"`
	// incrementTimeoutNFS is the increment of time added per Gi to ActiveDeadlineSeconds
	// for an NFS scrubber pod.
	IncrementTimeoutNFS int32 `json:"incrementTimeoutNFS"`
	// podTemplateFilePathHostPath is the file path to a pod definition used as a template for
	// HostPath persistent volume recycling. This is for development and testing only and
	// will not work in a multi-node cluster.
	PodTemplateFilePathHostPath string `json:"podTemplateFilePathHostPath"`
	// minimumTimeoutHostPath is the minimum ActiveDeadlineSeconds to use for a HostPath
	// Recycler pod.  This is for development and testing only and will not work in a multi-node
	// cluster.
	MinimumTimeoutHostPath int32 `json:"minimumTimeoutHostPath"`
	// incrementTimeoutHostPath is the increment of time added per Gi to ActiveDeadlineSeconds
	// for a HostPath scrubber pod.  This is for development and testing only and will not work
	// in a multi-node cluster.
	IncrementTimeoutHostPath int32 `json:"incrementTimeoutHostPath"`
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
	EnableHostPathProvisioning *bool `json:"enableHostPathProvisioning"`
	// enableDynamicProvisioning enables the provisioning of volumes when running within an environment
	// that supports dynamic provisioning. Defaults to true.
	EnableDynamicProvisioning *bool `json:"enableDynamicProvisioning"`
	// persistentVolumeRecyclerConfiguration holds configuration for persistent volume plugins.
	PersistentVolumeRecyclerConfiguration PersistentVolumeRecyclerConfiguration `json:"persistentVolumeRecyclerConfiguration"`
	// volumePluginDir is the full path of the directory in which the flex
	// volume plugin should search for additional third party volume plugins
	FlexVolumePluginDir string `json:"flexVolumePluginDir"`
}

type GroupResource struct {
	// group is the group portion of the GroupResource.
	Group string `json:"group"`
	// resource is the resource portion of the GroupResource.
	Resource string `json:"resource"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type KubeControllerManagerConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// CloudProviderConfiguration holds configuration for CloudProvider related features.
	CloudProvider CloudProviderConfiguration
	// DebuggingConfiguration holds configuration for Debugging related features.
	Debugging DebuggingConfiguration
	// GenericComponentConfiguration holds configuration for GenericComponent
	// related features both in cloud controller manager and kube-controller manager.
	GenericComponent GenericComponentConfiguration
	// KubeCloudSharedConfiguration holds configuration for shared related features
	// both in cloud controller manager and kube-controller manager.
	KubeCloudShared KubeCloudSharedConfiguration

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
	// DeprecatedControllerConfiguration holds configuration for some deprecated
	// features.
	DeprecatedController DeprecatedControllerConfiguration
	// EndPointControllerConfiguration holds configuration for EndPointController
	// related features.
	EndPointController EndPointControllerConfiguration
	// GarbageCollectorControllerConfiguration holds configuration for
	// GarbageCollectorController related features.
	GarbageCollectorController GarbageCollectorControllerConfiguration
	// HPAControllerConfiguration holds configuration for HPAController related features.
	HPAController HPAControllerConfiguration
	// JobControllerConfiguration holds configuration for JobController related features.
	JobController JobControllerConfiguration
	// NamespaceControllerConfiguration holds configuration for NamespaceController
	// related features.
	NamespaceController NamespaceControllerConfiguration
	// NodeIpamControllerConfiguration holds configuration for NodeIpamController
	// related features.
	NodeIpamController NodeIpamControllerConfiguration
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
	ServiceController ServiceControllerConfiguration

	// Controllers is the list of controllers to enable or disable
	// '*' means "all enabled by default controllers"
	// 'foo' means "enable 'foo'"
	// '-foo' means "disable 'foo'"
	// first item for a particular name wins
	Controllers []string `json:"controllers"`
	// externalCloudVolumePlugin specifies the plugin to use when cloudProvider is "external".
	// It is currently used by the in repo cloud providers to handle node and volume control in the KCM.
	ExternalCloudVolumePlugin string `json:"externalCloudVolumePlugin"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type CloudControllerManagerConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// CloudProviderConfiguration holds configuration for CloudProvider related features.
	CloudProvider CloudProviderConfiguration
	// DebuggingConfiguration holds configuration for Debugging related features.
	Debugging DebuggingConfiguration
	// GenericComponentConfiguration holds configuration for GenericComponent
	// related features both in cloud controller manager and kube-controller manager.
	GenericComponent GenericComponentConfiguration
	// KubeCloudSharedConfiguration holds configuration for shared related features
	// both in cloud controller manager and kube-controller manager.
	KubeCloudShared KubeCloudSharedConfiguration
	// ServiceControllerConfiguration holds configuration for ServiceController
	// related features.
	ServiceController ServiceControllerConfiguration
	// NodeStatusUpdateFrequency is the frequency at which the controller updates nodes' status
	NodeStatusUpdateFrequency metav1.Duration
}

type CloudProviderConfiguration struct {
	// Name is the provider for cloud services.
	Name string `json:"cloudProvider"`
	// cloudConfigFile is the path to the cloud provider configuration file.
	CloudConfigFile string `json:"cloudConfigFile"`
}

type DebuggingConfiguration struct {
	// enableProfiling enables profiling via web interface host:port/debug/pprof/
	EnableProfiling bool `json:"enableProfiling"`
	// EnableContentionProfiling enables lock contention profiling, if
	// EnableProfiling is true.
	EnableContentionProfiling bool `json:"enableContentionProfiling"`
}

type GenericComponentConfiguration struct {

	// minResyncPeriod is the resync period in reflectors; will be random between
	// minResyncPeriod and 2*minResyncPeriod.
	MinResyncPeriod metav1.Duration `json:"minResyncPeriod"`
	// contentType is contentType of requests sent to apiserver.
	ContentType string `json:"contentType"`
	// kubeAPIQPS is the QPS to use while talking with kubernetes apiserver.
	KubeAPIQPS float32 `json:"kubeAPIQPS"`
	// kubeAPIBurst is the burst to use while talking with kubernetes apiserver.
	KubeAPIBurst int32 `json:"kubeAPIBurst"`
	// How long to wait between starting controller managers
	ControllerStartInterval metav1.Duration `json:"controllerStartInterval"`
	// leaderElection defines the configuration of leader election client.
	LeaderElection LeaderElectionConfiguration `json:"leaderElection"`
}

type KubeCloudSharedConfiguration struct {
	// port is the port that the controller-manager's http service runs on.
	Port int32 `json:"port"`
	// address is the IP address to serve on (set to 0.0.0.0 for all interfaces).
	Address string `json:"address"`
	// useServiceAccountCredentials indicates whether controllers should be run with
	// individual service account credentials.
	UseServiceAccountCredentials bool `json:"useServiceAccountCredentials"`
	// run with untagged cloud instances
	AllowUntaggedCloud bool `json:"allowUntaggedCloud"`
	// routeReconciliationPeriod is the period for reconciling routes created for Nodes by cloud provider..
	RouteReconciliationPeriod metav1.Duration `json:"routeReconciliationPeriod"`
	// nodeMonitorPeriod is the period for syncing NodeStatus in NodeController.
	NodeMonitorPeriod metav1.Duration `json:"nodeMonitorPeriod"`
	// clusterName is the instance prefix for the cluster.
	ClusterName string `json:"clusterName"`
	// clusterCIDR is CIDR Range for Pods in cluster.
	ClusterCIDR string `json:"clusterCIDR"`
	// AllocateNodeCIDRs enables CIDRs for Pods to be allocated and, if
	// ConfigureCloudRoutes is true, to be set on the cloud provider.
	AllocateNodeCIDRs bool `json:"allocateNodeCIDRs"`
	// CIDRAllocatorType determines what kind of pod CIDR allocator will be used.
	CIDRAllocatorType string `json:"cIDRAllocatorType"`
	// configureCloudRoutes enables CIDRs allocated with allocateNodeCIDRs
	// to be configured on the cloud provider.
	ConfigureCloudRoutes *bool `json:"configureCloudRoutes"`
	// nodeSyncPeriod is the period for syncing nodes from cloudprovider. Longer
	// periods will result in fewer calls to cloud provider, but may delay addition
	// of new nodes to cluster.
	NodeSyncPeriod metav1.Duration `json:"nodeSyncPeriod"`
}

type AttachDetachControllerConfiguration struct {
	// Reconciler runs a periodic loop to reconcile the desired state of the with
	// the actual state of the world by triggering attach detach operations.
	// This flag enables or disables reconcile.  Is false by default, and thus enabled.
	DisableAttachDetachReconcilerSync bool `json:"disableAttachDetachReconcilerSync"`
	// ReconcilerSyncLoopPeriod is the amount of time the reconciler sync states loop
	// wait between successive executions. Is set to 5 sec by default.
	ReconcilerSyncLoopPeriod metav1.Duration `json:"reconcilerSyncLoopPeriod"`
}

type CSRSigningControllerConfiguration struct {
	// clusterSigningCertFile is the filename containing a PEM-encoded
	// X509 CA certificate used to issue cluster-scoped certificates
	ClusterSigningCertFile string `json:"clusterSigningCertFile"`
	// clusterSigningCertFile is the filename containing a PEM-encoded
	// RSA or ECDSA private key used to issue cluster-scoped certificates
	ClusterSigningKeyFile string `json:"clusterSigningKeyFile"`
	// clusterSigningDuration is the length of duration signed certificates
	// will be given.
	ClusterSigningDuration metav1.Duration `json:"clusterSigningDuration"`
}

type DaemonSetControllerConfiguration struct {
	// concurrentDaemonSetSyncs is the number of daemonset objects that are
	// allowed to sync concurrently. Larger number = more responsive daemonset,
	// but more CPU (and network) load.
	ConcurrentDaemonSetSyncs int32 `json:"concurrentDaemonSetSyncs"`
}

type DeploymentControllerConfiguration struct {
	// concurrentDeploymentSyncs is the number of deployment objects that are
	// allowed to sync concurrently. Larger number = more responsive deployments,
	// but more CPU (and network) load.
	ConcurrentDeploymentSyncs int32 `json:"concurrentDeploymentSyncs"`
	// deploymentControllerSyncPeriod is the period for syncing the deployments.
	DeploymentControllerSyncPeriod metav1.Duration `json:"deploymentControllerSyncPeriod"`
}

type DeprecatedControllerConfiguration struct {
	// DEPRECATED: deletingPodsQps is the number of nodes per second on which pods are deleted in
	// case of node failure.
	DeletingPodsQps float32 `json:"deletingPodsQps"`
	// DEPRECATED: deletingPodsBurst is the number of nodes on which pods are bursty deleted in
	// case of node failure. For more details look into RateLimiter.
	DeletingPodsBurst int32 `json:"deletingPodsBurst"`
	// registerRetryCount is the number of retries for initial node registration.
	// Retry interval equals node-sync-period.
	RegisterRetryCount int32 `json:"registerRetryCount"`
}

type EndPointControllerConfiguration struct {
	// concurrentEndpointSyncs is the number of endpoint syncing operations
	// that will be done concurrently. Larger number = faster endpoint updating,
	// but more CPU (and network) load.
	ConcurrentEndpointSyncs int32 `json:"concurrentEndpointSyncs"`
}

type GarbageCollectorControllerConfiguration struct {
	// enables the generic garbage collector. MUST be synced with the
	// corresponding flag of the kube-apiserver. WARNING: the generic garbage
	// collector is an alpha feature.
	EnableGarbageCollector *bool `json:"enableGarbageCollector"`
	// concurrentGCSyncs is the number of garbage collector workers that are
	// allowed to sync concurrently.
	ConcurrentGCSyncs int32 `json:"concurrentGCSyncs"`
	// gcIgnoredResources is the list of GroupResources that garbage collection should ignore.
	GCIgnoredResources []GroupResource `json:"gCIgnoredResources"`
}

type HPAControllerConfiguration struct {
	// horizontalPodAutoscalerSyncPeriod is the period for syncing the number of
	// pods in horizontal pod autoscaler.
	HorizontalPodAutoscalerSyncPeriod metav1.Duration `json:"horizontalPodAutoscalerSyncPeriod"`
	// horizontalPodAutoscalerUpscaleForbiddenWindow is a period after which next upscale allowed.
	HorizontalPodAutoscalerUpscaleForbiddenWindow metav1.Duration `json:"horizontalPodAutoscalerUpscaleForbiddenWindow"`
	// horizontalPodAutoscalerDownscaleForbiddenWindow is a period after which next downscale allowed.
	HorizontalPodAutoscalerDownscaleForbiddenWindow metav1.Duration `json:"horizontalPodAutoscalerDownscaleForbiddenWindow"`
	// horizontalPodAutoscalerTolerance is the tolerance for when
	// resource usage suggests upscaling/downscaling
	HorizontalPodAutoscalerTolerance float64 `json:"horizontalPodAutoscalerTolerance"`
	// HorizontalPodAutoscalerUseRESTClients causes the HPA controller to use REST clients
	// through the kube-aggregator when enabled, instead of using the legacy metrics client
	// through the API server proxy.
	HorizontalPodAutoscalerUseRESTClients *bool `json:"horizontalPodAutoscalerUseRESTClients"`
}

type JobControllerConfiguration struct {
	// concurrentJobSyncs is the number of job objects that are
	// allowed to sync concurrently. Larger number = more responsive jobs,
	// but more CPU (and network) load.
	ConcurrentJobSyncs int32
}

type NamespaceControllerConfiguration struct {
	// namespaceSyncPeriod is the period for syncing namespace life-cycle
	// updates.
	NamespaceSyncPeriod metav1.Duration `json:"namespaceSyncPeriod"`
	// concurrentNamespaceSyncs is the number of namespace objects that are
	// allowed to sync concurrently.
	ConcurrentNamespaceSyncs int32 `json:"concurrentNamespaceSyncs"`
}

type NodeIpamControllerConfiguration struct {
	// serviceCIDR is CIDR Range for Services in cluster.
	ServiceCIDR string `json:"serviceCIDR"`
	// NodeCIDRMaskSize is the mask size for node cidr in cluster.
	NodeCIDRMaskSize int32 `json:"nodeCIDRMaskSize"`
}

type NodeLifecycleControllerConfiguration struct {
	// If set to true enables NoExecute Taints and will evict all not-tolerating
	// Pod running on Nodes tainted with this kind of Taints.
	EnableTaintManager *bool `json:"enableTaintManager"`
	// nodeEvictionRate is the number of nodes per second on which pods are deleted in case of node failure when a zone is healthy
	NodeEvictionRate float32 `json:"nodeEvictionRate"`
	// secondaryNodeEvictionRate is the number of nodes per second on which pods are deleted in case of node failure when a zone is unhealthy
	SecondaryNodeEvictionRate float32 `json:"secondaryNodeEvictionRate"`
	// nodeStartupGracePeriod is the amount of time which we allow starting a node to
	// be unresponsive before marking it unhealthy.
	NodeStartupGracePeriod metav1.Duration `json:"nodeStartupGracePeriod"`
	// nodeMontiorGracePeriod is the amount of time which we allow a running node to be
	// unresponsive before marking it unhealthy. Must be N times more than kubelet's
	// nodeStatusUpdateFrequency, where N means number of retries allowed for kubelet
	// to post node status.
	NodeMonitorGracePeriod metav1.Duration `json:"nodeMonitorGracePeriod"`
	// podEvictionTimeout is the grace period for deleting pods on failed nodes.
	PodEvictionTimeout metav1.Duration `json:"podEvictionTimeout"`
	// secondaryNodeEvictionRate is implicitly overridden to 0 for clusters smaller than or equal to largeClusterSizeThreshold
	LargeClusterSizeThreshold int32 `json:"largeClusterSizeThreshold"`
	// Zone is treated as unhealthy in nodeEvictionRate and secondaryNodeEvictionRate when at least
	// unhealthyZoneThreshold (no less than 3) of Nodes in the zone are NotReady
	UnhealthyZoneThreshold float32 `json:"unhealthyZoneThreshold"`
}

type PersistentVolumeBinderControllerConfiguration struct {
	// pvClaimBinderSyncPeriod is the period for syncing persistent volumes
	// and persistent volume claims.
	PVClaimBinderSyncPeriod metav1.Duration `json:"pVClaimBinderSyncPeriod"`
	// volumeConfiguration holds configuration for volume related features.
	VolumeConfiguration VolumeConfiguration `json:"volumeConfiguration"`
}

type PodGCControllerConfiguration struct {
	// terminatedPodGCThreshold is the number of terminated pods that can exist
	// before the terminated pod garbage collector starts deleting terminated pods.
	// If <= 0, the terminated pod garbage collector is disabled.
	TerminatedPodGCThreshold int32 `json:"terminatedPodGCThreshold"`
}

type ReplicaSetControllerConfiguration struct {
	// concurrentRSSyncs is the number of replica sets that are  allowed to sync
	// concurrently. Larger number = more responsive replica  management, but more
	// CPU (and network) load.
	ConcurrentRSSyncs int32 `json:"concurrentRSSyncs"`
}

type ReplicationControllerConfiguration struct {
	// concurrentRCSyncs is the number of replication controllers that are
	// allowed to sync concurrently. Larger number = more responsive replica
	// management, but more CPU (and network) load.
	ConcurrentRCSyncs int32 `json:"concurrentRCSyncs"`
}

type ResourceQuotaControllerConfiguration struct {
	// resourceQuotaSyncPeriod is the period for syncing quota usage status
	// in the system.
	ResourceQuotaSyncPeriod metav1.Duration `json:"resourceQuotaSyncPeriod"`
	// concurrentResourceQuotaSyncs is the number of resource quotas that are
	// allowed to sync concurrently. Larger number = more responsive quota
	// management, but more CPU (and network) load.
	ConcurrentResourceQuotaSyncs int32 `json:"concurrentResourceQuotaSyncs"`
}

type SAControllerConfiguration struct {
	// serviceAccountKeyFile is the filename containing a PEM-encoded private RSA key
	// used to sign service account tokens.
	ServiceAccountKeyFile string `json:"serviceAccountKeyFile"`
	// concurrentSATokenSyncs is the number of service account token syncing operations
	// that will be done concurrently.
	ConcurrentSATokenSyncs int32 `json:"concurrentSATokenSyncs"`
	// rootCAFile is the root certificate authority will be included in service
	// account's token secret. This must be a valid PEM-encoded CA bundle.
	RootCAFile string `json:"rootCAFile"`
}

type ServiceControllerConfiguration struct {
	// concurrentServiceSyncs is the number of services that are
	// allowed to sync concurrently. Larger number = more responsive service
	// management, but more CPU (and network) load.
	ConcurrentServiceSyncs int32 `json:"concurrentServiceSyncs"`
}

const (
	// "kube-system" is the default scheduler lock object namespace
	SchedulerDefaultLockObjectNamespace string = "kube-system"

	// "kube-scheduler" is the default scheduler lock object name
	SchedulerDefaultLockObjectName = "kube-scheduler"

	SchedulerDefaultProviderName = "DefaultProvider"
)
