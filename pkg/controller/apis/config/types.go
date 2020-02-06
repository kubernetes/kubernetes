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

package config

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	componentbaseconfig "k8s.io/component-base/config"
	csrsigningconfig "k8s.io/kubernetes/pkg/controller/certificates/signer/config"
	daemonconfig "k8s.io/kubernetes/pkg/controller/daemon/config"
	deploymentconfig "k8s.io/kubernetes/pkg/controller/deployment/config"
	endpointconfig "k8s.io/kubernetes/pkg/controller/endpoint/config"
	endpointsliceconfig "k8s.io/kubernetes/pkg/controller/endpointslice/config"
	garbagecollectorconfig "k8s.io/kubernetes/pkg/controller/garbagecollector/config"
	jobconfig "k8s.io/kubernetes/pkg/controller/job/config"
	namespaceconfig "k8s.io/kubernetes/pkg/controller/namespace/config"
	nodeipamconfig "k8s.io/kubernetes/pkg/controller/nodeipam/config"
	nodelifecycleconfig "k8s.io/kubernetes/pkg/controller/nodelifecycle/config"
	poautosclerconfig "k8s.io/kubernetes/pkg/controller/podautoscaler/config"
	podgcconfig "k8s.io/kubernetes/pkg/controller/podgc/config"
	replicasetconfig "k8s.io/kubernetes/pkg/controller/replicaset/config"
	replicationconfig "k8s.io/kubernetes/pkg/controller/replication/config"
	resourcequotaconfig "k8s.io/kubernetes/pkg/controller/resourcequota/config"
	serviceconfig "k8s.io/kubernetes/pkg/controller/service/config"
	serviceaccountconfig "k8s.io/kubernetes/pkg/controller/serviceaccount/config"
	statefulsetconfig "k8s.io/kubernetes/pkg/controller/statefulset/config"
	ttlafterfinishedconfig "k8s.io/kubernetes/pkg/controller/ttlafterfinished/config"
	attachdetachconfig "k8s.io/kubernetes/pkg/controller/volume/attachdetach/config"
	persistentvolumeconfig "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/config"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// KubeControllerManagerConfiguration contains elements describing kube-controller manager.
type KubeControllerManagerConfiguration struct {
	metav1.TypeMeta

	// Generic holds configuration for a generic controller-manager
	Generic GenericControllerManagerConfiguration
	// KubeCloudSharedConfiguration holds configuration for shared related features
	// both in cloud controller manager and kube-controller manager.
	KubeCloudShared KubeCloudSharedConfiguration

	// AttachDetachControllerConfiguration holds configuration for
	// AttachDetachController related features.
	AttachDetachController attachdetachconfig.AttachDetachControllerConfiguration
	// CSRSigningControllerConfiguration holds configuration for
	// CSRSigningController related features.
	CSRSigningController csrsigningconfig.CSRSigningControllerConfiguration
	// DaemonSetControllerConfiguration holds configuration for DaemonSetController
	// related features.
	DaemonSetController daemonconfig.DaemonSetControllerConfiguration
	// DeploymentControllerConfiguration holds configuration for
	// DeploymentController related features.
	DeploymentController deploymentconfig.DeploymentControllerConfiguration
	// StatefulSetControllerConfiguration holds configuration for
	// StatefulSetController related features.
	StatefulSetController statefulsetconfig.StatefulSetControllerConfiguration
	// DeprecatedControllerConfiguration holds configuration for some deprecated
	// features.
	DeprecatedController DeprecatedControllerConfiguration
	// EndpointControllerConfiguration holds configuration for EndpointController
	// related features.
	EndpointController endpointconfig.EndpointControllerConfiguration
	// EndpointSliceControllerConfiguration holds configuration for
	// EndpointSliceController related features.
	EndpointSliceController endpointsliceconfig.EndpointSliceControllerConfiguration
	// GarbageCollectorControllerConfiguration holds configuration for
	// GarbageCollectorController related features.
	GarbageCollectorController garbagecollectorconfig.GarbageCollectorControllerConfiguration
	// HPAControllerConfiguration holds configuration for HPAController related features.
	HPAController poautosclerconfig.HPAControllerConfiguration
	// JobControllerConfiguration holds configuration for JobController related features.
	JobController jobconfig.JobControllerConfiguration
	// NamespaceControllerConfiguration holds configuration for NamespaceController
	// related features.
	NamespaceController namespaceconfig.NamespaceControllerConfiguration
	// NodeIPAMControllerConfiguration holds configuration for NodeIPAMController
	// related features.
	NodeIPAMController nodeipamconfig.NodeIPAMControllerConfiguration
	// NodeLifecycleControllerConfiguration holds configuration for
	// NodeLifecycleController related features.
	NodeLifecycleController nodelifecycleconfig.NodeLifecycleControllerConfiguration
	// PersistentVolumeBinderControllerConfiguration holds configuration for
	// PersistentVolumeBinderController related features.
	PersistentVolumeBinderController persistentvolumeconfig.PersistentVolumeBinderControllerConfiguration
	// PodGCControllerConfiguration holds configuration for PodGCController
	// related features.
	PodGCController podgcconfig.PodGCControllerConfiguration
	// ReplicaSetControllerConfiguration holds configuration for ReplicaSet related features.
	ReplicaSetController replicasetconfig.ReplicaSetControllerConfiguration
	// ReplicationControllerConfiguration holds configuration for
	// ReplicationController related features.
	ReplicationController replicationconfig.ReplicationControllerConfiguration
	// ResourceQuotaControllerConfiguration holds configuration for
	// ResourceQuotaController related features.
	ResourceQuotaController resourcequotaconfig.ResourceQuotaControllerConfiguration
	// SAControllerConfiguration holds configuration for ServiceAccountController
	// related features.
	SAController serviceaccountconfig.SAControllerConfiguration
	// ServiceControllerConfiguration holds configuration for ServiceController
	// related features.
	ServiceController serviceconfig.ServiceControllerConfiguration
	// TTLAfterFinishedControllerConfiguration holds configuration for
	// TTLAfterFinishedController related features.
	TTLAfterFinishedController ttlafterfinishedconfig.TTLAfterFinishedControllerConfiguration
}

// GenericControllerManagerConfiguration holds configuration for a generic controller-manager
type GenericControllerManagerConfiguration struct {
	// port is the port that the controller-manager's http service runs on.
	Port int32
	// address is the IP address to serve on (set to 0.0.0.0 for all interfaces).
	Address string
	// minResyncPeriod is the resync period in reflectors; will be random between
	// minResyncPeriod and 2*minResyncPeriod.
	MinResyncPeriod metav1.Duration
	// ClientConnection specifies the kubeconfig file and client connection
	// settings for the proxy server to use when communicating with the apiserver.
	ClientConnection componentbaseconfig.ClientConnectionConfiguration
	// How long to wait between starting controller managers
	ControllerStartInterval metav1.Duration
	// leaderElection defines the configuration of leader election client.
	LeaderElection componentbaseconfig.LeaderElectionConfiguration
	// Controllers is the list of controllers to enable or disable
	// '*' means "all enabled by default controllers"
	// 'foo' means "enable 'foo'"
	// '-foo' means "disable 'foo'"
	// first item for a particular name wins
	Controllers []string
	// DebuggingConfiguration holds configuration for Debugging related features.
	Debugging componentbaseconfig.DebuggingConfiguration
}

// KubeCloudSharedConfiguration contains elements shared by both kube-controller manager
// and cloud-controller manager, but not genericconfig.
type KubeCloudSharedConfiguration struct {
	// CloudProviderConfiguration holds configuration for CloudProvider related features.
	CloudProvider CloudProviderConfiguration
	// externalCloudVolumePlugin specifies the plugin to use when cloudProvider is "external".
	// It is currently used by the in repo cloud providers to handle node and volume control in the KCM.
	ExternalCloudVolumePlugin string
	// useServiceAccountCredentials indicates whether controllers should be run with
	// individual service account credentials.
	UseServiceAccountCredentials bool
	// run with untagged cloud instances
	AllowUntaggedCloud bool
	// routeReconciliationPeriod is the period for reconciling routes created for Nodes by cloud provider..
	RouteReconciliationPeriod metav1.Duration
	// nodeMonitorPeriod is the period for syncing NodeStatus in NodeController.
	NodeMonitorPeriod metav1.Duration
	// clusterName is the instance prefix for the cluster.
	ClusterName string
	// clusterCIDR is CIDR Range for Pods in cluster.
	ClusterCIDR string
	// AllocateNodeCIDRs enables CIDRs for Pods to be allocated and, if
	// ConfigureCloudRoutes is true, to be set on the cloud provider.
	AllocateNodeCIDRs bool
	// CIDRAllocatorType determines what kind of pod CIDR allocator will be used.
	CIDRAllocatorType string
	// configureCloudRoutes enables CIDRs allocated with allocateNodeCIDRs
	// to be configured on the cloud provider.
	ConfigureCloudRoutes bool
	// nodeSyncPeriod is the period for syncing nodes from cloudprovider. Longer
	// periods will result in fewer calls to cloud provider, but may delay addition
	// of new nodes to cluster.
	NodeSyncPeriod metav1.Duration
}

// CloudProviderConfiguration contains basically elements about cloud provider.
type CloudProviderConfiguration struct {
	// Name is the provider for cloud services.
	Name string
	// cloudConfigFile is the path to the cloud provider configuration file.
	CloudConfigFile string
}

// DeprecatedControllerConfiguration contains elements be deprecated.
type DeprecatedControllerConfiguration struct {
	// DEPRECATED: deletingPodsQps is the number of nodes per second on which pods are deleted in
	// case of node failure.
	DeletingPodsQPS float32
	// DEPRECATED: deletingPodsBurst is the number of nodes on which pods are bursty deleted in
	// case of node failure. For more details look into RateLimiter.
	DeletingPodsBurst int32
	// registerRetryCount is the number of retries for initial node registration.
	// Retry interval equals node-sync-period.
	RegisterRetryCount int32
}
