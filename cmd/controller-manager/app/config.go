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

package app

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiserver "k8s.io/apiserver/pkg/server"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	kubeconfig "k8s.io/kubernetes/pkg/kubeapiserver/config"
)

//GenericConfigInfo is the main context object for the controller manager.
type GenericConfigInfo struct {
	Port            int32
	Address         string
	CloudConfigFile string
	CloudProvider   string

	UseServiceAccountCredentials bool
	MinResyncPeriod              metav1.Duration
	ControllerStartInterval      metav1.Duration
	LeaderElection               componentconfig.LeaderElectionConfiguration

	ConcurrentServiceSyncs    int32
	ServiceAccountKeyFile     string
	AllowUntaggedCloud        bool
	RouteReconciliationPeriod metav1.Duration
	NodeMonitorPeriod         metav1.Duration
	ClusterName               string
	ClusterCIDR               string
	AllocateNodeCIDRs         bool
	CIDRAllocatorType         string
	ConfigureCloudRoutes      bool
	ContentType               string
	KubeAPIQPS                float32
	KubeAPIBurst              int32
}

//PersistentVolumeBinderControllerConfigInfo is the main context object for the controller manager.
type PersistentVolumeBinderControllerConfigInfo struct {
	PVClaimBinderSyncPeriod metav1.Duration
	VolumeConfiguration     componentconfig.VolumeConfiguration
}

//HPAControllerConfigInfo is the main context object for the controller manager.
type HPAControllerConfigInfo struct {
	HorizontalPodAutoscalerUseRESTClients           bool
	HorizontalPodAutoscalerTolerance                float64
	HorizontalPodAutoscalerDownscaleForbiddenWindow metav1.Duration
	HorizontalPodAutoscalerUpscaleForbiddenWindow   metav1.Duration
	HorizontalPodAutoscalerSyncPeriod               metav1.Duration
}

//NamespaceControllerConfigInfo is the main context object for the controller manager.
type NamespaceControllerConfigInfo struct {
	NamespaceSyncPeriod metav1.Duration
}

//NodeLifecycleControllerConfigInfo is the main context object for the controller manager.
type NodeLifecycleControllerConfigInfo struct {
	EnableTaintManager        bool
	NodeEvictionRate          float32
	SecondaryNodeEvictionRate float32
	NodeStartupGracePeriod    metav1.Duration
	NodeMonitorGracePeriod    metav1.Duration
	PodEvictionTimeout        metav1.Duration
	LargeClusterSizeThreshold int32
	UnhealthyZoneThreshold    float32
}

//CSRSigningControllerConfigInfo is the main context object for the controller manager.
type CSRSigningControllerConfigInfo struct {
	ClusterSigningDuration metav1.Duration
	ClusterSigningKeyFile  string
	ClusterSigningCertFile string
}

//AttachDetachControllerConfigInfo is the main context object for the controller manager.
type AttachDetachControllerConfigInfo struct {
	ReconcilerSyncLoopPeriod          metav1.Duration
	DisableAttachDetachReconcilerSync bool
}

//PodGCControllerConfigInfo is the main context object for the controller manager.
type PodGCControllerConfigInfo struct {
	TerminatedPodGCThreshold int32
}

//ResourceQuotaControllerConfigInfo is the main context object for the controller manager.
type ResourceQuotaControllerConfigInfo struct {
	ResourceQuotaSyncPeriod metav1.Duration
}

//GarbageCollectorControllerConfigInfo is the main context object for the controller manager.
type GarbageCollectorControllerConfigInfo struct {
	ConcurrentGCSyncs      int32
	GCIgnoredResources     []componentconfig.GroupResource
	EnableGarbageCollector bool
}

//ConcurrentResourcesSyncsConfigInfo is the main context object for the controller manager.
type ConcurrentResourcesSyncsConfigInfo struct {
	ConcurrentJobSyncs           int32
	ConcurrentDaemonSetSyncs     int32
	ConcurrentEndpointSyncs      int32
	ConcurrentRCSyncs            int32
	ConcurrentRSSyncs            int32
	ConcurrentDeploymentSyncs    int32
	ConcurrentSATokenSyncs       int32
	ConcurrentNamespaceSyncs     int32
	ConcurrentResourceQuotaSyncs int32
}

// ComponentConfigInfo is the main context object for the controller manager.
type ComponentConfigInfo struct {
	GenericConfig                          GenericConfigInfo
	PersistentVolumeBinderControllerConfig PersistentVolumeBinderControllerConfigInfo
	HPAControllerConfig                    HPAControllerConfigInfo
	NamespaceControllerConfig              NamespaceControllerConfigInfo
	NodeLifecycleControllerConfig          NodeLifecycleControllerConfigInfo
	CSRSigningControllerConfig             CSRSigningControllerConfigInfo
	AttachDetachControllerConfig           AttachDetachControllerConfigInfo
	PodGCControllerConfig                  PodGCControllerConfigInfo
	ResourceQuotaControllerConfig          ResourceQuotaControllerConfigInfo
	GarbageCollectorControllerConfig       GarbageCollectorControllerConfigInfo
	ConcurrentResourcesSyncsConfig         ConcurrentResourcesSyncsConfigInfo
	CloudProviderConfig                    kubeconfig.CloudProviderInfo

	// external config parameters
	Controllers                    []string
	ExternalCloudVolumePlugin      string
	NodeSyncPeriod                 metav1.Duration
	DeploymentControllerSyncPeriod metav1.Duration
	DeletingPodsQPS                float32
	DeletingPodsBurst              int32
	RegisterRetryCount             int32
	ServiceCIDR                    string
	NodeCIDRMaskSize               int32
	RootCAFile                     string

	EnableProfiling           bool
	EnableContentionProfiling bool
}

// Config is the main context object for the controller manager.
type Config struct {
	ComponentConfig ComponentConfigInfo

	SecureServing *apiserver.SecureServingInfo
	// TODO: remove deprecated insecure serving
	InsecureServing *InsecureServingInfo
	Authentication  apiserver.AuthenticationInfo
	Authorization   apiserver.AuthorizationInfo

	// the general kube client
	Client *clientset.Clientset

	// the client only used for leader election
	LeaderElectionClient *clientset.Clientset

	// the rest config for the master
	Kubeconfig *restclient.Config

	// the event sink
	EventRecorder record.EventRecorder
}

type completedConfig struct {
	*Config
}

// CompletedConfig same as Config, just to swap private object.
type CompletedConfig struct {
	// Embed a private pointer that cannot be instantiated outside of this package.
	*completedConfig
}

// Complete fills in any fields not set that are required to have valid data. It's mutating the receiver.
func (c *Config) Complete() CompletedConfig {
	cc := completedConfig{c}
	return CompletedConfig{&cc}
}
