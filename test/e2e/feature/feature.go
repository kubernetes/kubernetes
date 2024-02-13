/*
Copyright 2023 The Kubernetes Authors.

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

// Package feature contains pre-defined features used by test/e2e and/or
// test/e2e_node.
package feature

import (
	"k8s.io/kubernetes/test/e2e/framework"
)

var (
	// Please keep the list in alphabetical order.

	APIServerIdentity                       = framework.WithFeature(framework.ValidFeatures.Add("APIServerIdentity"))
	AppArmor                                = framework.WithFeature(framework.ValidFeatures.Add("AppArmor"))
	BootstrapTokens                         = framework.WithFeature(framework.ValidFeatures.Add("BootstrapTokens"))
	BoundServiceAccountTokenVolume          = framework.WithFeature(framework.ValidFeatures.Add("BoundServiceAccountTokenVolume"))
	CloudProvider                           = framework.WithFeature(framework.ValidFeatures.Add("CloudProvider"))
	ClusterAutoscalerScalability1           = framework.WithFeature(framework.ValidFeatures.Add("ClusterAutoscalerScalability1"))
	ClusterAutoscalerScalability2           = framework.WithFeature(framework.ValidFeatures.Add("ClusterAutoscalerScalability2"))
	ClusterAutoscalerScalability3           = framework.WithFeature(framework.ValidFeatures.Add("ClusterAutoscalerScalability3"))
	ClusterAutoscalerScalability4           = framework.WithFeature(framework.ValidFeatures.Add("ClusterAutoscalerScalability4"))
	ClusterAutoscalerScalability5           = framework.WithFeature(framework.ValidFeatures.Add("ClusterAutoscalerScalability5"))
	ClusterAutoscalerScalability6           = framework.WithFeature(framework.ValidFeatures.Add("ClusterAutoscalerScalability6"))
	ClusterDowngrade                        = framework.WithFeature(framework.ValidFeatures.Add("ClusterDowngrade"))
	ClusterScaleUpBypassScheduler           = framework.WithFeature(framework.ValidFeatures.Add("ClusterScaleUpBypassScheduler"))
	ClusterSizeAutoscalingGpu               = framework.WithFeature(framework.ValidFeatures.Add("ClusterSizeAutoscalingGpu"))
	ClusterSizeAutoscalingScaleDown         = framework.WithFeature(framework.ValidFeatures.Add("ClusterSizeAutoscalingScaleDown"))
	ClusterSizeAutoscalingScaleUp           = framework.WithFeature(framework.ValidFeatures.Add("ClusterSizeAutoscalingScaleUp"))
	ClusterTrustBundle                      = framework.WithFeature(framework.ValidFeatures.Add("ClusterTrustBundle"))
	ClusterTrustBundleProjection            = framework.WithFeature(framework.ValidFeatures.Add("ClusterTrustBundleProjection"))
	ClusterUpgrade                          = framework.WithFeature(framework.ValidFeatures.Add("ClusterUpgrade"))
	ComprehensiveNamespaceDraining          = framework.WithFeature(framework.ValidFeatures.Add("ComprehensiveNamespaceDraining"))
	CPUManager                              = framework.WithFeature(framework.ValidFeatures.Add("CPUManager"))
	CustomMetricsAutoscaling                = framework.WithFeature(framework.ValidFeatures.Add("CustomMetricsAutoscaling"))
	DeviceManager                           = framework.WithFeature(framework.ValidFeatures.Add("DeviceManager"))
	DevicePluginProbe                       = framework.WithFeature(framework.ValidFeatures.Add("DevicePluginProbe"))
	Downgrade                               = framework.WithFeature(framework.ValidFeatures.Add("Downgrade"))
	DynamicResourceAllocation               = framework.WithFeature(framework.ValidFeatures.Add("DynamicResourceAllocation"))
	EphemeralStorage                        = framework.WithFeature(framework.ValidFeatures.Add("EphemeralStorage"))
	Example                                 = framework.WithFeature(framework.ValidFeatures.Add("Example"))
	ExperimentalResourceUsageTracking       = framework.WithFeature(framework.ValidFeatures.Add("ExperimentalResourceUsageTracking"))
	Flexvolumes                             = framework.WithFeature(framework.ValidFeatures.Add("Flexvolumes"))
	GKENodePool                             = framework.WithFeature(framework.ValidFeatures.Add("GKENodePool"))
	GPUClusterDowngrade                     = framework.WithFeature(framework.ValidFeatures.Add("GPUClusterDowngrade"))
	GPUClusterUpgrade                       = framework.WithFeature(framework.ValidFeatures.Add("GPUClusterUpgrade"))
	GPUDevicePlugin                         = framework.WithFeature(framework.ValidFeatures.Add("GPUDevicePlugin"))
	GPUMasterUpgrade                        = framework.WithFeature(framework.ValidFeatures.Add("GPUMasterUpgrade"))
	GPUUpgrade                              = framework.WithFeature(framework.ValidFeatures.Add("GPUUpgrade"))
	HAMaster                                = framework.WithFeature(framework.ValidFeatures.Add("HAMaster"))
	HPA                                     = framework.WithFeature(framework.ValidFeatures.Add("HPA"))
	HugePages                               = framework.WithFeature(framework.ValidFeatures.Add("HugePages"))
	Ingress                                 = framework.WithFeature(framework.ValidFeatures.Add("Ingress"))
	IngressScale                            = framework.WithFeature(framework.ValidFeatures.Add("IngressScale"))
	InPlacePodVerticalScaling               = framework.WithFeature(framework.ValidFeatures.Add("InPlacePodVerticalScaling"))
	IPv6DualStack                           = framework.WithFeature(framework.ValidFeatures.Add("IPv6DualStack"))
	Kind                                    = framework.WithFeature(framework.ValidFeatures.Add("Kind"))
	KubeletCredentialProviders              = framework.WithFeature(framework.ValidFeatures.Add("KubeletCredentialProviders"))
	KubeletSecurity                         = framework.WithFeature(framework.ValidFeatures.Add("KubeletSecurity"))
	KubeProxyDaemonSetDowngrade             = framework.WithFeature(framework.ValidFeatures.Add("KubeProxyDaemonSetDowngrade"))
	KubeProxyDaemonSetUpgrade               = framework.WithFeature(framework.ValidFeatures.Add("KubeProxyDaemonSetUpgrade"))
	KubeProxyDaemonSetMigration             = framework.WithFeature(framework.ValidFeatures.Add("KubeProxyDaemonSetMigration"))
	LabelSelector                           = framework.WithFeature(framework.ValidFeatures.Add("LabelSelector"))
	LocalStorageCapacityIsolation           = framework.WithFeature(framework.ValidFeatures.Add("LocalStorageCapacityIsolation"))
	LocalStorageCapacityIsolationQuota      = framework.WithFeature(framework.ValidFeatures.Add("LocalStorageCapacityIsolationQuota"))
	MasterUpgrade                           = framework.WithFeature(framework.ValidFeatures.Add("MasterUpgrade"))
	MemoryManager                           = framework.WithFeature(framework.ValidFeatures.Add("MemoryManager"))
	NEG                                     = framework.WithFeature(framework.ValidFeatures.Add("NEG"))
	NetworkingDNS                           = framework.WithFeature(framework.ValidFeatures.Add("Networking-DNS"))
	NetworkingIPv4                          = framework.WithFeature(framework.ValidFeatures.Add("Networking-IPv4"))
	NetworkingIPv6                          = framework.WithFeature(framework.ValidFeatures.Add("Networking-IPv6"))
	NetworkingPerformance                   = framework.WithFeature(framework.ValidFeatures.Add("Networking-Performance"))
	NetworkPolicy                           = framework.WithFeature(framework.ValidFeatures.Add("NetworkPolicy"))
	NodeAuthenticator                       = framework.WithFeature(framework.ValidFeatures.Add("NodeAuthenticator"))
	NodeAuthorizer                          = framework.WithFeature(framework.ValidFeatures.Add("NodeAuthorizer"))
	NodeLogQuery                            = framework.WithFeature(framework.ValidFeatures.Add("NodeLogQuery"))
	NodeOutOfServiceVolumeDetach            = framework.WithFeature(framework.ValidFeatures.Add("NodeOutOfServiceVolumeDetach"))
	NoSNAT                                  = framework.WithFeature(framework.ValidFeatures.Add("NoSNAT"))
	PersistentVolumeLastPhaseTransitionTime = framework.WithFeature(framework.ValidFeatures.Add("PersistentVolumeLastPhaseTransitionTime"))
	PerformanceDNS                          = framework.WithFeature(framework.ValidFeatures.Add("PerformanceDNS"))
	PodGarbageCollector                     = framework.WithFeature(framework.ValidFeatures.Add("PodGarbageCollector"))
	PodHostIPs                              = framework.WithFeature(framework.ValidFeatures.Add("PodHostIPs"))
	PodLifecycleSleepAction                 = framework.WithFeature(framework.ValidFeatures.Add("PodLifecycleSleepAction"))
	PodPriority                             = framework.WithFeature(framework.ValidFeatures.Add("PodPriority"))
	PodReadyToStartContainersCondition      = framework.WithFeature(framework.ValidFeatures.Add("PodReadyToStartContainersCondition"))
	PodResources                            = framework.WithFeature(framework.ValidFeatures.Add("PodResources"))
	Reboot                                  = framework.WithFeature(framework.ValidFeatures.Add("Reboot"))
	ReclaimPolicy                           = framework.WithFeature(framework.ValidFeatures.Add("ReclaimPolicy"))
	RecoverVolumeExpansionFailure           = framework.WithFeature(framework.ValidFeatures.Add("RecoverVolumeExpansionFailure"))
	Recreate                                = framework.WithFeature(framework.ValidFeatures.Add("Recreate"))
	RegularResourceUsageTracking            = framework.WithFeature(framework.ValidFeatures.Add("RegularResourceUsageTracking"))
	ScopeSelectors                          = framework.WithFeature(framework.ValidFeatures.Add("ScopeSelectors"))
	SCTPConnectivity                        = framework.WithFeature(framework.ValidFeatures.Add("SCTPConnectivity"))
	SeccompDefault                          = framework.WithFeature(framework.ValidFeatures.Add("SeccompDefault"))
	SELinux                                 = framework.WithFeature(framework.ValidFeatures.Add("SELinux"))
	SELinuxMountReadWriteOncePod            = framework.WithFeature(framework.ValidFeatures.Add("SELinuxMountReadWriteOncePod"))
	ServiceCIDRs                            = framework.WithFeature(framework.ValidFeatures.Add("ServiceCIDRs"))
	SidecarContainers                       = framework.WithFeature(framework.ValidFeatures.Add("SidecarContainers"))
	StackdriverAcceleratorMonitoring        = framework.WithFeature(framework.ValidFeatures.Add("StackdriverAcceleratorMonitoring"))
	StackdriverCustomMetrics                = framework.WithFeature(framework.ValidFeatures.Add("StackdriverCustomMetrics"))
	StackdriverExternalMetrics              = framework.WithFeature(framework.ValidFeatures.Add("StackdriverExternalMetrics"))
	StackdriverMetadataAgent                = framework.WithFeature(framework.ValidFeatures.Add("StackdriverMetadataAgent"))
	StackdriverMonitoring                   = framework.WithFeature(framework.ValidFeatures.Add("StackdriverMonitoring"))
	StandaloneMode                          = framework.WithFeature(framework.ValidFeatures.Add("StandaloneMode"))
	StatefulSet                             = framework.WithFeature(framework.ValidFeatures.Add("StatefulSet"))
	StatefulSetStartOrdinal                 = framework.WithFeature(framework.ValidFeatures.Add("StatefulSetStartOrdinal"))
	StatefulUpgrade                         = framework.WithFeature(framework.ValidFeatures.Add("StatefulUpgrade"))
	StorageProvider                         = framework.WithFeature(framework.ValidFeatures.Add("StorageProvider"))
	StorageVersionAPI                       = framework.WithFeature(framework.ValidFeatures.Add("StorageVersionAPI"))
	TopologyHints                           = framework.WithFeature(framework.ValidFeatures.Add("Topology Hints"))
	TopologyManager                         = framework.WithFeature(framework.ValidFeatures.Add("TopologyManager"))
	UDP                                     = framework.WithFeature(framework.ValidFeatures.Add("UDP"))
	Upgrade                                 = framework.WithFeature(framework.ValidFeatures.Add("Upgrade"))
	UserNamespacesSupport                   = framework.WithFeature(framework.ValidFeatures.Add("UserNamespacesSupport"))
	ValidatingAdmissionPolicy               = framework.WithFeature(framework.ValidFeatures.Add("ValidatingAdmissionPolicy"))
	Volumes                                 = framework.WithFeature(framework.ValidFeatures.Add("Volumes"))
	VolumeSnapshotDataSource                = framework.WithFeature(framework.ValidFeatures.Add("VolumeSnapshotDataSource"))
	VolumeSourceXFS                         = framework.WithFeature(framework.ValidFeatures.Add("VolumeSourceXFS"))
	Vsphere                                 = framework.WithFeature(framework.ValidFeatures.Add("vsphere"))
	WatchList                               = framework.WithFeature(framework.ValidFeatures.Add("WatchList"))
	Windows                                 = framework.WithFeature(framework.ValidFeatures.Add("Windows"))
	WindowsHostProcessContainers            = framework.WithFeature(framework.ValidFeatures.Add("WindowsHostProcessContainers"))
	WindowsHyperVContainers                 = framework.WithFeature(framework.ValidFeatures.Add("WindowsHyperVContainers"))

	// Please keep the list in alphabetical order.
)

func init() {
	// This prevents adding additional ad-hoc features in tests.
	framework.ValidFeatures.Freeze()
}
