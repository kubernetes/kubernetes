/*
Copyright 2024 The Kubernetes Authors.

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

package features

import (
	"k8s.io/apimachinery/pkg/util/version"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/component-base/featuregate"
)

// defaultVersionedKubernetesFeatureGates consists of all known Kubernetes-specific feature keys with VersionedSpecs.
// To add a new feature, define a key for it and add it here. The features will be
// available throughout Kubernetes binaries.
//
// Entries are alphabetized and separated from each other with blank lines to avoid sweeping gofmt changes
// when adding or removing one entry.
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.VersionedSpecs{
	CrossNamespaceVolumeDataSource: {
		{Version: version.MustParse("1.26"), Default: false, PreRelease: featuregate.Alpha},
	},
	AnyVolumeDataSource: {
		{Version: version.MustParse("1.18"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.24"), Default: true, PreRelease: featuregate.Beta},
	},
	AppArmor: {
		{Version: version.MustParse("1.4"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.33
	},
	AppArmorFields: {
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.33
	},
	AuthorizeNodeWithSelectors: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Alpha},
	},
	ClusterTrustBundle: {
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
	},
	ClusterTrustBundleProjection: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
	},
	CPUCFSQuotaPeriod: {
		{Version: version.MustParse("1.12"), Default: false, PreRelease: featuregate.Alpha},
	},
	CPUManager: {
		{Version: version.MustParse("1.8"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.10"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.26"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.26
	},
	CPUManagerPolicyAlphaOptions: {
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Alpha},
	},
	CPUManagerPolicyBetaOptions: {
		{Version: version.MustParse("1.23"), Default: true, PreRelease: featuregate.Beta},
	},
	CPUManagerPolicyOptions: {
		{Version: version.MustParse("1.22"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.23"), Default: true, PreRelease: featuregate.Beta},
	},
	CSIMigrationPortworx: {
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta}, // On by default (requires Portworx CSI driver)
	},
	CSIVolumeHealth: {
		{Version: version.MustParse("1.21"), Default: false, PreRelease: featuregate.Alpha},
	},
	ContainerCheckpoint: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
	},
	CronJobsScheduledAnnotation: {
		{Version: version.MustParse("1.28"), Default: true, PreRelease: featuregate.Beta},
	},
	DevicePluginCDIDevices: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.29"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.33
	},
	DisableAllocatorDualWrite: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Alpha}, // remove after MultiCIDRServiceAllocator is GA
	},
	DisableCloudProviders: {
		{Version: version.MustParse("1.22"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.29"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},
	DisableKubeletCloudCredentialProviders: {
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.29"), Default: true, PreRelease: featuregate.Beta},
	},
	DRAControlPlaneController: {
		{Version: version.MustParse("1.26"), Default: false, PreRelease: featuregate.Alpha},
	},
	DynamicResourceAllocation: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
	},
	EventedPLEG: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Alpha},
	},
	ExecProbeTimeout: {
		{Version: version.MustParse("1.20"), Default: true, PreRelease: featuregate.GA}, // lock to default and remove after v1.22 based on KEP #1972 update
	},
	GracefulNodeShutdown: {
		{Version: version.MustParse("1.20"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.21"), Default: true, PreRelease: featuregate.Beta},
	},
	GracefulNodeShutdownBasedOnPodPriority: {
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.24"), Default: true, PreRelease: featuregate.Beta},
	},
	HPAContainerMetrics: {
		{Version: version.MustParse("1.20"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.27"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.32
	},
	HonorPVReclaimPolicy: {
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
	},
	InTreePluginPortworxUnregister: {
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Alpha},
	},
	JobBackoffLimitPerIndex: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.29"), Default: true, PreRelease: featuregate.Beta},
	},
	JobManagedBy: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},
	JobPodFailurePolicy: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.26"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.33
	},
	JobPodReplacementPolicy: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.29"), Default: true, PreRelease: featuregate.Beta},
	},
	JobSuccessPolicy: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
	},
	KubeletCgroupDriverFromCRI: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
	},
	KubeletInUserNamespace: {
		{Version: version.MustParse("1.22"), Default: false, PreRelease: featuregate.Alpha},
	},
	KubeletPodResourcesDynamicResources: {
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
	},
	KubeletPodResourcesGet: {
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
	},
	KubeletSeparateDiskGC: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
	},
	KubeletTracing: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.27"), Default: true, PreRelease: featuregate.Beta},
	},
	KubeProxyDrainingTerminatingNodes: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.31; remove in 1.33
	},
	LocalStorageCapacityIsolationFSQuotaMonitoring: {
		{Version: version.MustParse("1.15"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Beta},
	},
	LogarithmicScaleDown: {
		{Version: version.MustParse("1.21"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.22"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},
	MatchLabelKeysInPodAffinity: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
	},
	MatchLabelKeysInPodTopologySpread: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.27"), Default: true, PreRelease: featuregate.Beta},
	},
	MaxUnavailableStatefulSet: {
		{Version: version.MustParse("1.24"), Default: false, PreRelease: featuregate.Alpha},
	},
	MemoryManager: {
		{Version: version.MustParse("1.21"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.22"), Default: true, PreRelease: featuregate.Beta},
	},
	MemoryQoS: {
		{Version: version.MustParse("1.22"), Default: false, PreRelease: featuregate.Alpha},
	},
	MinDomainsInPodTopologySpread: {
		{Version: version.MustParse("1.24"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.27"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.32
	},
	MultiCIDRServiceAllocator: {
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Beta},
	},
	NewVolumeManagerReconstruction: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.28"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.32
	},
	NFTablesProxyMode: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
	},
	NodeLogQuery: {
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Beta},
	},
	NodeOutOfServiceVolumeDetach: {
		{Version: version.MustParse("1.24"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.26"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.28"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.31
	},
	NodeSwap: {
		{Version: version.MustParse("1.22"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
	},
	PDBUnhealthyPodEvictionPolicy: {
		{Version: version.MustParse("1.26"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.27"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.33
	},
	PersistentVolumeLastPhaseTransitionTime: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.29"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.33
	},
	PodAndContainerStatsFromCRI: {
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Alpha},
	},
	PodDeletionCost: {
		{Version: version.MustParse("1.21"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.22"), Default: true, PreRelease: featuregate.Beta},
	},
	PodDisruptionConditions: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.26"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.33
	},
	PodIndexLabel: {
		{Version: version.MustParse("1.28"), Default: true, PreRelease: featuregate.Beta},
	},
	PodReadyToStartContainersCondition: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.29"), Default: true, PreRelease: featuregate.Beta},
	},
	PodHostIPs: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.29"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.32
	},
	PodLifecycleSleepAction: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
	},
	PodSchedulingReadiness: {
		{Version: version.MustParse("1.26"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.27"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.30; remove in 1.32
	},
	PortForwardWebsockets: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
	},
	ProcMountType: {
		{Version: version.MustParse("1.12"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Beta},
	},
	QOSReserved: {
		{Version: version.MustParse("1.11"), Default: false, PreRelease: featuregate.Alpha},
	},
	RecoverVolumeExpansionFailure: {
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Alpha},
	},
	genericfeatures.AnonymousAuthConfigurableEndpoints: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.Beta},
	},

	RelaxedEnvironmentVariableValidation: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},
	ReloadKubeletServerCertificateFile: {
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
	},
	ResourceHealthStatus: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Alpha},
	},
	RotateKubeletServerCertificate: {
		{Version: version.MustParse("1.7"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.12"), Default: true, PreRelease: featuregate.Beta},
	},
	RuntimeClassInImageCriAPI: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
	},
	ElasticIndexedJob: {
		{Version: version.MustParse("1.27"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.31, remove in 1.32
	},
	SchedulerQueueingHints: {
		{Version: version.MustParse("1.27"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Beta},
	},
	SeparateTaintEvictionController: {
		{Version: version.MustParse("1.29"), Default: true, PreRelease: featuregate.Beta},
	},
	ServiceAccountTokenJTI: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
	},
	ServiceAccountTokenNodeBinding: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
	},
	ServiceAccountTokenNodeBindingValidation: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
	},
	ServiceAccountTokenPodNodeInfo: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
	},
	ServiceTrafficDistribution: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
	},
	SidecarContainers: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.29"), Default: true, PreRelease: featuregate.Beta},
	},
	SizeMemoryBackedVolumes: {
		{Version: version.MustParse("1.20"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.22"), Default: true, PreRelease: featuregate.Beta},
	},
	StatefulSetAutoDeletePVC: {
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.27"), Default: true, PreRelease: featuregate.Beta},
	},
	StatefulSetStartOrdinal: {
		{Version: version.MustParse("1.26"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.27"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.31, remove in 1.33
	},
	StorageVersionMigrator: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},
	TopologyAwareHints: {
		{Version: version.MustParse("1.21"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.24"), Default: true, PreRelease: featuregate.Beta},
	},
	TopologyManagerPolicyAlphaOptions: {
		{Version: version.MustParse("1.26"), Default: false, PreRelease: featuregate.Alpha},
	},
	TopologyManagerPolicyBetaOptions: {
		{Version: version.MustParse("1.26"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.28"), Default: true, PreRelease: featuregate.Beta},
	},
	TopologyManagerPolicyOptions: {
		{Version: version.MustParse("1.26"), Default: true, PreRelease: featuregate.Beta},
	},
	TranslateStreamCloseWebsocketRequests: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
	},
	UnknownVersionInteroperabilityProxy: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
	},
	UserNamespacesSupport: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Beta},
	},
	VolumeAttributesClass: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Beta},
	},
	VolumeCapacityPriority: {
		{Version: version.MustParse("1.21"), Default: false, PreRelease: featuregate.Alpha},
	},
	WinDSR: {
		{Version: version.MustParse("1.14"), Default: false, PreRelease: featuregate.Alpha},
	},
	WinOverlay: {
		{Version: version.MustParse("1.14"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.20"), Default: true, PreRelease: featuregate.Beta},
	},
	WindowsHostNetwork: {
		{Version: version.MustParse("1.26"), Default: true, PreRelease: featuregate.Alpha},
	},
	NodeInclusionPolicyInPodTopologySpread: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.26"), Default: true, PreRelease: featuregate.Beta},
	},
	SELinuxMountReadWriteOncePod: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.28"), Default: true, PreRelease: featuregate.Beta},
	},
	LoadBalancerIPMode: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
	},
	ImageMaximumGCAge: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
	},
	UserNamespacesPodSecurityStandards: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
	},
	SELinuxMount: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},
	SupplementalGroupsPolicy: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Alpha},
	},
	ImageVolume: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Alpha},
	},
	KubeletRegistrationGetOnExistsOnly: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Deprecated},
	},
}
