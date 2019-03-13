/*
Copyright 2017 The Kubernetes Authors.

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
	apiextensionsfeatures "k8s.io/apiextensions-apiserver/pkg/features"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cloudfeatures "k8s.io/cloud-provider/features"
)

const (
	// Every feature gate should add method here following this template:
	//
	// // owner: @username
	// // alpha: v1.X
	// MyFeature utilfeature.Feature = "MyFeature"

	// owner: @tallclair
	// beta: v1.4
	AppArmor utilfeature.Feature = "AppArmor"

	// owner: @mtaufen
	// alpha: v1.4
	// beta: v1.11
	DynamicKubeletConfig utilfeature.Feature = "DynamicKubeletConfig"

	// owner: @pweil-
	// alpha: v1.5
	//
	// Default userns=host for containers that are using other host namespaces, host mounts, the pod
	// contains a privileged container, or specific non-namespaced capabilities (MKNOD, SYS_MODULE,
	// SYS_TIME). This should only be enabled if user namespace remapping is enabled in the docker daemon.
	ExperimentalHostUserNamespaceDefaultingGate utilfeature.Feature = "ExperimentalHostUserNamespaceDefaulting"

	// owner: @vishh
	// alpha: v1.5
	//
	// DEPRECATED - This feature is deprecated by Pod Priority and Preemption as of Kubernetes 1.13.
	// Ensures guaranteed scheduling of pods marked with a special pod annotation `scheduler.alpha.kubernetes.io/critical-pod`
	// and also prevents them from being evicted from a node.
	// Note: This feature is not supported for `BestEffort` pods.
	ExperimentalCriticalPodAnnotation utilfeature.Feature = "ExperimentalCriticalPodAnnotation"

	// owner: @jiayingz
	// beta: v1.10
	//
	// Enables support for Device Plugins
	DevicePlugins utilfeature.Feature = "DevicePlugins"

	// owner: @Huang-Wei
	// beta: v1.13
	//
	// Changes the logic behind evicting Pods from not ready Nodes
	// to take advantage of NoExecute Taints and Tolerations.
	TaintBasedEvictions utilfeature.Feature = "TaintBasedEvictions"

	// owner: @mikedanese
	// alpha: v1.7
	// beta: v1.12
	//
	// Gets a server certificate for the kubelet from the Certificate Signing
	// Request API instead of generating one self signed and auto rotates the
	// certificate as expiration approaches.
	RotateKubeletServerCertificate utilfeature.Feature = "RotateKubeletServerCertificate"

	// owner: @mikedanese
	// beta: v1.8
	//
	// Automatically renews the client certificate used for communicating with
	// the API server as the certificate approaches expiration.
	RotateKubeletClientCertificate utilfeature.Feature = "RotateKubeletClientCertificate"

	// owner: @msau42
	// alpha: v1.7
	// beta: v1.10
	// ga: v1.14
	//
	// A new volume type that supports local disks on a node.
	PersistentLocalVolumes utilfeature.Feature = "PersistentLocalVolumes"

	// owner: @jinxu
	// beta: v1.10
	//
	// New local storage types to support local storage capacity isolation
	LocalStorageCapacityIsolation utilfeature.Feature = "LocalStorageCapacityIsolation"

	// owner: @gnufied
	// beta: v1.11
	// Ability to Expand persistent volumes
	ExpandPersistentVolumes utilfeature.Feature = "ExpandPersistentVolumes"

	// owner: @mlmhl
	// alpha: v1.11
	// Ability to expand persistent volumes' file system without unmounting volumes.
	ExpandInUsePersistentVolumes utilfeature.Feature = "ExpandInUsePersistentVolumes"

	// owner: @gnufied
	// alpha: v1.14
	// Ability to expand CSI volumes
	ExpandCSIVolumes utilfeature.Feature = "ExpandCSIVolumes"

	// owner: @verb
	// alpha: v1.10
	//
	// Allows running a "debug container" in a pod namespaces to troubleshoot a running pod.
	DebugContainers utilfeature.Feature = "DebugContainers"

	// owner: @verb
	// beta: v1.12
	//
	// Allows all containers in a pod to share a process namespace.
	PodShareProcessNamespace utilfeature.Feature = "PodShareProcessNamespace"

	// owner: @bsalamat
	// alpha: v1.8
	// beta: v1.11
	// GA: v1.14
	//
	// Add priority to pods. Priority affects scheduling and preemption of pods.
	PodPriority utilfeature.Feature = "PodPriority"

	// owner: @k82cn
	// beta: v1.12
	//
	// Taint nodes based on their condition status for 'NetworkUnavailable',
	// 'MemoryPressure', 'PIDPressure' and 'DiskPressure'.
	TaintNodesByCondition utilfeature.Feature = "TaintNodesByCondition"

	// owner: @sjenning
	// alpha: v1.11
	//
	// Allows resource reservations at the QoS level preventing pods at lower QoS levels from
	// bursting into resources requested at higher QoS levels (memory only for now)
	QOSReserved utilfeature.Feature = "QOSReserved"

	// owner: @ConnorDoyle
	// alpha: v1.8
	// beta: v1.10
	//
	// Alternative container-level CPU affinity policies.
	CPUManager utilfeature.Feature = "CPUManager"

	// owner: @szuecs
	// alpha: v1.12
	//
	// Enable nodes to change CPUCFSQuotaPeriod
	CPUCFSQuotaPeriod utilfeature.Feature = "CustomCPUCFSQuotaPeriod"

	// owner: @derekwaynecarr
	// beta: v1.10
	// GA: v1.14
	//
	// Enable pods to consume pre-allocated huge pages of varying page sizes
	HugePages utilfeature.Feature = "HugePages"

	// owner: @sjenning
	// beta: v1.11
	//
	// Enable pods to set sysctls on a pod
	Sysctls utilfeature.Feature = "Sysctls"

	// owner @brendandburns
	// alpha: v1.9
	//
	// Enable nodes to exclude themselves from service load balancers
	ServiceNodeExclusion utilfeature.Feature = "ServiceNodeExclusion"

	// owner: @jsafrane
	// alpha: v1.9
	//
	// Enable running mount utilities in containers.
	MountContainers utilfeature.Feature = "MountContainers"

	// owner: @msau42
	// GA: v1.13
	//
	// Extend the default scheduler to be aware of PV topology and handle PV binding
	VolumeScheduling utilfeature.Feature = "VolumeScheduling"

	// owner: @vladimirvivien
	// GA: v1.13
	//
	// Enable mount/attachment of Container Storage Interface (CSI) backed PVs
	CSIPersistentVolume utilfeature.Feature = "CSIPersistentVolume"

	// owner: @saad-ali
	// alpha: v1.12
	// beta:  v1.14
	// Enable all logic related to the CSIDriver API object in storage.k8s.io
	CSIDriverRegistry utilfeature.Feature = "CSIDriverRegistry"

	// owner: @verult
	// alpha: v1.12
	// beta:  v1.14
	// Enable all logic related to the CSINode API object in storage.k8s.io
	CSINodeInfo utilfeature.Feature = "CSINodeInfo"

	// owner @MrHohn
	// GA: v1.14
	//
	// Support configurable pod DNS parameters.
	CustomPodDNS utilfeature.Feature = "CustomPodDNS"

	// owner: @screeley44
	// alpha: v1.9
	// beta: v1.13
	//
	// Enable Block volume support in containers.
	BlockVolume utilfeature.Feature = "BlockVolume"

	// owner: @pospispa
	// GA: v1.11
	//
	// Postpone deletion of a PV or a PVC when they are being used
	StorageObjectInUseProtection utilfeature.Feature = "StorageObjectInUseProtection"

	// owner: @aveshagarwal
	// alpha: v1.9
	//
	// Enable resource limits priority function
	ResourceLimitsPriorityFunction utilfeature.Feature = "ResourceLimitsPriorityFunction"

	// owner: @m1093782566
	// GA: v1.11
	//
	// Implement IPVS-based in-cluster service load balancing
	SupportIPVSProxyMode utilfeature.Feature = "SupportIPVSProxyMode"

	// owner: @dims, @derekwaynecarr
	// alpha: v1.10
	// beta: v1.14
	//
	// Implement support for limiting pids in pods
	SupportPodPidsLimit utilfeature.Feature = "SupportPodPidsLimit"

	// owner: @feiskyer
	// alpha: v1.10
	//
	// Enable Hyper-V containers on Windows
	HyperVContainer utilfeature.Feature = "HyperVContainer"

	// owner: @k82cn
	// beta: v1.12
	//
	// Schedule DaemonSet Pods by default scheduler instead of DaemonSet controller
	ScheduleDaemonSetPods utilfeature.Feature = "ScheduleDaemonSetPods"

	// owner: @mikedanese
	// beta: v1.12
	//
	// Implement TokenRequest endpoint on service account resources.
	TokenRequest utilfeature.Feature = "TokenRequest"

	// owner: @mikedanese
	// beta: v1.12
	//
	// Enable ServiceAccountTokenVolumeProjection support in ProjectedVolumes.
	TokenRequestProjection utilfeature.Feature = "TokenRequestProjection"

	// owner: @mikedanese
	// alpha: v1.13
	//
	// Migrate ServiceAccount volumes to use a projected volume consisting of a
	// ServiceAccountTokenVolumeProjection. This feature adds new required flags
	// to the API server.
	BoundServiceAccountTokenVolume utilfeature.Feature = "BoundServiceAccountTokenVolume"

	// owner: @Random-Liu
	// beta: v1.11
	//
	// Enable container log rotation for cri container runtime
	CRIContainerLogRotation utilfeature.Feature = "CRIContainerLogRotation"

	// owner: @krmayankk
	// beta: v1.14
	//
	// Enables control over the primary group ID of containers' init processes.
	RunAsGroup utilfeature.Feature = "RunAsGroup"

	// owner: @saad-ali
	// ga
	//
	// Allow mounting a subpath of a volume in a container
	// Do not remove this feature gate even though it's GA
	VolumeSubpath utilfeature.Feature = "VolumeSubpath"

	// owner: @gnufied
	// beta : v1.12
	//
	// Add support for volume plugins to report node specific
	// volume limits
	AttachVolumeLimit utilfeature.Feature = "AttachVolumeLimit"

	// owner: @ravig
	// alpha: v1.11
	//
	// Include volume count on node to be considered for balanced resource allocation while scheduling.
	// A node which has closer cpu,memory utilization and volume count is favoured by scheduler
	// while making decisions.
	BalanceAttachedNodeVolumes utilfeature.Feature = "BalanceAttachedNodeVolumes"

	// owner @freehan
	// GA: v1.14
	//
	// Allow user to specify additional conditions to be evaluated for Pod readiness.
	PodReadinessGates utilfeature.Feature = "PodReadinessGates"

	// owner: @kevtaylor
	// alpha: v1.11
	//
	// Allow subpath environment variable substitution
	// Only applicable if the VolumeSubpath feature is also enabled
	VolumeSubpathEnvExpansion utilfeature.Feature = "VolumeSubpathEnvExpansion"

	// owner: @vikaschoudhary16
	// GA: v1.13
	//
	//
	// Enable probe based plugin watcher utility for discovering Kubelet plugins
	KubeletPluginsWatcher utilfeature.Feature = "KubeletPluginsWatcher"

	// owner: @vikaschoudhary16
	// beta: v1.12
	//
	//
	// Enable resource quota scope selectors
	ResourceQuotaScopeSelectors utilfeature.Feature = "ResourceQuotaScopeSelectors"

	// owner: @vladimirvivien
	// alpha: v1.11
	// beta: v1.14
	//
	// Enables CSI to use raw block storage volumes
	CSIBlockVolume utilfeature.Feature = "CSIBlockVolume"

	// owner: @vladimirvivien
	// alpha: v1.14
	//
	// Enables CSI Inline volumes support for pods
	CSIInlineVolume utilfeature.Feature = "CSIInlineVolume"

	// owner: @tallclair
	// alpha: v1.12
	// beta:  v1.14
	//
	// Enables RuntimeClass, for selecting between multiple runtimes to run a pod.
	RuntimeClass utilfeature.Feature = "RuntimeClass"

	// owner: @mtaufen
	// alpha: v1.12
	//
	// Kubelet uses the new Lease API to report node heartbeats,
	// (Kube) Node Lifecycle Controller uses these heartbeats as a node health signal.
	NodeLease utilfeature.Feature = "NodeLease"

	// owner: @janosi
	// alpha: v1.12
	//
	// Enables SCTP as new protocol for Service ports, NetworkPolicy, and ContainerPort in Pod/Containers definition
	SCTPSupport utilfeature.Feature = "SCTPSupport"

	// owner: @xing-yang
	// alpha: v1.12
	//
	// Enable volume snapshot data source support.
	VolumeSnapshotDataSource utilfeature.Feature = "VolumeSnapshotDataSource"

	// owner: @jessfraz
	// alpha: v1.12
	//
	// Enables control over ProcMountType for containers.
	ProcMountType utilfeature.Feature = "ProcMountType"

	// owner: @janetkuo
	// alpha: v1.12
	//
	// Allow TTL controller to clean up Pods and Jobs after they finish.
	TTLAfterFinished utilfeature.Feature = "TTLAfterFinished"

	// owner: @dashpole
	// alpha: v1.13
	//
	// Enables the kubelet's pod resources grpc endpoint
	KubeletPodResources utilfeature.Feature = "KubeletPodResources"

	// owner: @davidz627
	// alpha: v1.14
	//
	// Enables the in-tree storage to CSI Plugin migration feature.
	CSIMigration utilfeature.Feature = "CSIMigration"

	// owner: @davidz627
	// alpha: v1.14
	//
	// Enables the GCE PD in-tree driver to GCE CSI Driver migration feature.
	CSIMigrationGCE utilfeature.Feature = "CSIMigrationGCE"

	// owner: @leakingtapan
	// alpha: v1.14
	//
	// Enables the AWS EBS in-tree driver to AWS EBS CSI Driver migration feature.
	CSIMigrationAWS utilfeature.Feature = "CSIMigrationAWS"

	// owner: @RobertKrawitz
	// alpha: v1.14
	//
	// Implement support for limiting pids in nodes
	SupportNodePidsLimit utilfeature.Feature = "SupportNodePidsLimit"

	// owner: @wk8
	// alpha: v1.14
	//
	// Enables GMSA support for Windows workloads.
	WindowsGMSA utilfeature.Feature = "WindowsGMSA"

	// owner: @adisky
	// alpha: v1.14
	//
	// Enables the OpenStack Cinder in-tree driver to OpenStack Cinder CSI Driver migration feature.
	CSIMigrationOpenStack utilfeature.Feature = "CSIMigrationOpenStack"
)

func init() {
	utilfeature.DefaultMutableFeatureGate.Add(defaultKubernetesFeatureGates)
}

// defaultKubernetesFeatureGates consists of all known Kubernetes-specific feature keys.
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
var defaultKubernetesFeatureGates = map[utilfeature.Feature]utilfeature.FeatureSpec{
	AppArmor:             {Default: true, PreRelease: utilfeature.Beta},
	DynamicKubeletConfig: {Default: true, PreRelease: utilfeature.Beta},
	ExperimentalHostUserNamespaceDefaultingGate: {Default: false, PreRelease: utilfeature.Beta},
	ExperimentalCriticalPodAnnotation:           {Default: false, PreRelease: utilfeature.Alpha},
	DevicePlugins:                               {Default: true, PreRelease: utilfeature.Beta},
	TaintBasedEvictions:                         {Default: true, PreRelease: utilfeature.Beta},
	RotateKubeletServerCertificate:              {Default: true, PreRelease: utilfeature.Beta},
	RotateKubeletClientCertificate:              {Default: true, PreRelease: utilfeature.Beta},
	PersistentLocalVolumes:                      {Default: true, PreRelease: utilfeature.GA, LockToDefault: true}, // remove in 1.17
	LocalStorageCapacityIsolation:               {Default: true, PreRelease: utilfeature.Beta},
	HugePages:                                   {Default: true, PreRelease: utilfeature.GA, LockToDefault: true}, // remove in 1.16
	Sysctls:                                     {Default: true, PreRelease: utilfeature.Beta},
	DebugContainers:                             {Default: false, PreRelease: utilfeature.Alpha},
	PodShareProcessNamespace:                    {Default: true, PreRelease: utilfeature.Beta},
	PodPriority:                                 {Default: true, PreRelease: utilfeature.GA},
	TaintNodesByCondition:                       {Default: true, PreRelease: utilfeature.Beta},
	QOSReserved:                                 {Default: false, PreRelease: utilfeature.Alpha},
	ExpandPersistentVolumes:                     {Default: true, PreRelease: utilfeature.Beta},
	ExpandInUsePersistentVolumes:                {Default: false, PreRelease: utilfeature.Alpha},
	ExpandCSIVolumes:                            {Default: false, PreRelease: utilfeature.Alpha},
	AttachVolumeLimit:                           {Default: true, PreRelease: utilfeature.Beta},
	CPUManager:                                  {Default: true, PreRelease: utilfeature.Beta},
	CPUCFSQuotaPeriod:                           {Default: false, PreRelease: utilfeature.Alpha},
	ServiceNodeExclusion:                        {Default: false, PreRelease: utilfeature.Alpha},
	MountContainers:                             {Default: false, PreRelease: utilfeature.Alpha},
	VolumeScheduling:                            {Default: true, PreRelease: utilfeature.GA, LockToDefault: true}, // remove in 1.16
	CSIPersistentVolume:                         {Default: true, PreRelease: utilfeature.GA, LockToDefault: true}, // remove in 1.16
	CSIDriverRegistry:                           {Default: true, PreRelease: utilfeature.Beta},
	CSINodeInfo:                                 {Default: true, PreRelease: utilfeature.Beta},
	CustomPodDNS:                                {Default: true, PreRelease: utilfeature.GA, LockToDefault: true}, // remove in 1.16
	BlockVolume:                                 {Default: true, PreRelease: utilfeature.Beta},
	StorageObjectInUseProtection:                {Default: true, PreRelease: utilfeature.GA},
	ResourceLimitsPriorityFunction:              {Default: false, PreRelease: utilfeature.Alpha},
	SupportIPVSProxyMode:                        {Default: true, PreRelease: utilfeature.GA},
	SupportPodPidsLimit:                         {Default: true, PreRelease: utilfeature.Beta},
	SupportNodePidsLimit:                        {Default: false, PreRelease: utilfeature.Alpha},
	HyperVContainer:                             {Default: false, PreRelease: utilfeature.Alpha},
	ScheduleDaemonSetPods:                       {Default: true, PreRelease: utilfeature.Beta},
	TokenRequest:                                {Default: true, PreRelease: utilfeature.Beta},
	TokenRequestProjection:                      {Default: true, PreRelease: utilfeature.Beta},
	BoundServiceAccountTokenVolume:              {Default: false, PreRelease: utilfeature.Alpha},
	CRIContainerLogRotation:                     {Default: true, PreRelease: utilfeature.Beta},
	cloudfeatures.GCERegionalPersistentDisk:     {Default: true, PreRelease: utilfeature.GA},
	CSIMigration:                                {Default: false, PreRelease: utilfeature.Alpha},
	CSIMigrationGCE:                             {Default: false, PreRelease: utilfeature.Alpha},
	CSIMigrationAWS:                             {Default: false, PreRelease: utilfeature.Alpha},
	RunAsGroup:                                  {Default: true, PreRelease: utilfeature.Beta},
	CSIMigrationOpenStack:                       {Default: false, PreRelease: utilfeature.Alpha},
	VolumeSubpath:                               {Default: true, PreRelease: utilfeature.GA},
	BalanceAttachedNodeVolumes:                  {Default: false, PreRelease: utilfeature.Alpha},
	PodReadinessGates:                           {Default: true, PreRelease: utilfeature.GA, LockToDefault: true}, // remove in 1.16
	VolumeSubpathEnvExpansion:                   {Default: false, PreRelease: utilfeature.Alpha},
	KubeletPluginsWatcher:                       {Default: true, PreRelease: utilfeature.GA, LockToDefault: true}, // remove in 1.16
	ResourceQuotaScopeSelectors:                 {Default: true, PreRelease: utilfeature.Beta},
	CSIBlockVolume:                              {Default: true, PreRelease: utilfeature.Beta},
	CSIInlineVolume:                             {Default: false, PreRelease: utilfeature.Alpha},
	RuntimeClass:                                {Default: true, PreRelease: utilfeature.Beta},
	NodeLease:                                   {Default: true, PreRelease: utilfeature.Beta},
	SCTPSupport:                                 {Default: false, PreRelease: utilfeature.Alpha},
	VolumeSnapshotDataSource:                    {Default: false, PreRelease: utilfeature.Alpha},
	ProcMountType:                               {Default: false, PreRelease: utilfeature.Alpha},
	TTLAfterFinished:                            {Default: false, PreRelease: utilfeature.Alpha},
	KubeletPodResources:                         {Default: false, PreRelease: utilfeature.Alpha},
	WindowsGMSA:                                 {Default: false, PreRelease: utilfeature.Alpha},

	// inherited features from generic apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:
	genericfeatures.StreamingProxyRedirects: {Default: true, PreRelease: utilfeature.Beta},
	genericfeatures.ValidateProxyRedirects:  {Default: true, PreRelease: utilfeature.Beta},
	genericfeatures.AdvancedAuditing:        {Default: true, PreRelease: utilfeature.GA},
	genericfeatures.DynamicAuditing:         {Default: false, PreRelease: utilfeature.Alpha},
	genericfeatures.APIResponseCompression:  {Default: false, PreRelease: utilfeature.Alpha},
	genericfeatures.APIListChunking:         {Default: true, PreRelease: utilfeature.Beta},
	genericfeatures.DryRun:                  {Default: true, PreRelease: utilfeature.Beta},
	genericfeatures.ServerSideApply:         {Default: false, PreRelease: utilfeature.Alpha},

	// inherited features from apiextensions-apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:
	apiextensionsfeatures.CustomResourceValidation:        {Default: true, PreRelease: utilfeature.Beta},
	apiextensionsfeatures.CustomResourceSubresources:      {Default: true, PreRelease: utilfeature.Beta},
	apiextensionsfeatures.CustomResourceWebhookConversion: {Default: false, PreRelease: utilfeature.Alpha},
	apiextensionsfeatures.CustomResourcePublishOpenAPI:    {Default: false, PreRelease: utilfeature.Alpha},

	// features that enable backwards compatibility but are scheduled to be removed
	// ...
}
