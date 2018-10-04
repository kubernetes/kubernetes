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
	// Ensures guaranteed scheduling of pods marked with a special pod annotation `scheduler.alpha.kubernetes.io/critical-pod`
	// and also prevents them from being evicted from a node.
	// Note: This feature is not supported for `BestEffort` pods.
	ExperimentalCriticalPodAnnotation utilfeature.Feature = "ExperimentalCriticalPodAnnotation"

	// owner: @jiayingz
	// beta: v1.10
	//
	// Enables support for Device Plugins
	DevicePlugins utilfeature.Feature = "DevicePlugins"

	// owner: @gmarek
	// alpha: v1.6
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
	//
	// Add priority to pods. Priority affects scheduling and preemption of pods.
	PodPriority utilfeature.Feature = "PodPriority"

	// owner: @resouer
	// alpha: v1.8
	//
	// Enable equivalence class cache for scheduler.
	EnableEquivalenceClassCache utilfeature.Feature = "EnableEquivalenceClassCache"

	// owner: @k82cn
	// beta: v1.12
	//
	// Taint nodes based on their condition status for 'NetworkUnavailable',
	// 'MemoryPressure', 'OutOfDisk' and 'DiskPressure'.
	TaintNodesByCondition utilfeature.Feature = "TaintNodesByCondition"

	// owner: @jsafrane
	// GA: v1.12
	//
	// Enable mount propagation of volumes.
	MountPropagation utilfeature.Feature = "MountPropagation"

	// owner: @sjenning
	// alpha: v1.11
	//
	// Allows resource reservations at the QoS level preventing pods at lower QoS levels from
	// bursting into resources requested at higher QoS levels (memory only for now)
	QOSReserved utilfeature.Feature = "QOSReserved"

	// owner: @ConnorDoyle
	// alpha: v1.8
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
	// alpha: v1.9
	//
	// Extend the default scheduler to be aware of PV topology and handle PV binding
	// Before moving to beta, resolve Kubernetes issue #56180
	VolumeScheduling utilfeature.Feature = "VolumeScheduling"

	// owner: @vladimirvivien
	// beta: v1.10
	//
	// Enable mount/attachment of Container Storage Interface (CSI) backed PVs
	CSIPersistentVolume utilfeature.Feature = "CSIPersistentVolume"

	// owner: @saad-ali
	// alpha: v1.12
	// Enable all logic related to the CSIDriver API object in csi.storage.k8s.io
	CSIDriverRegistry utilfeature.Feature = "CSIDriverRegistry"

	// owner: @verult
	// alpha: v1.12
	// Enable all logic related to the CSINodeInfo API object in csi.storage.k8s.io
	CSINodeInfo utilfeature.Feature = "CSINodeInfo"

	// owner @MrHohn
	// beta: v1.10
	//
	// Support configurable pod DNS parameters.
	CustomPodDNS utilfeature.Feature = "CustomPodDNS"

	// owner: @screeley44
	// alpha: v1.9
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

	// owner: @dims
	// alpha: v1.10
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

	// owner: @Random-Liu
	// beta: v1.11
	//
	// Enable container log rotation for cri container runtime
	CRIContainerLogRotation utilfeature.Feature = "CRIContainerLogRotation"

	// owner: @verult
	// beta: v1.10
	//
	// Enables the regional PD feature on GCE.
	GCERegionalPersistentDisk utilfeature.Feature = "GCERegionalPersistentDisk"

	// owner: @krmayankk
	// alpha: v1.10
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
	// beta: v1.11
	//
	// Support Pod Ready++
	PodReadinessGates utilfeature.Feature = "PodReadinessGates"

	// owner: @kevtaylor
	// alpha: v1.11
	//
	// Allow subpath environment variable substitution
	// Only applicable if the VolumeSubpath feature is also enabled
	VolumeSubpathEnvExpansion utilfeature.Feature = "VolumeSubpathEnvExpansion"

	// owner: @vikaschoudhary16
	// alpha: v1.11
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
	//
	// Enables CSI to use raw block storage volumes
	CSIBlockVolume utilfeature.Feature = "CSIBlockVolume"

	// owner: @tallclair
	// alpha: v1.12
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
)

func init() {
	utilfeature.DefaultFeatureGate.Add(defaultKubernetesFeatureGates)
}

// defaultKubernetesFeatureGates consists of all known Kubernetes-specific feature keys.
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
var defaultKubernetesFeatureGates = map[utilfeature.Feature]utilfeature.FeatureSpec{
	AppArmor:                                    {Default: true, PreRelease: utilfeature.Beta},
	DynamicKubeletConfig:                        {Default: true, PreRelease: utilfeature.Beta},
	ExperimentalHostUserNamespaceDefaultingGate: {Default: false, PreRelease: utilfeature.Beta},
	ExperimentalCriticalPodAnnotation:           {Default: false, PreRelease: utilfeature.Alpha},
	DevicePlugins:                               {Default: true, PreRelease: utilfeature.Beta},
	TaintBasedEvictions:                         {Default: false, PreRelease: utilfeature.Alpha},
	RotateKubeletServerCertificate:              {Default: true, PreRelease: utilfeature.Beta},
	RotateKubeletClientCertificate:              {Default: true, PreRelease: utilfeature.Beta},
	PersistentLocalVolumes:                      {Default: true, PreRelease: utilfeature.Beta},
	LocalStorageCapacityIsolation:               {Default: true, PreRelease: utilfeature.Beta},
	HugePages:                                   {Default: true, PreRelease: utilfeature.Beta},
	Sysctls:                                     {Default: true, PreRelease: utilfeature.Beta},
	DebugContainers:                             {Default: false, PreRelease: utilfeature.Alpha},
	PodShareProcessNamespace:                    {Default: true, PreRelease: utilfeature.Beta},
	PodPriority:                                 {Default: true, PreRelease: utilfeature.Beta},
	EnableEquivalenceClassCache:                 {Default: false, PreRelease: utilfeature.Alpha},
	TaintNodesByCondition:                       {Default: true, PreRelease: utilfeature.Beta},
	MountPropagation:                            {Default: true, PreRelease: utilfeature.GA},
	QOSReserved:                                 {Default: false, PreRelease: utilfeature.Alpha},
	ExpandPersistentVolumes:                     {Default: true, PreRelease: utilfeature.Beta},
	ExpandInUsePersistentVolumes:                {Default: false, PreRelease: utilfeature.Alpha},
	AttachVolumeLimit:                           {Default: true, PreRelease: utilfeature.Beta},
	CPUManager:                                  {Default: true, PreRelease: utilfeature.Beta},
	CPUCFSQuotaPeriod:                           {Default: false, PreRelease: utilfeature.Alpha},
	ServiceNodeExclusion:                        {Default: false, PreRelease: utilfeature.Alpha},
	MountContainers:                             {Default: false, PreRelease: utilfeature.Alpha},
	VolumeScheduling:                            {Default: true, PreRelease: utilfeature.Beta},
	CSIPersistentVolume:                         {Default: true, PreRelease: utilfeature.Beta},
	CSIDriverRegistry:                           {Default: false, PreRelease: utilfeature.Alpha},
	CSINodeInfo:                                 {Default: false, PreRelease: utilfeature.Alpha},
	CustomPodDNS:                                {Default: true, PreRelease: utilfeature.Beta},
	BlockVolume:                                 {Default: false, PreRelease: utilfeature.Alpha},
	StorageObjectInUseProtection:                {Default: true, PreRelease: utilfeature.GA},
	ResourceLimitsPriorityFunction:              {Default: false, PreRelease: utilfeature.Alpha},
	SupportIPVSProxyMode:                        {Default: true, PreRelease: utilfeature.GA},
	SupportPodPidsLimit:                         {Default: false, PreRelease: utilfeature.Alpha},
	HyperVContainer:                             {Default: false, PreRelease: utilfeature.Alpha},
	ScheduleDaemonSetPods:                       {Default: true, PreRelease: utilfeature.Beta},
	TokenRequest:                                {Default: true, PreRelease: utilfeature.Beta},
	TokenRequestProjection:                      {Default: true, PreRelease: utilfeature.Beta},
	CRIContainerLogRotation:                     {Default: true, PreRelease: utilfeature.Beta},
	GCERegionalPersistentDisk:                   {Default: true, PreRelease: utilfeature.Beta},
	RunAsGroup:                                  {Default: false, PreRelease: utilfeature.Alpha},
	VolumeSubpath:                               {Default: true, PreRelease: utilfeature.GA},
	BalanceAttachedNodeVolumes:                  {Default: false, PreRelease: utilfeature.Alpha},
	PodReadinessGates:                           {Default: true, PreRelease: utilfeature.Beta},
	VolumeSubpathEnvExpansion:                   {Default: false, PreRelease: utilfeature.Alpha},
	KubeletPluginsWatcher:                       {Default: true, PreRelease: utilfeature.Beta},
	ResourceQuotaScopeSelectors:                 {Default: true, PreRelease: utilfeature.Beta},
	CSIBlockVolume:                              {Default: false, PreRelease: utilfeature.Alpha},
	RuntimeClass:                                {Default: false, PreRelease: utilfeature.Alpha},
	NodeLease:                                   {Default: false, PreRelease: utilfeature.Alpha},
	SCTPSupport:                                 {Default: false, PreRelease: utilfeature.Alpha},
	VolumeSnapshotDataSource:                    {Default: false, PreRelease: utilfeature.Alpha},
	ProcMountType:                               {Default: false, PreRelease: utilfeature.Alpha},
	TTLAfterFinished:                            {Default: false, PreRelease: utilfeature.Alpha},

	// inherited features from generic apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:
	genericfeatures.StreamingProxyRedirects: {Default: true, PreRelease: utilfeature.Beta},
	genericfeatures.AdvancedAuditing:        {Default: true, PreRelease: utilfeature.GA},
	genericfeatures.APIResponseCompression:  {Default: false, PreRelease: utilfeature.Alpha},
	genericfeatures.Initializers:            {Default: false, PreRelease: utilfeature.Alpha},
	genericfeatures.APIListChunking:         {Default: true, PreRelease: utilfeature.Beta},
	genericfeatures.DryRun:                  {Default: false, PreRelease: utilfeature.Alpha},

	// inherited features from apiextensions-apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:
	apiextensionsfeatures.CustomResourceValidation:   {Default: true, PreRelease: utilfeature.Beta},
	apiextensionsfeatures.CustomResourceSubresources: {Default: true, PreRelease: utilfeature.Beta},

	// features that enable backwards compatibility but are scheduled to be removed
	// ...
}
