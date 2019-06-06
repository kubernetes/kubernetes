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
	"k8s.io/apimachinery/pkg/util/runtime"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
)

const (
	// Every feature gate should add method here following this template:
	//
	// // owner: @username
	// // alpha: v1.X
	// MyFeature featuregate.Feature = "MyFeature"

	// owner: @tallclair
	// beta: v1.4
	AppArmor featuregate.Feature = "AppArmor"

	// owner: @mtaufen
	// alpha: v1.4
	// beta: v1.11
	DynamicKubeletConfig featuregate.Feature = "DynamicKubeletConfig"

	// owner: @pweil-
	// alpha: v1.5
	//
	// Default userns=host for containers that are using other host namespaces, host mounts, the pod
	// contains a privileged container, or specific non-namespaced capabilities (MKNOD, SYS_MODULE,
	// SYS_TIME). This should only be enabled if user namespace remapping is enabled in the docker daemon.
	ExperimentalHostUserNamespaceDefaultingGate featuregate.Feature = "ExperimentalHostUserNamespaceDefaulting"

	// owner: @vishh
	// alpha: v1.5
	//
	// DEPRECATED - This feature is deprecated by Pod Priority and Preemption as of Kubernetes 1.13.
	// Ensures guaranteed scheduling of pods marked with a special pod annotation `scheduler.alpha.kubernetes.io/critical-pod`
	// and also prevents them from being evicted from a node.
	// Note: This feature is not supported for `BestEffort` pods.
	ExperimentalCriticalPodAnnotation featuregate.Feature = "ExperimentalCriticalPodAnnotation"

	// owner: @jiayingz
	// beta: v1.10
	//
	// Enables support for Device Plugins
	DevicePlugins featuregate.Feature = "DevicePlugins"

	// owner: @Huang-Wei
	// beta: v1.13
	//
	// Changes the logic behind evicting Pods from not ready Nodes
	// to take advantage of NoExecute Taints and Tolerations.
	TaintBasedEvictions featuregate.Feature = "TaintBasedEvictions"

	// owner: @mikedanese
	// alpha: v1.7
	// beta: v1.12
	//
	// Gets a server certificate for the kubelet from the Certificate Signing
	// Request API instead of generating one self signed and auto rotates the
	// certificate as expiration approaches.
	RotateKubeletServerCertificate featuregate.Feature = "RotateKubeletServerCertificate"

	// owner: @mikedanese
	// beta: v1.8
	//
	// Automatically renews the client certificate used for communicating with
	// the API server as the certificate approaches expiration.
	RotateKubeletClientCertificate featuregate.Feature = "RotateKubeletClientCertificate"

	// owner: @msau42
	// alpha: v1.7
	// beta: v1.10
	// ga: v1.14
	//
	// A new volume type that supports local disks on a node.
	PersistentLocalVolumes featuregate.Feature = "PersistentLocalVolumes"

	// owner: @jinxu
	// beta: v1.10
	//
	// New local storage types to support local storage capacity isolation
	LocalStorageCapacityIsolation featuregate.Feature = "LocalStorageCapacityIsolation"

	// owner: @gnufied
	// beta: v1.11
	// Ability to Expand persistent volumes
	ExpandPersistentVolumes featuregate.Feature = "ExpandPersistentVolumes"

	// owner: @mlmhl
	// beta: v1.15
	// Ability to expand persistent volumes' file system without unmounting volumes.
	ExpandInUsePersistentVolumes featuregate.Feature = "ExpandInUsePersistentVolumes"

	// owner: @gnufied
	// alpha: v1.14
	// Ability to expand CSI volumes
	ExpandCSIVolumes featuregate.Feature = "ExpandCSIVolumes"

	// owner: @verb
	// alpha: v1.10
	//
	// Allows running a "debug container" in a pod namespaces to troubleshoot a running pod.
	DebugContainers featuregate.Feature = "DebugContainers"

	// owner: @verb
	// beta: v1.12
	//
	// Allows all containers in a pod to share a process namespace.
	PodShareProcessNamespace featuregate.Feature = "PodShareProcessNamespace"

	// owner: @bsalamat
	// alpha: v1.8
	// beta: v1.11
	// GA: v1.14
	//
	// Add priority to pods. Priority affects scheduling and preemption of pods.
	PodPriority featuregate.Feature = "PodPriority"

	// owner: @k82cn
	// beta: v1.12
	//
	// Taint nodes based on their condition status for 'NetworkUnavailable',
	// 'MemoryPressure', 'PIDPressure' and 'DiskPressure'.
	TaintNodesByCondition featuregate.Feature = "TaintNodesByCondition"

	// owner: @sjenning
	// alpha: v1.11
	//
	// Allows resource reservations at the QoS level preventing pods at lower QoS levels from
	// bursting into resources requested at higher QoS levels (memory only for now)
	QOSReserved featuregate.Feature = "QOSReserved"

	// owner: @ConnorDoyle
	// alpha: v1.8
	// beta: v1.10
	//
	// Alternative container-level CPU affinity policies.
	CPUManager featuregate.Feature = "CPUManager"

	// owner: @szuecs
	// alpha: v1.12
	//
	// Enable nodes to change CPUCFSQuotaPeriod
	CPUCFSQuotaPeriod featuregate.Feature = "CustomCPUCFSQuotaPeriod"

	// owner: @derekwaynecarr
	// beta: v1.10
	// GA: v1.14
	//
	// Enable pods to consume pre-allocated huge pages of varying page sizes
	HugePages featuregate.Feature = "HugePages"

	// owner: @sjenning
	// beta: v1.11
	//
	// Enable pods to set sysctls on a pod
	Sysctls featuregate.Feature = "Sysctls"

	// owner @brendandburns
	// alpha: v1.9
	//
	// Enable nodes to exclude themselves from service load balancers
	ServiceNodeExclusion featuregate.Feature = "ServiceNodeExclusion"

	// owner: @jsafrane
	// alpha: v1.9
	//
	// Enable running mount utilities in containers.
	MountContainers featuregate.Feature = "MountContainers"

	// owner: @msau42
	// GA: v1.13
	//
	// Extend the default scheduler to be aware of PV topology and handle PV binding
	VolumeScheduling featuregate.Feature = "VolumeScheduling"

	// owner: @vladimirvivien
	// GA: v1.13
	//
	// Enable mount/attachment of Container Storage Interface (CSI) backed PVs
	CSIPersistentVolume featuregate.Feature = "CSIPersistentVolume"

	// owner: @saad-ali
	// alpha: v1.12
	// beta:  v1.14
	// Enable all logic related to the CSIDriver API object in storage.k8s.io
	CSIDriverRegistry featuregate.Feature = "CSIDriverRegistry"

	// owner: @verult
	// alpha: v1.12
	// beta:  v1.14
	// Enable all logic related to the CSINode API object in storage.k8s.io
	CSINodeInfo featuregate.Feature = "CSINodeInfo"

	// owner @MrHohn
	// GA: v1.14
	//
	// Support configurable pod DNS parameters.
	CustomPodDNS featuregate.Feature = "CustomPodDNS"

	// owner: @screeley44
	// alpha: v1.9
	// beta: v1.13
	//
	// Enable Block volume support in containers.
	BlockVolume featuregate.Feature = "BlockVolume"

	// owner: @pospispa
	// GA: v1.11
	//
	// Postpone deletion of a PV or a PVC when they are being used
	StorageObjectInUseProtection featuregate.Feature = "StorageObjectInUseProtection"

	// owner: @aveshagarwal
	// alpha: v1.9
	//
	// Enable resource limits priority function
	ResourceLimitsPriorityFunction featuregate.Feature = "ResourceLimitsPriorityFunction"

	// owner: @m1093782566
	// GA: v1.11
	//
	// Implement IPVS-based in-cluster service load balancing
	SupportIPVSProxyMode featuregate.Feature = "SupportIPVSProxyMode"

	// owner: @dims, @derekwaynecarr
	// alpha: v1.10
	// beta: v1.14
	//
	// Implement support for limiting pids in pods
	SupportPodPidsLimit featuregate.Feature = "SupportPodPidsLimit"

	// owner: @feiskyer
	// alpha: v1.10
	//
	// Enable Hyper-V containers on Windows
	HyperVContainer featuregate.Feature = "HyperVContainer"

	// owner: @k82cn
	// beta: v1.12
	//
	// Schedule DaemonSet Pods by default scheduler instead of DaemonSet controller
	ScheduleDaemonSetPods featuregate.Feature = "ScheduleDaemonSetPods"

	// owner: @mikedanese
	// beta: v1.12
	//
	// Implement TokenRequest endpoint on service account resources.
	TokenRequest featuregate.Feature = "TokenRequest"

	// owner: @mikedanese
	// beta: v1.12
	//
	// Enable ServiceAccountTokenVolumeProjection support in ProjectedVolumes.
	TokenRequestProjection featuregate.Feature = "TokenRequestProjection"

	// owner: @mikedanese
	// alpha: v1.13
	//
	// Migrate ServiceAccount volumes to use a projected volume consisting of a
	// ServiceAccountTokenVolumeProjection. This feature adds new required flags
	// to the API server.
	BoundServiceAccountTokenVolume featuregate.Feature = "BoundServiceAccountTokenVolume"

	// owner: @Random-Liu
	// beta: v1.11
	//
	// Enable container log rotation for cri container runtime
	CRIContainerLogRotation featuregate.Feature = "CRIContainerLogRotation"

	// owner: @krmayankk
	// beta: v1.14
	//
	// Enables control over the primary group ID of containers' init processes.
	RunAsGroup featuregate.Feature = "RunAsGroup"

	// owner: @saad-ali
	// ga
	//
	// Allow mounting a subpath of a volume in a container
	// Do not remove this feature gate even though it's GA
	VolumeSubpath featuregate.Feature = "VolumeSubpath"

	// owner: @gnufied
	// beta : v1.12
	//
	// Add support for volume plugins to report node specific
	// volume limits
	AttachVolumeLimit featuregate.Feature = "AttachVolumeLimit"

	// owner: @ravig
	// alpha: v1.11
	//
	// Include volume count on node to be considered for balanced resource allocation while scheduling.
	// A node which has closer cpu,memory utilization and volume count is favoured by scheduler
	// while making decisions.
	BalanceAttachedNodeVolumes featuregate.Feature = "BalanceAttachedNodeVolumes"

	// owner @freehan
	// GA: v1.14
	//
	// Allow user to specify additional conditions to be evaluated for Pod readiness.
	PodReadinessGates featuregate.Feature = "PodReadinessGates"

	// owner: @kevtaylor
	// beta: v1.15
	//
	// Allow subpath environment variable substitution
	// Only applicable if the VolumeSubpath feature is also enabled
	VolumeSubpathEnvExpansion featuregate.Feature = "VolumeSubpathEnvExpansion"

	// owner: @vikaschoudhary16
	// GA: v1.13
	//
	//
	// Enable probe based plugin watcher utility for discovering Kubelet plugins
	KubeletPluginsWatcher featuregate.Feature = "KubeletPluginsWatcher"

	// owner: @vikaschoudhary16
	// beta: v1.12
	//
	//
	// Enable resource quota scope selectors
	ResourceQuotaScopeSelectors featuregate.Feature = "ResourceQuotaScopeSelectors"

	// owner: @vladimirvivien
	// alpha: v1.11
	// beta: v1.14
	//
	// Enables CSI to use raw block storage volumes
	CSIBlockVolume featuregate.Feature = "CSIBlockVolume"

	// owner: @vladimirvivien
	// alpha: v1.14
	//
	// Enables CSI Inline volumes support for pods
	CSIInlineVolume featuregate.Feature = "CSIInlineVolume"

	// owner: @tallclair
	// alpha: v1.12
	// beta:  v1.14
	//
	// Enables RuntimeClass, for selecting between multiple runtimes to run a pod.
	RuntimeClass featuregate.Feature = "RuntimeClass"

	// owner: @mtaufen
	// alpha: v1.12
	// beta:  v1.14
	//
	// Kubelet uses the new Lease API to report node heartbeats,
	// (Kube) Node Lifecycle Controller uses these heartbeats as a node health signal.
	NodeLease featuregate.Feature = "NodeLease"

	// owner: @janosi
	// alpha: v1.12
	//
	// Enables SCTP as new protocol for Service ports, NetworkPolicy, and ContainerPort in Pod/Containers definition
	SCTPSupport featuregate.Feature = "SCTPSupport"

	// owner: @xing-yang
	// alpha: v1.12
	//
	// Enable volume snapshot data source support.
	VolumeSnapshotDataSource featuregate.Feature = "VolumeSnapshotDataSource"

	// owner: @jessfraz
	// alpha: v1.12
	//
	// Enables control over ProcMountType for containers.
	ProcMountType featuregate.Feature = "ProcMountType"

	// owner: @janetkuo
	// alpha: v1.12
	//
	// Allow TTL controller to clean up Pods and Jobs after they finish.
	TTLAfterFinished featuregate.Feature = "TTLAfterFinished"

	// owner: @dashpole
	// alpha: v1.13
	// beta: v1.15
	//
	// Enables the kubelet's pod resources grpc endpoint
	KubeletPodResources featuregate.Feature = "KubeletPodResources"

	// owner: @davidz627
	// alpha: v1.14
	//
	// Enables the in-tree storage to CSI Plugin migration feature.
	CSIMigration featuregate.Feature = "CSIMigration"

	// owner: @davidz627
	// alpha: v1.14
	//
	// Enables the GCE PD in-tree driver to GCE CSI Driver migration feature.
	CSIMigrationGCE featuregate.Feature = "CSIMigrationGCE"

	// owner: @leakingtapan
	// alpha: v1.14
	//
	// Enables the AWS EBS in-tree driver to AWS EBS CSI Driver migration feature.
	CSIMigrationAWS featuregate.Feature = "CSIMigrationAWS"

	// owner: @andyzhangx
	// alpha: v1.15
	//
	// Enables the Azure Disk in-tree driver to Azure Disk Driver migration feature.
	CSIMigrationAzureDisk featuregate.Feature = "CSIMigrationAzureDisk"

	// owner: @andyzhangx
	// alpha: v1.15
	//
	// Enables the Azure File in-tree driver to Azure File Driver migration feature.
	CSIMigrationAzureFile featuregate.Feature = "CSIMigrationAzureFile"

	// owner: @RobertKrawitz
	// beta: v1.15
	//
	// Implement support for limiting pids in nodes
	SupportNodePidsLimit featuregate.Feature = "SupportNodePidsLimit"

	// owner: @wk8
	// alpha: v1.14
	//
	// Enables GMSA support for Windows workloads.
	WindowsGMSA featuregate.Feature = "WindowsGMSA"

	// owner: @adisky
	// alpha: v1.14
	//
	// Enables the OpenStack Cinder in-tree driver to OpenStack Cinder CSI Driver migration feature.
	CSIMigrationOpenStack featuregate.Feature = "CSIMigrationOpenStack"

	// owner: @verult
	// GA: v1.13
	//
	// Enables the regional PD feature on GCE.
	deprecatedGCERegionalPersistentDisk featuregate.Feature = "GCERegionalPersistentDisk"

	// owner: @MrHohn
	// alpha: v1.15
	//
	// Enables Finalizer Protection for Service LoadBalancers.
	ServiceLoadBalancerFinalizer featuregate.Feature = "ServiceLoadBalancerFinalizer"

	// owner: @RobertKrawitz
	// alpha: v1.15
	//
	// Allow use of filesystems for ephemeral storage monitoring.
	// Only applies if LocalStorageCapacityIsolation is set.
	LocalStorageCapacityIsolationFSQuotaMonitoring featuregate.Feature = "LocalStorageCapacityIsolationFSQuotaMonitoring"

	// owner: @denkensk
	// alpha: v1.15
	//
	// Enables NonPreempting option for priorityClass and pod.
	NonPreemptingPriority featuregate.Feature = "NonPreemptingPriority"

	// owner: @j-griffith
	// alpha: v1.15
	//
	// Enable support for specifying an existing PVC as a DataSource
	VolumePVCDataSource featuregate.Feature = "VolumePVCDataSource"
)

func init() {
	runtime.Must(utilfeature.DefaultMutableFeatureGate.Add(defaultKubernetesFeatureGates))
}

// defaultKubernetesFeatureGates consists of all known Kubernetes-specific feature keys.
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
var defaultKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	AppArmor:             {Default: true, PreRelease: featuregate.Beta},
	DynamicKubeletConfig: {Default: true, PreRelease: featuregate.Beta},
	ExperimentalHostUserNamespaceDefaultingGate: {Default: false, PreRelease: featuregate.Beta},
	ExperimentalCriticalPodAnnotation:           {Default: false, PreRelease: featuregate.Alpha},
	DevicePlugins:                               {Default: true, PreRelease: featuregate.Beta},
	TaintBasedEvictions:                         {Default: true, PreRelease: featuregate.Beta},
	RotateKubeletServerCertificate:              {Default: true, PreRelease: featuregate.Beta},
	RotateKubeletClientCertificate:              {Default: true, PreRelease: featuregate.Beta},
	PersistentLocalVolumes:                      {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.17
	LocalStorageCapacityIsolation:               {Default: true, PreRelease: featuregate.Beta},
	HugePages:                                   {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.16
	Sysctls:                                     {Default: true, PreRelease: featuregate.Beta},
	DebugContainers:                             {Default: false, PreRelease: featuregate.Alpha},
	PodShareProcessNamespace:                    {Default: true, PreRelease: featuregate.Beta},
	PodPriority:                                 {Default: true, PreRelease: featuregate.GA},
	TaintNodesByCondition:                       {Default: true, PreRelease: featuregate.Beta},
	QOSReserved:                                 {Default: false, PreRelease: featuregate.Alpha},
	ExpandPersistentVolumes:                     {Default: true, PreRelease: featuregate.Beta},
	ExpandInUsePersistentVolumes:                {Default: true, PreRelease: featuregate.Beta},
	ExpandCSIVolumes:                            {Default: false, PreRelease: featuregate.Alpha},
	AttachVolumeLimit:                           {Default: true, PreRelease: featuregate.Beta},
	CPUManager:                                  {Default: true, PreRelease: featuregate.Beta},
	CPUCFSQuotaPeriod:                           {Default: false, PreRelease: featuregate.Alpha},
	ServiceNodeExclusion:                        {Default: false, PreRelease: featuregate.Alpha},
	MountContainers:                             {Default: false, PreRelease: featuregate.Alpha},
	VolumeScheduling:                            {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.16
	CSIPersistentVolume:                         {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.16
	CSIDriverRegistry:                           {Default: true, PreRelease: featuregate.Beta},
	CSINodeInfo:                                 {Default: true, PreRelease: featuregate.Beta},
	CustomPodDNS:                                {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.16
	BlockVolume:                                 {Default: true, PreRelease: featuregate.Beta},
	StorageObjectInUseProtection:                {Default: true, PreRelease: featuregate.GA},
	ResourceLimitsPriorityFunction:              {Default: false, PreRelease: featuregate.Alpha},
	SupportIPVSProxyMode:                        {Default: true, PreRelease: featuregate.GA},
	SupportPodPidsLimit:                         {Default: true, PreRelease: featuregate.Beta},
	SupportNodePidsLimit:                        {Default: true, PreRelease: featuregate.Beta},
	HyperVContainer:                             {Default: false, PreRelease: featuregate.Alpha},
	ScheduleDaemonSetPods:                       {Default: true, PreRelease: featuregate.Beta},
	TokenRequest:                                {Default: true, PreRelease: featuregate.Beta},
	TokenRequestProjection:                      {Default: true, PreRelease: featuregate.Beta},
	BoundServiceAccountTokenVolume:              {Default: false, PreRelease: featuregate.Alpha},
	CRIContainerLogRotation:                     {Default: true, PreRelease: featuregate.Beta},
	deprecatedGCERegionalPersistentDisk:         {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.17
	CSIMigration:                                {Default: false, PreRelease: featuregate.Alpha},
	CSIMigrationGCE:                             {Default: false, PreRelease: featuregate.Alpha},
	CSIMigrationAWS:                             {Default: false, PreRelease: featuregate.Alpha},
	CSIMigrationAzureDisk:                       {Default: false, PreRelease: featuregate.Alpha},
	CSIMigrationAzureFile:                       {Default: false, PreRelease: featuregate.Alpha},
	RunAsGroup:                                  {Default: true, PreRelease: featuregate.Beta},
	CSIMigrationOpenStack:                       {Default: false, PreRelease: featuregate.Alpha},
	VolumeSubpath:                               {Default: true, PreRelease: featuregate.GA},
	BalanceAttachedNodeVolumes:                  {Default: false, PreRelease: featuregate.Alpha},
	PodReadinessGates:                           {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.16
	VolumeSubpathEnvExpansion:                   {Default: true, PreRelease: featuregate.Beta},
	KubeletPluginsWatcher:                       {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.16
	ResourceQuotaScopeSelectors:                 {Default: true, PreRelease: featuregate.Beta},
	CSIBlockVolume:                              {Default: true, PreRelease: featuregate.Beta},
	CSIInlineVolume:                             {Default: false, PreRelease: featuregate.Alpha},
	RuntimeClass:                                {Default: true, PreRelease: featuregate.Beta},
	NodeLease:                                   {Default: true, PreRelease: featuregate.Beta},
	SCTPSupport:                                 {Default: false, PreRelease: featuregate.Alpha},
	VolumeSnapshotDataSource:                    {Default: false, PreRelease: featuregate.Alpha},
	ProcMountType:                               {Default: false, PreRelease: featuregate.Alpha},
	TTLAfterFinished:                            {Default: false, PreRelease: featuregate.Alpha},
	KubeletPodResources:                         {Default: true, PreRelease: featuregate.Beta},
	WindowsGMSA:                                 {Default: false, PreRelease: featuregate.Alpha},
	ServiceLoadBalancerFinalizer:                {Default: false, PreRelease: featuregate.Alpha},
	LocalStorageCapacityIsolationFSQuotaMonitoring: {Default: false, PreRelease: featuregate.Alpha},
	NonPreemptingPriority:                          {Default: false, PreRelease: featuregate.Alpha},
	VolumePVCDataSource:                            {Default: false, PreRelease: featuregate.Alpha},

	// inherited features from generic apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:
	genericfeatures.StreamingProxyRedirects: {Default: true, PreRelease: featuregate.Beta},
	genericfeatures.ValidateProxyRedirects:  {Default: true, PreRelease: featuregate.Beta},
	genericfeatures.AdvancedAuditing:        {Default: true, PreRelease: featuregate.GA},
	genericfeatures.DynamicAuditing:         {Default: false, PreRelease: featuregate.Alpha},
	genericfeatures.APIResponseCompression:  {Default: false, PreRelease: featuregate.Alpha},
	genericfeatures.APIListChunking:         {Default: true, PreRelease: featuregate.Beta},
	genericfeatures.DryRun:                  {Default: true, PreRelease: featuregate.Beta},
	genericfeatures.ServerSideApply:         {Default: false, PreRelease: featuregate.Alpha},
	genericfeatures.RequestManagement:       {Default: false, PreRelease: featuregate.Alpha},

	// inherited features from apiextensions-apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:
	apiextensionsfeatures.CustomResourceValidation:        {Default: true, PreRelease: featuregate.Beta},
	apiextensionsfeatures.CustomResourceSubresources:      {Default: true, PreRelease: featuregate.Beta},
	apiextensionsfeatures.CustomResourceWebhookConversion: {Default: true, PreRelease: featuregate.Beta},
	apiextensionsfeatures.CustomResourcePublishOpenAPI:    {Default: true, PreRelease: featuregate.Beta},
	apiextensionsfeatures.CustomResourceDefaulting:        {Default: false, PreRelease: featuregate.Alpha},

	// features that enable backwards compatibility but are scheduled to be removed
	// ...
}
