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

	// owner: @girishkalele
	// alpha: v1.4
	ExternalTrafficLocalOnly utilfeature.Feature = "AllowExtTrafficLocalEndpoints"

	// owner: @mtaufen
	// alpha: v1.4
	DynamicKubeletConfig utilfeature.Feature = "DynamicKubeletConfig"

	// owner: @mtaufen
	// alpha: v1.8
	KubeletConfigFile utilfeature.Feature = "KubeletConfigFile"

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

	// owner: @vishh
	// alpha: v1.6
	//
	// Enables support for GPUs as a schedulable resource.
	// Only Nvidia GPUs are supported as of v1.6.
	// Works only with Docker Container Runtime.
	Accelerators utilfeature.Feature = "Accelerators"

	// owner: @jiayingz
	// alpha: v1.8
	//
	// Enables support for Device Plugins
	// Only Nvidia GPUs are tested as of v1.8.
	DevicePlugins utilfeature.Feature = "DevicePlugins"

	// owner: @gmarek
	// alpha: v1.6
	//
	// Changes the logic behind evicting Pods from not ready Nodes
	// to take advantage of NoExecute Taints and Tolerations.
	TaintBasedEvictions utilfeature.Feature = "TaintBasedEvictions"

	// owner: @jcbsmpsn
	// alpha: v1.7
	//
	// Gets a server certificate for the kubelet from the Certificate Signing
	// Request API instead of generating one self signed and auto rotates the
	// certificate as expiration approaches.
	RotateKubeletServerCertificate utilfeature.Feature = "RotateKubeletServerCertificate"

	// owner: @jcbsmpsn
	// alpha: v1.7
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
	// alpha: v1.7
	//
	// New local storage types to support local storage capacity isolation
	LocalStorageCapacityIsolation utilfeature.Feature = "LocalStorageCapacityIsolation"

	// owner: @gnufied
	// alpha: v1.8
	// Ability to Expand persistent volumes
	ExpandPersistentVolumes utilfeature.Feature = "ExpandPersistentVolumes"

	// owner: @verb
	// alpha: v1.8
	//
	// Allows running a "debug container" in a pod namespaces to troubleshoot a running pod.
	DebugContainers utilfeature.Feature = "DebugContainers"

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
	// alpha: v1.8
	//
	// Taint nodes based on their condition status for 'NetworkUnavailable',
	// 'MemoryPressure', 'OutOfDisk' and 'DiskPressure'.
	TaintNodesByCondition utilfeature.Feature = "TaintNodesByCondition"

	// owner: @jsafrane
	// alpha: v1.8
	//
	// Enable mount propagation of volumes.
	MountPropagation utilfeature.Feature = "MountPropagation"

	// owner: @ConnorDoyle
	// alpha: v1.8
	//
	// Alternative container-level CPU affinity policies.
	CPUManager utilfeature.Feature = "CPUManager"

	// owner: @derekwaynecarr
	// alpha: v1.8
	//
	// Enable pods to consume pre-allocated huge pages of varying page sizes
	HugePages utilfeature.Feature = "HugePages"

	// owner @brendandburns
	// alpha: v1.8
	//
	// Enable nodes to exclude themselves from service load balancers
	ServiceNodeExclusion utilfeature.Feature = "ServiceNodeExclusion"

	// owner @brendandburns
	// deprecated: v1.10
	//
	// Enable the service proxy to contact external IP addresses. Note this feature is present
	// only for backward compatability, it will be removed in the 1.11 release.
	ServiceProxyAllowExternalIPs utilfeature.Feature = "ServiceProxyAllowExternalIPs"

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
	// alpha: v1.9
	//
	// Enable mount/attachment of Container Storage Interface (CSI) backed PVs
	CSIPersistentVolume utilfeature.Feature = "CSIPersistentVolume"

	// owner @MrHohn
	// alpha: v1.9
	//
	// Support configurable pod DNS parameters.
	CustomPodDNS utilfeature.Feature = "CustomPodDNS"

	// owner: @screeley44
	// alpha: v1.9
	//
	// Enable Block volume support in containers.
	BlockVolume utilfeature.Feature = "BlockVolume"

	// owner: @pospispa
	//
	// alpha: v1.9
	// Postpone deletion of a persistent volume claim in case it is used by a pod
	PVCProtection utilfeature.Feature = "PVCProtection"

	// owner: @aveshagarwal
	// alpha: v1.9
	//
	// Enable resource limits priority function
	ResourceLimitsPriorityFunction utilfeature.Feature = "ResourceLimitsPriorityFunction"

	// owner: @m1093782566
	// beta: v1.9
	//
	// Implement IPVS-based in-cluster service load balancing
	SupportIPVSProxyMode utilfeature.Feature = "SupportIPVSProxyMode"

	// owner: @joelsmith
	// deprecated: v1.10
	//
	// Mount secret, configMap, downwardAPI and projected volumes ReadOnly. Note: this feature
	// gate is present only for backward compatability, it will be removed in the 1.11 release.
	ReadOnlyAPIDataVolumes utilfeature.Feature = "ReadOnlyAPIDataVolumes"

	// owner: @saad-ali
	// ga
	//
	// Allow mounting a subpath of a volume in a container
	// Do not remove this feature gate even though it's GA
	VolumeSubpath utilfeature.Feature = "VolumeSubpath"
)

func init() {
	utilfeature.DefaultFeatureGate.Add(defaultKubernetesFeatureGates)
}

// defaultKubernetesFeatureGates consists of all known Kubernetes-specific feature keys.
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
var defaultKubernetesFeatureGates = map[utilfeature.Feature]utilfeature.FeatureSpec{
	ExternalTrafficLocalOnly:                    {Default: true, PreRelease: utilfeature.GA},
	AppArmor:                                    {Default: true, PreRelease: utilfeature.Beta},
	DynamicKubeletConfig:                        {Default: false, PreRelease: utilfeature.Alpha},
	KubeletConfigFile:                           {Default: false, PreRelease: utilfeature.Alpha},
	ExperimentalHostUserNamespaceDefaultingGate: {Default: false, PreRelease: utilfeature.Beta},
	ExperimentalCriticalPodAnnotation:           {Default: false, PreRelease: utilfeature.Alpha},
	Accelerators:                                {Default: false, PreRelease: utilfeature.Alpha},
	DevicePlugins:                               {Default: false, PreRelease: utilfeature.Alpha},
	TaintBasedEvictions:                         {Default: false, PreRelease: utilfeature.Alpha},
	RotateKubeletServerCertificate:              {Default: false, PreRelease: utilfeature.Alpha},
	RotateKubeletClientCertificate:              {Default: true, PreRelease: utilfeature.Beta},
	PersistentLocalVolumes:                      {Default: false, PreRelease: utilfeature.Alpha},
	LocalStorageCapacityIsolation:               {Default: false, PreRelease: utilfeature.Alpha},
	HugePages:                                   {Default: false, PreRelease: utilfeature.Alpha},
	DebugContainers:                             {Default: false, PreRelease: utilfeature.Alpha},
	PodPriority:                                 {Default: false, PreRelease: utilfeature.Alpha},
	EnableEquivalenceClassCache:                 {Default: false, PreRelease: utilfeature.Alpha},
	TaintNodesByCondition:                       {Default: false, PreRelease: utilfeature.Alpha},
	MountPropagation:                            {Default: false, PreRelease: utilfeature.Alpha},
	ExpandPersistentVolumes:                     {Default: false, PreRelease: utilfeature.Alpha},
	CPUManager:                                  {Default: false, PreRelease: utilfeature.Alpha},
	ServiceNodeExclusion:                        {Default: false, PreRelease: utilfeature.Alpha},
	MountContainers:                             {Default: false, PreRelease: utilfeature.Alpha},
	VolumeScheduling:                            {Default: false, PreRelease: utilfeature.Alpha},
	CSIPersistentVolume:                         {Default: false, PreRelease: utilfeature.Alpha},
	CustomPodDNS:                                {Default: false, PreRelease: utilfeature.Alpha},
	BlockVolume:                                 {Default: false, PreRelease: utilfeature.Alpha},
	PVCProtection:                               {Default: false, PreRelease: utilfeature.Alpha},
	ResourceLimitsPriorityFunction:              {Default: false, PreRelease: utilfeature.Alpha},
	SupportIPVSProxyMode:                        {Default: false, PreRelease: utilfeature.Beta},
	VolumeSubpath:                               {Default: true, PreRelease: utilfeature.GA},

	// inherited features from generic apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:
	genericfeatures.StreamingProxyRedirects: {Default: true, PreRelease: utilfeature.Beta},
	genericfeatures.AdvancedAuditing:        {Default: true, PreRelease: utilfeature.Beta},
	genericfeatures.APIResponseCompression:  {Default: false, PreRelease: utilfeature.Alpha},
	genericfeatures.Initializers:            {Default: false, PreRelease: utilfeature.Alpha},
	genericfeatures.APIListChunking:         {Default: true, PreRelease: utilfeature.Beta},

	// inherited features from apiextensions-apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:
	apiextensionsfeatures.CustomResourceValidation: {Default: true, PreRelease: utilfeature.Beta},

	// features that enable backwards compatability but are scheduled to be removed
	ServiceProxyAllowExternalIPs: {Default: false, PreRelease: utilfeature.Deprecated},
	ReadOnlyAPIDataVolumes:       {Default: true, PreRelease: utilfeature.Deprecated},
}
