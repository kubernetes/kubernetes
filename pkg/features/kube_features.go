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

	// owner: @msau
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

	// owner: @haibinxie
	// alpha: v1.8
	//
	// Implement IPVS-based in-cluster service load balancing
	SupportIPVSProxyMode utilfeature.Feature = "SupportIPVSProxyMode"

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

	// inherited features from generic apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:
	genericfeatures.StreamingProxyRedirects: {Default: true, PreRelease: utilfeature.Beta},
	genericfeatures.AdvancedAuditing:        {Default: true, PreRelease: utilfeature.Beta},
	genericfeatures.APIResponseCompression:  {Default: false, PreRelease: utilfeature.Alpha},
	genericfeatures.Initializers:            {Default: false, PreRelease: utilfeature.Alpha},
	genericfeatures.APIListChunking:         {Default: false, PreRelease: utilfeature.Alpha},

	// inherited features from apiextensions-apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:
	apiextensionsfeatures.CustomResourceValidation: {Default: false, PreRelease: utilfeature.Alpha},
	SupportIPVSProxyMode:                           {Default: false, PreRelease: utilfeature.Alpha},
}
