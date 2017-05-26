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
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

const (
	// Every feature gate should add method here following this template:
	//
	// // owner: @username
	// // alpha: v1.X
	// MyFeature utilfeature.Feature = "MyFeature"

	// owner: @timstclair
	// beta: v1.4
	AppArmor utilfeature.Feature = "AppArmor"

	// owner: @girishkalele
	// alpha: v1.4
	ExternalTrafficLocalOnly utilfeature.Feature = "AllowExtTrafficLocalEndpoints"

	// owner: @saad-ali
	// alpha: v1.3
	DynamicVolumeProvisioning utilfeature.Feature = "DynamicVolumeProvisioning"

	// owner: @mtaufen
	// alpha: v1.4
	DynamicKubeletConfig utilfeature.Feature = "DynamicKubeletConfig"

	// owner: timstclair
	// alpha: v1.5
	//
	// StreamingProxyRedirects controls whether the apiserver should intercept (and follow)
	// redirects from the backend (Kubelet) for streaming requests (exec/attach/port-forward).
	StreamingProxyRedirects utilfeature.Feature = genericfeatures.StreamingProxyRedirects

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

	// owner: @davidopp
	// alpha: v1.6
	//
	// Determines if affinity defined in annotations should be processed
	// TODO: remove when alpha support for affinity is removed
	AffinityInAnnotations utilfeature.Feature = "AffinityInAnnotations"

	// owner: @vishh
	// alpha: v1.6
	//
	// Enables support for GPUs as a schedulable resource.
	// Only Nvidia GPUs are supported as of v1.6.
	// Works only with Docker Container Runtime.
	Accelerators utilfeature.Feature = "Accelerators"

	// owner: @gmarek
	// alpha: v1.6
	//
	// Changes the logic behind evicting Pods from not ready Nodes
	// to take advantage of NoExecute Taints and Tolerations.
	TaintBasedEvictions utilfeature.Feature = "TaintBasedEvictions"

	// owner: @msau
	// alpha: v1.7
	//
	// A new volume type that supports local disks on a node.
	PersistentLocalVolumes utilfeature.Feature = "PersistentLocalVolumes"
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
	DynamicVolumeProvisioning:                   {Default: true, PreRelease: utilfeature.Alpha},
	ExperimentalHostUserNamespaceDefaultingGate: {Default: false, PreRelease: utilfeature.Beta},
	ExperimentalCriticalPodAnnotation:           {Default: false, PreRelease: utilfeature.Alpha},
	AffinityInAnnotations:                       {Default: false, PreRelease: utilfeature.Alpha},
	Accelerators:                                {Default: false, PreRelease: utilfeature.Alpha},
	TaintBasedEvictions:                         {Default: false, PreRelease: utilfeature.Alpha},
	PersistentLocalVolumes:                      {Default: false, PreRelease: utilfeature.Alpha},

	// inherited features from generic apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:
	StreamingProxyRedirects:          {Default: true, PreRelease: utilfeature.Beta},
	genericfeatures.AdvancedAuditing: {Default: false, PreRelease: utilfeature.Alpha},
}
