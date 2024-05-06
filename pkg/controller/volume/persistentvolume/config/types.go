/*
Copyright 2019 The Kubernetes Authors.

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
)

// PersistentVolumeBinderControllerConfiguration contains elements describing
// PersistentVolumeBinderController.
type PersistentVolumeBinderControllerConfiguration struct {
	// pvClaimBinderSyncPeriod is the period for syncing persistent volumes
	// and persistent volume claims.
	PVClaimBinderSyncPeriod metav1.Duration
	// volumeConfiguration holds configuration for volume related features.
	VolumeConfiguration VolumeConfiguration
	// DEPRECATED: VolumeHostCIDRDenylist is a list of CIDRs that should not be reachable by the
	// controller from plugins.
	VolumeHostCIDRDenylist []string
	// DEPRECATED: VolumeHostAllowLocalLoopback indicates if local loopback hosts (127.0.0.1, etc)
	// should be allowed from plugins.
	VolumeHostAllowLocalLoopback bool
}

// VolumeConfiguration contains *all* enumerated flags meant to configure all volume
// plugins. From this config, the controller-manager binary will create many instances of
// volume.VolumeConfig, each containing only the configuration needed for that plugin which
// are then passed to the appropriate plugin. The ControllerManager binary is the only part
// of the code which knows what plugins are supported and which flags correspond to each plugin.
type VolumeConfiguration struct {
	// enableHostPathProvisioning enables HostPath PV provisioning when running without a
	// cloud provider. This allows testing and development of provisioning features. HostPath
	// provisioning is not supported in any way, won't work in a multi-node cluster, and
	// should not be used for anything other than testing or development.
	EnableHostPathProvisioning bool
	// enableDynamicProvisioning enables the provisioning of volumes when running within an environment
	// that supports dynamic provisioning. Defaults to true.
	EnableDynamicProvisioning bool
	// persistentVolumeRecyclerConfiguration holds configuration for persistent volume plugins.
	PersistentVolumeRecyclerConfiguration PersistentVolumeRecyclerConfiguration
	// volumePluginDir is the full path of the directory in which the flex
	// volume plugin should search for additional third party volume plugins
	FlexVolumePluginDir string
}

// PersistentVolumeRecyclerConfiguration contains elements describing persistent volume plugins.
type PersistentVolumeRecyclerConfiguration struct {
	// maximumRetry is number of retries the PV recycler will execute on failure to recycle
	// PV.
	MaximumRetry int32
	// minimumTimeoutNFS is the minimum ActiveDeadlineSeconds to use for an NFS Recycler
	// pod.
	MinimumTimeoutNFS int32
	// podTemplateFilePathNFS is the file path to a pod definition used as a template for
	// NFS persistent volume recycling
	PodTemplateFilePathNFS string
	// incrementTimeoutNFS is the increment of time added per Gi to ActiveDeadlineSeconds
	// for an NFS scrubber pod.
	IncrementTimeoutNFS int32
	// podTemplateFilePathHostPath is the file path to a pod definition used as a template for
	// HostPath persistent volume recycling. This is for development and testing only and
	// will not work in a multi-node cluster.
	PodTemplateFilePathHostPath string
	// minimumTimeoutHostPath is the minimum ActiveDeadlineSeconds to use for a HostPath
	// Recycler pod.  This is for development and testing only and will not work in a multi-node
	// cluster.
	MinimumTimeoutHostPath int32
	// incrementTimeoutHostPath is the increment of time added per Gi to ActiveDeadlineSeconds
	// for a HostPath scrubber pod.  This is for development and testing only and will not work
	// in a multi-node cluster.
	IncrementTimeoutHostPath int32
}
