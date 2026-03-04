/*
Copyright 2016 The Kubernetes Authors.

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

package options

import (
	"time"

	"github.com/spf13/pflag"
)

// VolumeConfigFlags is used to bind CLI flags to variables.  This top-level struct contains *all* enumerated
// CLI flags meant to configure all volume plugins.  From this config, the binary will create many instances
// of volume.VolumeConfig which are then passed to the appropriate plugin. The ControllerManager binary is the only
// part of the code which knows what plugins are supported and which CLI flags correspond to each plugin.
type VolumeConfigFlags struct {
	PersistentVolumeRecyclerMaximumRetry                int
	PersistentVolumeRecyclerMinimumTimeoutNFS           int
	PersistentVolumeRecyclerPodTemplateFilePathNFS      string
	PersistentVolumeRecyclerIncrementTimeoutNFS         int
	PersistentVolumeRecyclerPodTemplateFilePathHostPath string
	PersistentVolumeRecyclerMinimumTimeoutHostPath      int
	PersistentVolumeRecyclerIncrementTimeoutHostPath    int
	EnableHostPathProvisioning                          bool
	EnableDynamicProvisioning                           bool
}

// PersistentVolumeControllerOptions holds the PersistentVolumeController options.
type PersistentVolumeControllerOptions struct {
	PVClaimBinderSyncPeriod time.Duration
	VolumeConfigFlags       VolumeConfigFlags
}

// NewPersistentVolumeControllerOptions creates a new PersistentVolumeControllerOptions with
// default values.
func NewPersistentVolumeControllerOptions() PersistentVolumeControllerOptions {
	return PersistentVolumeControllerOptions{
		PVClaimBinderSyncPeriod: 15 * time.Second,
		VolumeConfigFlags: VolumeConfigFlags{
			// default values here
			PersistentVolumeRecyclerMaximumRetry:             3,
			PersistentVolumeRecyclerMinimumTimeoutNFS:        300,
			PersistentVolumeRecyclerIncrementTimeoutNFS:      30,
			PersistentVolumeRecyclerMinimumTimeoutHostPath:   60,
			PersistentVolumeRecyclerIncrementTimeoutHostPath: 30,
			EnableHostPathProvisioning:                       false,
			EnableDynamicProvisioning:                        true,
		},
	}
}

// AddFlags adds flags related to PersistentVolumeControllerOptions to the specified FlagSet.
func (o *PersistentVolumeControllerOptions) AddFlags(fs *pflag.FlagSet) {
	fs.DurationVar(&o.PVClaimBinderSyncPeriod, "pvclaimbinder-sync-period", o.PVClaimBinderSyncPeriod,
		"The period for syncing persistent volumes and persistent volume claims")
	fs.StringVar(&o.VolumeConfigFlags.PersistentVolumeRecyclerPodTemplateFilePathNFS,
		"pv-recycler-pod-template-filepath-nfs", o.VolumeConfigFlags.PersistentVolumeRecyclerPodTemplateFilePathNFS,
		"The file path to a pod definition used as a template for NFS persistent volume recycling")
	fs.IntVar(&o.VolumeConfigFlags.PersistentVolumeRecyclerMinimumTimeoutNFS, "pv-recycler-minimum-timeout-nfs",
		o.VolumeConfigFlags.PersistentVolumeRecyclerMinimumTimeoutNFS, "The minimum ActiveDeadlineSeconds to use for an NFS Recycler pod")
	fs.IntVar(&o.VolumeConfigFlags.PersistentVolumeRecyclerIncrementTimeoutNFS, "pv-recycler-increment-timeout-nfs",
		o.VolumeConfigFlags.PersistentVolumeRecyclerIncrementTimeoutNFS, "the increment of time added per Gi to ActiveDeadlineSeconds for an NFS scrubber pod")
	fs.StringVar(&o.VolumeConfigFlags.PersistentVolumeRecyclerPodTemplateFilePathHostPath, "pv-recycler-pod-template-filepath-hostpath",
		o.VolumeConfigFlags.PersistentVolumeRecyclerPodTemplateFilePathHostPath,
		"The file path to a pod definition used as a template for HostPath persistent volume recycling. "+
			"This is for development and testing only and will not work in a multi-node cluster.")
	fs.IntVar(&o.VolumeConfigFlags.PersistentVolumeRecyclerMinimumTimeoutHostPath, "pv-recycler-minimum-timeout-hostpath",
		o.VolumeConfigFlags.PersistentVolumeRecyclerMinimumTimeoutHostPath,
		"The minimum ActiveDeadlineSeconds to use for a HostPath Recycler pod. This is for development and testing only and will not work in a multi-node cluster.")
	fs.IntVar(&o.VolumeConfigFlags.PersistentVolumeRecyclerIncrementTimeoutHostPath, "pv-recycler-timeout-increment-hostpath",
		o.VolumeConfigFlags.PersistentVolumeRecyclerIncrementTimeoutHostPath,
		"the increment of time added per Gi to ActiveDeadlineSeconds for a HostPath scrubber pod. "+
			"This is for development and testing only and will not work in a multi-node cluster.")
	fs.IntVar(&o.VolumeConfigFlags.PersistentVolumeRecyclerMaximumRetry, "pv-recycler-maximum-retry",
		o.VolumeConfigFlags.PersistentVolumeRecyclerMaximumRetry,
		"Maximum number of attempts to recycle or delete a persistent volume")
	fs.BoolVar(&o.VolumeConfigFlags.EnableHostPathProvisioning, "enable-hostpath-provisioner", o.VolumeConfigFlags.EnableHostPathProvisioning,
		"Enable HostPath PV provisioning when running without a cloud provider. This allows testing and development of provisioning features. "+
			"HostPath provisioning is not supported in any way, won't work in a multi-node cluster, and should not be used for anything other than testing or development.")
	fs.BoolVar(&o.VolumeConfigFlags.EnableDynamicProvisioning, "enable-dynamic-provisioning", o.VolumeConfigFlags.EnableDynamicProvisioning,
		"Enable dynamic provisioning for environments that support it.")
}
