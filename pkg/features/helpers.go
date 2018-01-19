/*
Copyright 2015 The Kubernetes Authors.

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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// DropDisabledAlphaFields removes disabled fields from the pod spec.
// This should be called from PrepareForCreate/PrepareForUpdate for all resources containing a pod spec.
func DropDisabledPodSpecAlphaFields(podSpec *api.PodSpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(PodPriority) {
		podSpec.Priority = nil
		podSpec.PriorityClassName = ""
	}

	if !utilfeature.DefaultFeatureGate.Enabled(LocalStorageCapacityIsolation) {
		for i := range podSpec.Volumes {
			if podSpec.Volumes[i].EmptyDir != nil {
				podSpec.Volumes[i].EmptyDir.SizeLimit = nil
			}
		}
	}

	for i := range podSpec.Containers {
		DropDisabledVolumeMountsAlphaFields(podSpec.Containers[i].VolumeMounts)
	}
	for i := range podSpec.InitContainers {
		DropDisabledVolumeMountsAlphaFields(podSpec.InitContainers[i].VolumeMounts)
	}

	DropDisabledVolumeDevicesAlphaFields(podSpec)
}

// DropDisabledVolumeMountsAlphaFields removes disabled fields from []VolumeMount.
// This should be called from PrepareForCreate/PrepareForUpdate for all resources containing a VolumeMount
func DropDisabledVolumeMountsAlphaFields(volumeMounts []api.VolumeMount) {
	if !utilfeature.DefaultFeatureGate.Enabled(MountPropagation) {
		for i := range volumeMounts {
			volumeMounts[i].MountPropagation = nil
		}
	}
}

// DropDisabledVolumeDevicesAlphaFields removes disabled fields from []VolumeDevice.
// This should be called from PrepareForCreate/PrepareForUpdate for all resources containing a VolumeDevice
func DropDisabledVolumeDevicesAlphaFields(podSpec *api.PodSpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(BlockVolume) {
		for i := range podSpec.Containers {
			podSpec.Containers[i].VolumeDevices = nil
		}
		for i := range podSpec.InitContainers {
			podSpec.InitContainers[i].VolumeDevices = nil
		}
	}
}

// DropDisabledPersistentVolumeSpecAlphaFields removes disabled fields from the pv spec.
// This should be called from PrepareForCreate/PrepareForUpdate for all resources containing a pv spec.
func DropDisabledPersistentVolumeSpecAlphaFields(pvSpec *api.PersistentVolumeSpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(BlockVolume) {
		pvSpec.VolumeMode = nil
	}
}
