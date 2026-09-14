/*
Copyright The Kubernetes Authors.

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

package cache

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csi"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

// VolumeToHealthCheck describes a volume that should be probed for health.
type VolumeToHealthCheck struct {
	VolumeName      v1.UniqueVolumeName
	PodName         volumetypes.UniquePodName
	Pod             *v1.Pod
	OuterVolumeName string // pod.spec.volumes[].name
	DriverName      string
	CSIVolumeHandle string
	StagingPath     string
	PublishPath     string
	VolumeSpec      *volume.Spec
}

// GetVolumesToReportHealth returns CSI volumes in DSW that have been attempted
// (or are mounted/uncertain in ASW, which implies attempt). Paths may be empty
// if mount never succeeded.
func GetVolumesToReportHealth(dsw DesiredStateOfWorld, asw ActualStateOfWorld) []VolumeToHealthCheck {
	volumesToMount := dsw.GetVolumesToMount()
	result := make([]VolumeToHealthCheck, 0, len(volumesToMount))

	for _, vtm := range volumesToMount {
		if !isCSIVolumeSpec(vtm.VolumeSpec) {
			continue
		}

		attempted := asw.IsVolumeMountAttempted(vtm.PodName, vtm.VolumeName)
		mountState := asw.GetVolumeMountState(vtm.VolumeName, vtm.PodName)
		if !attempted && mountState == operationexecutor.VolumeNotMounted {
			continue
		}

		driverName, err := csi.GetCSIDriverName(vtm.VolumeSpec)
		if err != nil {
			continue
		}

		handle, ok := csiVolumeHandle(vtm.VolumeSpec)
		if !ok {
			// Ephemeral CSI volumes without a PV handle are skipped for now.
			continue
		}

		outerVolumeName := ""
		if len(vtm.OuterVolumeSpecNames) > 0 {
			outerVolumeName = vtm.OuterVolumeSpecNames[0]
		}

		stagingPath, publishPath := volumeHealthPaths(asw, vtm.PodName, vtm.VolumeName)

		result = append(result, VolumeToHealthCheck{
			VolumeName:      vtm.VolumeName,
			PodName:         vtm.PodName,
			Pod:             vtm.Pod,
			OuterVolumeName: outerVolumeName,
			DriverName:      driverName,
			CSIVolumeHandle: handle,
			StagingPath:     stagingPath,
			PublishPath:     publishPath,
			VolumeSpec:      vtm.VolumeSpec,
		})
	}

	return result
}

func isCSIVolumeSpec(spec *volume.Spec) bool {
	if spec == nil {
		return false
	}
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.CSI != nil {
		return true
	}
	if spec.Volume != nil && spec.Volume.CSI != nil {
		return true
	}
	return false
}

func csiVolumeHandle(spec *volume.Spec) (string, bool) {
	if spec == nil {
		return "", false
	}
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.CSI != nil {
		handle := spec.PersistentVolume.Spec.CSI.VolumeHandle
		if handle == "" {
			return "", false
		}
		return handle, true
	}
	// Ephemeral CSI volumes: handle is derived at mount time; skip for now.
	return "", false
}

func volumeHealthPaths(asw ActualStateOfWorld, podName volumetypes.UniquePodName, volumeName v1.UniqueVolumeName) (stagingPath, publishPath string) {
	for _, mv := range asw.GetPossiblyMountedVolumesForPod(podName) {
		if mv.VolumeName != volumeName {
			continue
		}
		stagingPath = mv.DeviceMountPath
		if mv.Mounter != nil {
			publishPath = mv.Mounter.GetPath()
		}
		return stagingPath, publishPath
	}
	if attached, ok := asw.GetAttachedVolume(volumeName); ok {
		stagingPath = attached.DeviceMountPath
	}
	return stagingPath, publishPath
}
