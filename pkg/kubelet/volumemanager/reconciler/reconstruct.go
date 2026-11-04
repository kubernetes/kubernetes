/*
Copyright 2022 The Kubernetes Authors.

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

package reconciler

import (
	"context"
	"path/filepath"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

// readyToUnmount returns true when reconciler can start unmounting volumes.
func (rc *reconciler) readyToUnmount() bool {
	// During kubelet startup, all volumes present on disk are added as uncertain to ASW.
	// Allow unmount only when DSW is fully populated to prevent unmounting volumes that
	// did not reach DSW yet.
	if !rc.populatorHasAddedPods() {
		return false
	}

	// Allow unmount only when ASW device paths were corrected from node.status to prevent
	// calling unmount with a wrong devicePath.
	if len(rc.volumesNeedUpdateFromNodeStatus) != 0 {
		return false
	}
	return true
}

// reconstructVolumes tries to reconstruct the actual state of world by scanning all pods' volume
// directories from the disk. For the volumes that cannot support or fail reconstruction, it will
// put the volumes to volumesFailedReconstruction to be cleaned up later when DesiredStateOfWorld
// is populated.
func (rc *reconciler) reconstructVolumes(logger klog.Logger) {
	// 1. Existing: scan pod volume dirs
	podVolumes, err := getVolumesFromPodDir(logger, rc.kubeletPodsDir)
	if err != nil {
		logger.Info("Cannot get volumes from disk, continuing with CSI global reconstruction", "err", err)
		podVolumes = []podVolume{}
	}

	reconstructedVolumes := make(map[v1.UniqueVolumeName]*globalVolumeInfo)
	reconstructedVolumeNames := []v1.UniqueVolumeName{}

	// 2. Existing flow: reconstruct pod volumes
	for _, volume := range podVolumes {
		if rc.actualStateOfWorld.VolumeExistsWithSpecName(volume.podName, volume.volumeSpecName) {
			logger.V(4).Info("Volume exists in actual state, skip cleaning up mounts",
				"podName", volume.podName, "volumeSpecName", volume.volumeSpecName)
			continue
		}

		reconVol, err := rc.reconstructVolume(volume)
		if err != nil {
			logger.Info("Could not construct volume information",
				"podName", volume.podName, "volumeSpecName", volume.volumeSpecName, "err", err)
			rc.volumesFailedReconstruction = append(rc.volumesFailedReconstruction, volume)
			continue
		}

		gvl := &globalVolumeInfo{
			volumeName:        reconVol.volumeName,
			volumeSpec:        reconVol.volumeSpec,
			devicePath:        reconVol.devicePath,
			deviceMounter:     reconVol.deviceMounter,
			blockVolumeMapper: reconVol.blockVolumeMapper,
			mounter:           reconVol.mounter,
			podVolumes:        make(map[volumetypes.UniquePodName]*reconstructedVolume),
		}

		if cachedInfo, ok := reconstructedVolumes[reconVol.volumeName]; ok {
			gvl = cachedInfo
		}

		gvl.addPodVolume(reconVol)

		reconstructedVolumes[reconVol.volumeName] = gvl
		reconstructedVolumeNames = append(reconstructedVolumeNames, reconVol.volumeName)
	}

	// 3. NEW: reconstruct CSI global mounts (NO pod dirs)
	csiVolumes, err := rc.getVolumesFromCSIGlobalDir(logger, filepath.Dir(rc.kubeletPodsDir))
	if err != nil {
		logger.Error(err, "Failed to scan CSI global mounts")
	} else {
		for _, csiVol := range csiVolumes {

			// Avoid duplicates
			if _, exists := reconstructedVolumes[csiVol.volumeName]; exists {
				continue
			}

			gvl := &globalVolumeInfo{
				volumeName:        csiVol.volumeName,
				volumeSpec:        csiVol.volumeSpec,
				devicePath:        csiVol.devicePath,
				deviceMounter:     csiVol.deviceMounter,
				blockVolumeMapper: csiVol.blockVolumeMapper,
				mounter:           csiVol.mounter,
				podVolumes:        nil,
			}

			reconstructedVolumes[csiVol.volumeName] = gvl
			reconstructedVolumeNames = append(reconstructedVolumeNames, csiVol.volumeName)

			logger.V(4).Info("Reconstructed CSI global mount volume",
				"volumeName", csiVol.volumeName)
		}
	}

	// 4. Final state update
	if len(reconstructedVolumes) > 0 {
		rc.updateStates(logger, reconstructedVolumes)

		// Ensure node status reconciliation happens
		rc.volumesNeedUpdateFromNodeStatus = reconstructedVolumeNames
	}

	logger.V(2).Info("Volume reconstruction finished")
}

func (rc *reconciler) updateStates(logger klog.Logger, reconstructedVolumes map[v1.UniqueVolumeName]*globalVolumeInfo) {
	for _, gvl := range reconstructedVolumes {
		var err error
		var seLinuxMountContext string
		err = rc.actualStateOfWorld.AddAttachUncertainReconstructedVolume(
			logger,
			gvl.volumeName,
			gvl.volumeSpec,
			rc.nodeName,
			gvl.devicePath,
		)
		if err != nil {
			logger.Error(err, "Could not add volume to actual state of world", "volumeName", gvl.volumeName)
			continue
		}

		if len(gvl.podVolumes) > 0 {
			for _, volume := range gvl.podVolumes {
				markVolumeOpts := operationexecutor.MarkVolumeOpts{
					PodName:             volume.podName,
					PodUID:              types.UID(volume.podName),
					VolumeName:          volume.volumeName,
					Mounter:             volume.mounter,
					BlockVolumeMapper:   volume.blockVolumeMapper,
					VolumeGIDVolume:     volume.volumeGIDValue,
					VolumeSpec:          volume.volumeSpec,
					VolumeMountState:    operationexecutor.VolumeMountUncertain,
					SELinuxMountContext: volume.seLinuxMountContext,
				}

				_, err = rc.actualStateOfWorld.CheckAndMarkVolumeAsUncertainViaReconstruction(markVolumeOpts)
				if err != nil {
					logger.Error(err, "Could not add pod to volume information to actual state of world", "pod", klog.KObj(volume.pod))
					continue
				}
				seLinuxMountContext = volume.seLinuxMountContext
				logger.V(2).Info("Volume is marked as uncertain and added into the actual state", "pod", klog.KObj(volume.pod), "podName", volume.podName, "volumeName", volume.volumeName, "seLinuxMountContext", volume.seLinuxMountContext)
			}
		}
		// If the volume has device to mount, we mark its device as uncertain.
		if gvl.deviceMounter != nil || gvl.blockVolumeMapper != nil {
			deviceMountPath, err := getDeviceMountPath(gvl)
			if err != nil {
				logger.Error(err, "Could not find device mount path for volume", "volumeName", gvl.volumeName)
				continue
			}
			if len(gvl.podVolumes) == 0 {
				err = rc.actualStateOfWorld.MarkDeviceAsUncertainViaReconstruction(
					gvl.volumeName,
					gvl.devicePath,
					deviceMountPath,
					seLinuxMountContext,
				)
				if err != nil {
					logger.Error(err, "Could not mark device as uncertain via reconstruction", "volumeName", gvl.volumeName, "deviceMountPath", deviceMountPath)
					continue
				}
				logger.V(2).Info("CSI globalmount-only volume marked as device uncertain via reconstruction", "volumeName", gvl.volumeName, "deviceMountPath", deviceMountPath)
				continue
			}

			err = rc.actualStateOfWorld.MarkDeviceAsUncertain(gvl.volumeName, gvl.devicePath, deviceMountPath, seLinuxMountContext)
			if err != nil {
				logger.Error(err, "Could not mark device is uncertain to actual state of world", "volumeName", gvl.volumeName, "deviceMountPath", deviceMountPath)
				continue
			}
			logger.V(2).Info("Volume is marked device as uncertain and added into the actual state", "volumeName", gvl.volumeName, "deviceMountPath", deviceMountPath)
		}
	}
}

// cleanOrphanVolumes tries to clean up all volumes that failed reconstruction.
func (rc *reconciler) cleanOrphanVolumes(logger klog.Logger) {
	if len(rc.volumesFailedReconstruction) == 0 {
		return
	}

	for _, volume := range rc.volumesFailedReconstruction {
		if rc.desiredStateOfWorld.VolumeExistsWithSpecName(volume.podName, volume.volumeSpecName) {
			// Some pod needs the volume, don't clean it up and hope that
			// reconcile() calls SetUp and reconstructs the volume in ASW.
			logger.V(4).Info("Volume exists in desired state, skip cleaning up mounts", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName)
			continue
		}
		logger.Info("Cleaning up mounts for volume that could not be reconstructed", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName)
		rc.cleanupMounts(logger, volume)
	}

	logger.V(2).Info("Orphan volume cleanup finished")
	// Clean the cache, cleanup is one shot operation.
	rc.volumesFailedReconstruction = make([]podVolume, 0)
}

// updateReconstructedFromNodeStatus tries to file devicePaths of reconstructed volumes from
// node.Status.VolumesAttached. This can be done only after connection to the API
// server is established, i.e. it can't be part of reconstructVolumes().
func (rc *reconciler) updateReconstructedFromNodeStatus(ctx context.Context) {
	logger := klog.FromContext(ctx)
	logger.V(4).Info("Updating reconstructed devicePaths")

	if rc.kubeClient == nil {
		// Skip reconstructing devicePath from node objects if kubelet is in standalone mode.
		// Such kubelet is not expected to mount any attachable volume or Secrets / ConfigMap.
		logger.V(2).Info("Skipped reconstruction of DevicePaths from node.status in standalone mode")
		rc.volumesNeedUpdateFromNodeStatus = nil
		return
	}

	node, fetchErr := rc.kubeClient.CoreV1().Nodes().Get(ctx, string(rc.nodeName), metav1.GetOptions{})
	if fetchErr != nil {
		// This may repeat few times per second until kubelet is able to read its own status for the first time.
		logger.V(4).Info("Failed to get Node status to reconstruct device paths", "err", fetchErr)
		return
	}

	for _, volumeID := range rc.volumesNeedUpdateFromNodeStatus {
		attachable := false
		for _, attachedVolume := range node.Status.VolumesAttached {
			if volumeID != attachedVolume.Name {
				continue
			}
			rc.actualStateOfWorld.UpdateReconstructedDevicePath(volumeID, attachedVolume.DevicePath)
			attachable = true
			logger.V(4).Info("Updated devicePath from node status for volume", "volumeName", attachedVolume.Name, "path", attachedVolume.DevicePath)
		}
		rc.actualStateOfWorld.UpdateReconstructedVolumeAttachability(volumeID, attachable)
	}

	logger.V(2).Info("DevicePaths of reconstructed volumes updated")
	rc.volumesNeedUpdateFromNodeStatus = nil

}
