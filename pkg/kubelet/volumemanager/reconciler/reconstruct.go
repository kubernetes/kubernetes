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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	volumepkg "k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
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
	// Volumes that are still staged but have no pod directory naming them are
	// found by asking the plugins, which is independent of the walk below and
	// runs even when that walk cannot.
	if utilfeature.DefaultFeatureGate.Enabled(features.CSIGlobalMountReconstruction) {
		rc.reconstructGlobalVolumes(logger)
	}

	// Get volumes information by reading the pod's directory
	podVolumes, err := getVolumesFromPodDir(logger, rc.kubeletPodsDir)
	if err != nil {
		logger.Error(err, "Cannot get volumes from disk, skip sync states for volume reconstruction")
		return
	}
	reconstructedVolumes := make(map[v1.UniqueVolumeName]*globalVolumeInfo)
	reconstructedVolumeNames := []v1.UniqueVolumeName{}
	for _, volume := range podVolumes {
		if rc.actualStateOfWorld.VolumeExistsWithSpecName(volume.podName, volume.volumeSpecName) {
			logger.V(4).Info("Volume exists in actual state, skip cleaning up mounts", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName)
			// There is nothing to reconstruct
			continue
		}
		reconstructedVolume, err := rc.reconstructVolume(volume)
		if err != nil {
			logger.Info("Could not construct volume information", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName, "err", err)
			// We can't reconstruct the volume. Remember to check DSW after it's fully populated and force unmount the volume when it's orphaned.
			rc.volumesFailedReconstruction = append(rc.volumesFailedReconstruction, volume)
			continue
		}
		logger.V(4).Info("Adding reconstructed volume to actual state and node status", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName)
		gvl := &globalVolumeInfo{
			volumeName:        reconstructedVolume.volumeName,
			volumeSpec:        reconstructedVolume.volumeSpec,
			devicePath:        reconstructedVolume.devicePath,
			deviceMounter:     reconstructedVolume.deviceMounter,
			blockVolumeMapper: reconstructedVolume.blockVolumeMapper,
			mounter:           reconstructedVolume.mounter,
		}
		if cachedInfo, ok := reconstructedVolumes[reconstructedVolume.volumeName]; ok {
			gvl = cachedInfo
		}
		gvl.addPodVolume(reconstructedVolume)

		reconstructedVolumeNames = append(reconstructedVolumeNames, reconstructedVolume.volumeName)
		reconstructedVolumes[reconstructedVolume.volumeName] = gvl
	}

	if len(reconstructedVolumes) > 0 {
		// Add the volumes to ASW
		rc.updateStates(logger, reconstructedVolumes)

		// Remember to update devicePath from node.status.volumesAttached. A
		// volume the plugins reported above is already queued, and a pod
		// directory for it does not queue it twice.
		queued := sets.New(rc.volumesNeedUpdateFromNodeStatus...)
		for _, volumeName := range reconstructedVolumeNames {
			if !queued.Has(volumeName) {
				rc.volumesNeedUpdateFromNodeStatus = append(rc.volumesNeedUpdateFromNodeStatus, volumeName)
			}
		}
	}
	logger.V(2).Info("Volume reconstruction finished")
}

// reconstructGlobalVolumes finds volumes that are still staged on this node but
// have no pod directory left to be found through, and records them in the
// actual state of world as uncertain.
//
// The walk of /var/lib/kubelet/pods above cannot see them: kubelet lets a pod be
// deleted once NodeUnpublishVolume succeeds and does not wait for
// NodeUnstageVolume, so a restart in between leaves a staged volume behind with
// its pod directory already gone. Nothing then unstages it and nothing keeps it
// in node.status.volumesInUse, which is what lets the attach/detach controller
// attach it somewhere else while it is still mounted here.
//
// Each plugin reports its own global mounts, so this stays free of any one
// plugin's on-disk layout.
func (rc *reconciler) reconstructGlobalVolumes(logger klog.Logger) {
	for _, plugin := range rc.volumePluginMgr.FindGlobalVolumeListerPlugins() {
		globalVolumes, err := plugin.ListGlobalVolumes()
		if err != nil {
			logger.Error(err, "Could not list global volumes", "pluginName", plugin.GetPluginName())
			continue
		}
		for _, globalVolume := range globalVolumes {
			rc.reconstructGlobalVolume(logger, plugin, globalVolume)
		}
	}
}

func (rc *reconciler) reconstructGlobalVolume(logger klog.Logger, plugin volumepkg.GlobalVolumeListerPlugin, globalVolume volumepkg.GlobalVolume) {
	// Raw block volumes reach the actual state of world through the block
	// mapper rather than through a device mount, so they are not handled here.
	if globalVolume.VolumeMode == v1.PersistentVolumeBlock {
		logger.V(4).Info("Skipping block volume reported as a global mount", "deviceMountPath", globalVolume.DeviceMountPath)
		return
	}

	// The desired state of world names a volume by device when the plugin can
	// device mount it, and by pod otherwise. Only the first kind can be matched
	// against what is found on disk, since there is no pod here to name it
	// with, and unstaging a volume under a name the desired state never
	// produces would unstage one a pod still needs.
	if canMount, err := plugin.CanDeviceMount(globalVolume.Spec); err != nil || !canMount {
		logger.V(4).Info("Skipping global mount of a volume that is not device mountable", "deviceMountPath", globalVolume.DeviceMountPath, "err", err)
		return
	}

	volumeName, err := volumeutil.GetUniqueVolumeNameFromSpec(plugin, globalVolume.Spec)
	if err != nil {
		logger.Error(err, "Could not determine volume name for global mount", "deviceMountPath", globalVolume.DeviceMountPath)
		return
	}
	if rc.actualStateOfWorld.VolumeExists(volumeName) {
		// Another plugin, or another entry from this one, already reported it.
		logger.V(4).Info("Global mount is already in actual state, skipping", "volumeName", volumeName)
		return
	}

	// devicePath is left empty on purpose: it is filled in later from
	// node.status.volumesAttached, the same way reconstructed pod volumes are.
	if err := rc.actualStateOfWorld.AddAttachUncertainReconstructedVolume(
		logger, volumeName, globalVolume.Spec, rc.nodeName, ""); err != nil {
		logger.Error(err, "Could not add global mount to actual state of world", "volumeName", volumeName)
		return
	}
	if err := rc.actualStateOfWorld.MarkDeviceAsUncertain(
		volumeName, "", globalVolume.DeviceMountPath, globalVolume.SELinuxMountContext); err != nil {
		logger.Error(err, "Could not mark global mount device as uncertain", "volumeName", volumeName, "deviceMountPath", globalVolume.DeviceMountPath)
		// Leaving the volume behind would be worse than not having found it:
		// with no pod and no mounted device, the reconciler would take it for a
		// volume to detach, report it detached and drop it, and nothing would
		// unstage the mount until the next kubelet start.
		rc.actualStateOfWorld.DeleteVolume(volumeName)
		return
	}

	rc.volumesNeedUpdateFromNodeStatus = append(rc.volumesNeedUpdateFromNodeStatus, volumeName)
	logger.V(2).Info("Global mount with no pod directory is marked uncertain and added into the actual state", "volumeName", volumeName, "deviceMountPath", globalVolume.DeviceMountPath)
}

func (rc *reconciler) updateStates(logger klog.Logger, reconstructedVolumes map[v1.UniqueVolumeName]*globalVolumeInfo) {
	for _, gvl := range reconstructedVolumes {
		err := rc.actualStateOfWorld.AddAttachUncertainReconstructedVolume(
			//TODO: the devicePath might not be correct for some volume plugins: see issue #54108
			logger, gvl.volumeName, gvl.volumeSpec, rc.nodeName, gvl.devicePath)
		if err != nil {
			logger.Error(err, "Could not add volume information to actual state of world", "volumeName", gvl.volumeName)
			continue
		}
		var seLinuxMountContext string
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
		// If the volume has device to mount, we mark its device as uncertain.
		if gvl.deviceMounter != nil || gvl.blockVolumeMapper != nil {
			deviceMountPath, err := getDeviceMountPath(gvl)
			if err != nil {
				logger.Error(err, "Could not find device mount path for volume", "volumeName", gvl.volumeName)
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
