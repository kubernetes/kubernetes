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
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
)

// TODO: move to reconstruct.go and remove old code there.

// TODO: Replace Run() when SELinuxMountReadWriteOncePod is GA
func (rc *reconciler) runNew(stopCh <-chan struct{}) {
	rc.reconstructVolumes()
	klog.InfoS("Reconciler: start to sync state")
	wait.Until(rc.reconcileNew, rc.loopSleepDuration, stopCh)
}

func (rc *reconciler) reconcileNew() {
	readyToUnmount := rc.readyToUnmount()
	if readyToUnmount {
		// Unmounts are triggered before mounts so that a volume that was
		// referenced by a pod that was deleted and is now referenced by another
		// pod is unmounted from the first pod before being mounted to the new
		// pod.
		rc.unmountVolumes()
	}

	// Next we mount required volumes. This function could also trigger
	// attach if kubelet is responsible for attaching volumes.
	// If underlying PVC was resized while in-use then this function also handles volume
	// resizing.
	rc.mountAttachVolumes()

	// Unmount volumes only when DSW and ASW are fully populated to prevent unmounting a volume
	// that is still needed, but it did not reach DSW yet.
	if readyToUnmount {
		// Ensure devices that should be detached/unmounted are detached/unmounted.
		rc.unmountDetachDevices()

		// Clean up any orphan volumes that failed reconstruction.
		rc.cleanOrphanVolumes()
	}

	if len(rc.volumesNeedDevicePath) != 0 {
		rc.updateReconstructedDevicePaths()
	}

	if len(rc.volumesNeedReportedInUse) != 0 && rc.populatorHasAddedPods() {
		// Once DSW is populated, mark all reconstructed as reported in node.status,
		// so they can proceed with MountDevice / SetUp.
		rc.desiredStateOfWorld.MarkVolumesReportedInUse(rc.volumesNeedReportedInUse)
		rc.volumesNeedReportedInUse = nil
	}
}

func (rc *reconciler) readyToUnmount() bool {
	// Allow unmount only when DSW is fully populated to prevent unmounting volumes that
	// did not reach DSW yet.
	if !rc.populatorHasAddedPods() {
		return false
	}

	// Allow unmount only when ASW device paths were corrected from node.status to prevent
	// calling unmount with a wrong devicePath.
	if len(rc.volumesNeedDevicePath) != 0 {
		return false
	}
	return true
}

// reconstructVolumes tries to reconstruct the actual state of world by scanning all pods' volume
// directories from the disk. For the volumes that cannot support or fail reconstruction, it will
// put the volumes to volumesFailedReconstruction to be cleaned up later when DesiredStateOfWorld
// is populated.
func (rc *reconciler) reconstructVolumes() {
	defer rc.updateLastSyncTime()
	// Get volumes information by reading the pod's directory
	podVolumes, err := getVolumesFromPodDir(rc.kubeletPodsDir)
	if err != nil {
		klog.ErrorS(err, "Cannot get volumes from disk, skip sync states for volume reconstruction")
		return
	}
	reconstructedVolumes := make(map[v1.UniqueVolumeName]*reconstructedVolume)
	reconstructedVolumeNames := []v1.UniqueVolumeName{}
	for _, volume := range podVolumes {
		if rc.actualStateOfWorld.VolumeExistsWithSpecName(volume.podName, volume.volumeSpecName) {
			klog.V(4).InfoS("Volume exists in actual state, skip cleaning up mounts", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName)
			// There is nothing to reconstruct
			continue
		}
		reconstructedVolume, err := rc.reconstructVolume(volume)
		if err != nil {
			klog.InfoS("Could not construct volume information", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName, "err", err)
			// We can't reconstruct the volume. Remember to check DSW after it's fully populated and force unmount the volume when it's orphaned.
			rc.volumesFailedReconstruction = append(rc.volumesFailedReconstruction, volume)
			continue
		}
		klog.V(4).InfoS("Adding reconstructed volume to actual state and node status", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName)
		reconstructedVolumeNames = append(reconstructedVolumeNames, reconstructedVolume.volumeName)
		reconstructedVolumes[reconstructedVolume.volumeName] = reconstructedVolume
	}

	if len(reconstructedVolumes) > 0 {
		// Add the volumes to ASW
		rc.updateStatesNew(reconstructedVolumes)
		// The reconstructed volumes are mounted, hence a previous kubelet must have already put it into node.status.volumesInUse.
		// Remember to update DSW with this information.
		rc.volumesNeedReportedInUse = reconstructedVolumeNames
		// Remember to update devicePath from node.status.volumesAttached
		rc.volumesNeedDevicePath = reconstructedVolumeNames
	}
	klog.V(2).InfoS("Volume reconstruction finished")
}

func (rc *reconciler) updateStatesNew(reconstructedVolumes map[v1.UniqueVolumeName]*reconstructedVolume) {
	for _, volume := range reconstructedVolumes {
		err := rc.actualStateOfWorld.MarkVolumeAsAttached(
			//TODO: the devicePath might not be correct for some volume plugins: see issue #54108
			volume.volumeName, volume.volumeSpec, "" /* nodeName */, volume.devicePath)
		if err != nil {
			klog.ErrorS(err, "Could not add volume information to actual state of world", "pod", klog.KObj(volume.pod))
			continue
		}
		markVolumeOpts := operationexecutor.MarkVolumeOpts{
			PodName:             volume.podName,
			PodUID:              types.UID(volume.podName),
			VolumeName:          volume.volumeName,
			Mounter:             volume.mounter,
			BlockVolumeMapper:   volume.blockVolumeMapper,
			OuterVolumeSpecName: volume.outerVolumeSpecName,
			VolumeGidVolume:     volume.volumeGidValue,
			VolumeSpec:          volume.volumeSpec,
			VolumeMountState:    operationexecutor.VolumeMountUncertain,
		}
		err = rc.actualStateOfWorld.MarkVolumeMountAsUncertain(markVolumeOpts)
		if err != nil {
			klog.ErrorS(err, "Could not add pod to volume information to actual state of world", "pod", klog.KObj(volume.pod))
			continue
		}
		klog.V(4).InfoS("Volume is marked as uncertain and added into the actual state", "pod", klog.KObj(volume.pod), "podName", volume.podName, "volumeName", volume.volumeName)
		// If the volume has device to mount, we mark its device as mounted.
		if volume.deviceMounter != nil || volume.blockVolumeMapper != nil {
			deviceMountPath, err := getDeviceMountPath(volume)
			if err != nil {
				klog.ErrorS(err, "Could not find device mount path for volume", "volumeName", volume.volumeName, "pod", klog.KObj(volume.pod))
				continue
			}
			err = rc.actualStateOfWorld.MarkDeviceAsUncertain(volume.volumeName, volume.devicePath, deviceMountPath)
			if err != nil {
				klog.ErrorS(err, "Could not mark device is uncertain to actual state of world", "pod", klog.KObj(volume.pod))
				continue
			}
			klog.V(4).InfoS("Volume is marked device as uncertain and added into the actual state", "pod", klog.KObj(volume.pod), "podName", volume.podName, "volumeName", volume.volumeName)
		}
	}
}

// cleanOrphanVolumes tries to clean up all volumes that failed reconstruction.
func (rc *reconciler) cleanOrphanVolumes() {
	if len(rc.volumesFailedReconstruction) == 0 {
		return
	}

	for _, volume := range rc.volumesFailedReconstruction {
		if rc.desiredStateOfWorld.VolumeExistsWithSpecName(volume.podName, volume.volumeSpecName) {
			// Some pod needs the volume, don't clean it up and hope that
			// reconcile() calls SetUp and reconstructs the volume in ASW.
			klog.V(4).InfoS("Volume exists in desired state, skip cleaning up mounts", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName)
			continue
		}
		klog.InfoS("Cleaning up mounts for volume that could not be reconstructed", "podName", volume.podName, "volumeSpecName", volume.volumeSpecName)
		rc.cleanupMounts(volume)
	}

	klog.V(2).InfoS("Orphan volume cleanup finished")
	// Clean the cache, cleanup is one shot operation.
	rc.volumesFailedReconstruction = make([]podVolume, 0)
}

// updateReconstructedDevicePaths tries to file devicePaths of reconstructed volumes from
// node.Status.VolumesAttached. This can be done only after connection to the API
// server is established, i.e. it can't be part of reconstructVolumes().
func (rc *reconciler) updateReconstructedDevicePaths() {
	klog.V(4).InfoS("Updating reconstructed devicePaths")

	node, fetchErr := rc.kubeClient.CoreV1().Nodes().Get(context.TODO(), string(rc.nodeName), metav1.GetOptions{})
	if fetchErr != nil {
		// This may repeat few times per second until kubelet is able to read its own status for the first time.
		klog.ErrorS(fetchErr, "Failed to get Node status to reconstruct device paths")
		return
	}

	for _, volumeID := range rc.volumesNeedDevicePath {
		for _, attachedVolume := range node.Status.VolumesAttached {
			if volumeID != attachedVolume.Name {
				continue
			}
			rc.actualStateOfWorld.UpdateReconstructedDevicePath(volumeID, attachedVolume.DevicePath)
			klog.V(4).InfoS("Updated devicePath from node status for volume", "volumeName", attachedVolume.Name, "path", attachedVolume.DevicePath)
		}
	}
	klog.V(2).InfoS("DevicePaths of reconstructed volumes updated")
	rc.volumesNeedDevicePath = nil
}
