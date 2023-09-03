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

// Package reconciler implements interfaces that attempt to reconcile the
// desired state of the world with the actual state of the world by triggering
// relevant actions (attach, detach, mount, unmount).
package reconciler

import (
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
)

func (rc *reconciler) runOld(stopCh <-chan struct{}) {
	wait.Until(rc.reconciliationLoopFunc(), rc.loopSleepDuration, stopCh)
}

func (rc *reconciler) reconciliationLoopFunc() func() {
	return func() {
		rc.reconcile()

		// Sync the state with the reality once after all existing pods are added to the desired state from all sources.
		// Otherwise, the reconstruct process may clean up pods' volumes that are still in use because
		// desired state of world does not contain a complete list of pods.
		if rc.populatorHasAddedPods() && !rc.StatesHasBeenSynced() {
			klog.InfoS("Reconciler: start to sync state")
			rc.sync()
		}
	}
}

func (rc *reconciler) reconcile() {
	// Unmounts are triggered before mounts so that a volume that was
	// referenced by a pod that was deleted and is now referenced by another
	// pod is unmounted from the first pod before being mounted to the new
	// pod.
	rc.unmountVolumes()

	// Next we mount required volumes. This function could also trigger
	// attach if kubelet is responsible for attaching volumes.
	// If underlying PVC was resized while in-use then this function also handles volume
	// resizing.
	rc.mountOrAttachVolumes()

	// Ensure devices that should be detached/unmounted are detached/unmounted.
	rc.unmountDetachDevices()

	// After running the above operations if skippedDuringReconstruction is not empty
	// then ensure that all volumes which were discovered and skipped during reconstruction
	// are added to actualStateOfWorld in uncertain state.
	if len(rc.skippedDuringReconstruction) > 0 {
		rc.processReconstructedVolumes()
	}
}

// processReconstructedVolumes checks volumes which were skipped during the reconstruction
// process because it was assumed that since these volumes were present in DSOW they would get
// mounted correctly and make it into ASOW.
// But if mount operation fails for some reason then we still need to mark the volume as uncertain
// and wait for the next reconciliation loop to deal with it.
func (rc *reconciler) processReconstructedVolumes() {
	for volumeName, glblVolumeInfo := range rc.skippedDuringReconstruction {
		// check if volume is marked as attached to the node
		// for now lets only process volumes which are at least known as attached to the node
		// this should help with most volume types (including secret, configmap etc)
		if !rc.actualStateOfWorld.VolumeExists(volumeName) {
			klog.V(4).InfoS("Volume is not marked as attached to the node. Skipping processing of the volume", "volumeName", volumeName)
			continue
		}
		uncertainVolumeCount := 0
		// only delete volumes which were marked as attached here.
		// This should ensure that  - we will wait for volumes which were not marked as attached
		// before adding them in uncertain state during reconstruction.
		delete(rc.skippedDuringReconstruction, volumeName)

		for podName, volume := range glblVolumeInfo.podVolumes {
			markVolumeOpts := operationexecutor.MarkVolumeOpts{
				PodName:             volume.podName,
				PodUID:              types.UID(podName),
				VolumeName:          volume.volumeName,
				Mounter:             volume.mounter,
				BlockVolumeMapper:   volume.blockVolumeMapper,
				OuterVolumeSpecName: volume.outerVolumeSpecName,
				VolumeGidVolume:     volume.volumeGidValue,
				VolumeSpec:          volume.volumeSpec,
				VolumeMountState:    operationexecutor.VolumeMountUncertain,
			}

			volumeAdded, err := rc.actualStateOfWorld.CheckAndMarkVolumeAsUncertainViaReconstruction(markVolumeOpts)

			// if volume is not mounted then lets mark volume mounted in uncertain state in ASOW
			if volumeAdded {
				uncertainVolumeCount += 1
				if err != nil {
					klog.ErrorS(err, "Could not add pod to volume information to actual state of world", "pod", klog.KObj(volume.pod))
					continue
				}
				klog.V(4).InfoS("Volume is marked as mounted in uncertain state and added to the actual state", "pod", klog.KObj(volume.pod), "podName", volume.podName, "volumeName", volume.volumeName)
			}
		}

		if uncertainVolumeCount > 0 {
			// If the volume has device to mount, we mark its device as uncertain
			if glblVolumeInfo.deviceMounter != nil || glblVolumeInfo.blockVolumeMapper != nil {
				deviceMountPath, err := getDeviceMountPath(glblVolumeInfo)
				if err != nil {
					klog.ErrorS(err, "Could not find device mount path for volume", "volumeName", glblVolumeInfo.volumeName)
					continue
				}
				deviceMounted := rc.actualStateOfWorld.CheckAndMarkDeviceUncertainViaReconstruction(glblVolumeInfo.volumeName, deviceMountPath)
				if !deviceMounted {
					klog.V(3).InfoS("Could not mark device as mounted in uncertain state", "volumeName", glblVolumeInfo.volumeName)
				}
			}
		}
	}
}
