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
	"reflect"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
)

func (rc *reconciler) Run(stopCh <-chan struct{}) {
	rc.reconstructVolumes()
	klog.InfoS("Reconciler: start to sync state")
	wait.DyanmicPeriodUntil(func() {
		mountedVolumesOldState := rc.actualStateOfWorld.GetAllMountedVolumes()
		rc.reconcile()
		mountedVolumesNewState := rc.actualStateOfWorld.GetAllMountedVolumes()
		// affect the next iteration
		if reflect.DeepEqual(mountedVolumesOldState, mountedVolumesNewState) {
			klog.InfoS("reconciler sleep period moved to initial state * 10")
			// No volume change => increase the sleep duration
			rc.loopSleepDuration = rc.initialLoopSleepDuration * 10
		} else {
			klog.InfoS("reconciler sleep period moved to initial state")
			// Pods have changed => reset to the default duration
			rc.loopSleepDuration = rc.initialLoopSleepDuration
		}
	}, func() time.Duration {
		return rc.loopSleepDuration
	}, 0, true, stopCh)
}

func (rc *reconciler) reconcile() {
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
	rc.mountOrAttachVolumes()

	// Unmount volumes only when DSW and ASW are fully populated to prevent unmounting a volume
	// that is still needed, but it did not reach DSW yet.
	if readyToUnmount {
		// Ensure devices that should be detached/unmounted are detached/unmounted.
		rc.unmountDetachDevices()

		// Clean up any orphan volumes that failed reconstruction.
		rc.cleanOrphanVolumes()
	}

	if len(rc.volumesNeedUpdateFromNodeStatus) != 0 {
		rc.updateReconstructedFromNodeStatus()
	}
	if len(rc.volumesNeedUpdateFromNodeStatus) == 0 {
		// ASW is fully populated only after both devicePaths and uncertain volume attach-ability
		// were reconstructed from the API server.
		// This will start reconciliation of node.status.volumesInUse.
		rc.updateLastSyncTime()
	}
}
