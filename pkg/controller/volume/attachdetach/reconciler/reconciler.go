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
// desired state of the with the actual state of the world by triggering
// actions.
package reconciler

import (
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/statusupdater"
	"k8s.io/kubernetes/pkg/util/goroutinemap/exponentialbackoff"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/volume/util/nestedpendingoperations"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
)

// Reconciler runs a periodic loop to reconcile the desired state of the with
// the actual state of the world by triggering attach detach operations.
// Note: This is distinct from the Reconciler implemented by the kubelet volume
// manager. This reconciles state for the attach/detach controller. That
// reconciles state for the kubelet volume manager.
type Reconciler interface {
	// Starts running the reconciliation loop which executes periodically, checks
	// if volumes that should be attached are attached and volumes that should
	// be detached are detached. If not, it will trigger attach/detach
	// operations to rectify.
	Run(stopCh <-chan struct{})
}

// NewReconciler returns a new instance of Reconciler that waits loopPeriod
// between successive executions.
// loopPeriod is the amount of time the reconciler loop waits between
// successive executions.
// maxWaitForUnmountDuration is the max amount of time the reconciler will wait
// for the volume to be safely unmounted, after this it will detach the volume
// anyway (to handle crashed/unavailable nodes). If during this time the volume
// becomes used by a new pod, the detach request will be aborted and the timer
// cleared.
func NewReconciler(
	loopPeriod time.Duration,
	maxWaitForUnmountDuration time.Duration,
	syncDuration time.Duration,
	disableReconciliationSync bool,
	desiredStateOfWorld cache.DesiredStateOfWorld,
	actualStateOfWorld cache.ActualStateOfWorld,
	attacherDetacher operationexecutor.OperationExecutor,
	nodeStatusUpdater statusupdater.NodeStatusUpdater) Reconciler {
	return &reconciler{
		loopPeriod:                loopPeriod,
		maxWaitForUnmountDuration: maxWaitForUnmountDuration,
		syncDuration:              syncDuration,
		disableReconciliationSync: disableReconciliationSync,
		desiredStateOfWorld:       desiredStateOfWorld,
		actualStateOfWorld:        actualStateOfWorld,
		attacherDetacher:          attacherDetacher,
		nodeStatusUpdater:         nodeStatusUpdater,
		timeOfLastSync:            time.Now(),
	}
}

type reconciler struct {
	loopPeriod                time.Duration
	maxWaitForUnmountDuration time.Duration
	syncDuration              time.Duration
	desiredStateOfWorld       cache.DesiredStateOfWorld
	actualStateOfWorld        cache.ActualStateOfWorld
	attacherDetacher          operationexecutor.OperationExecutor
	nodeStatusUpdater         statusupdater.NodeStatusUpdater
	timeOfLastSync            time.Time
	disableReconciliationSync bool
}

func (rc *reconciler) Run(stopCh <-chan struct{}) {
	wait.Until(rc.reconciliationLoopFunc(), rc.loopPeriod, stopCh)
}

// reconciliationLoopFunc this can be disabled via cli option disableReconciliation.
// It periodically checks whether the attached volumes from actual state
// are still attached to the node and udpate the status if they are not.
func (rc *reconciler) reconciliationLoopFunc() func() {
	return func() {

		rc.reconcile()

		if rc.disableReconciliationSync {
			glog.V(5).Info("Skipping reconciling attached volumes still attached since it is disabled via the command line.")
		} else if rc.syncDuration < time.Second {
			glog.V(5).Info("Skipping reconciling attached volumes still attached since it is set to less than one second via the command line.")
		} else if time.Since(rc.timeOfLastSync) > rc.syncDuration {
			glog.V(5).Info("Starting reconciling attached volumes still attached")
			rc.sync()
		}
	}
}

func (rc *reconciler) sync() {
	defer rc.updateSyncTime()
	rc.syncStates()
}

func (rc *reconciler) updateSyncTime() {
	rc.timeOfLastSync = time.Now()
}

func (rc *reconciler) syncStates() {
	volumesPerNode := rc.actualStateOfWorld.GetAttachedVolumesPerNode()
	for nodeName, volumes := range volumesPerNode {
		err := rc.attacherDetacher.VerifyVolumesAreAttached(volumes, nodeName, rc.actualStateOfWorld)
		if err != nil {
			glog.Errorf("Error in syncing states for volumes: %v", err)
		}
	}
}

func (rc *reconciler) reconcile() {
	// Detaches are triggered before attaches so that volumes referenced by
	// pods that are rescheduled to a different node are detached first.

	// Ensure volumes that should be detached are detached.
	for _, attachedVolume := range rc.actualStateOfWorld.GetAttachedVolumes() {
		if !rc.desiredStateOfWorld.VolumeExists(
			attachedVolume.VolumeName, attachedVolume.NodeName) {
			// Set the detach request time
			elapsedTime, err := rc.actualStateOfWorld.SetDetachRequestTime(attachedVolume.VolumeName, attachedVolume.NodeName)
			if err != nil {
				glog.Errorf("Cannot trigger detach because it fails to set detach request time with error %v", err)
				continue
			}
			// Check whether timeout has reached the maximum waiting time
			timeout := elapsedTime > rc.maxWaitForUnmountDuration
			// Check whether volume is still mounted. Skip detach if it is still mounted unless timeout
			if attachedVolume.MountedByNode && !timeout {
				glog.V(12).Infof("Cannot trigger detach for volume %q on node %q because volume is still mounted",
					attachedVolume.VolumeName,
					attachedVolume.NodeName)
				continue
			}

			// Before triggering volume detach, mark volume as detached and update the node status
			// If it fails to update node status, skip detach volume
			rc.actualStateOfWorld.RemoveVolumeFromReportAsAttached(attachedVolume.VolumeName, attachedVolume.NodeName)

			// Update Node Status to indicate volume is no longer safe to mount.
			err = rc.nodeStatusUpdater.UpdateNodeStatuses()
			if err != nil {
				// Skip detaching this volume if unable to update node status
				glog.Errorf("UpdateNodeStatuses failed while attempting to report volume %q as attached to node %q with: %v ",
					attachedVolume.VolumeName,
					attachedVolume.NodeName,
					err)
				continue
			}

			// Trigger detach volume which requires verifing safe to detach step
			// If timeout is true, skip verifySafeToDetach check
			glog.V(5).Infof("Attempting to start DetachVolume for volume %q from node %q", attachedVolume.VolumeName, attachedVolume.NodeName)
			verifySafeToDetach := !timeout
			err = rc.attacherDetacher.DetachVolume(attachedVolume.AttachedVolume, verifySafeToDetach, rc.actualStateOfWorld)
			if err == nil {
				if !timeout {
					glog.Infof("Started DetachVolume for volume %q from node %q", attachedVolume.VolumeName, attachedVolume.NodeName)
				} else {
					glog.Infof("Started DetachVolume for volume %q from node %q. This volume is not safe to detach, but maxWaitForUnmountDuration %v expired, force detaching",
						attachedVolume.VolumeName,
						attachedVolume.NodeName,
						rc.maxWaitForUnmountDuration)
				}
			}
			if err != nil &&
				!nestedpendingoperations.IsAlreadyExists(err) &&
				!exponentialbackoff.IsExponentialBackoff(err) {
				// Ignore nestedpendingoperations.IsAlreadyExists && exponentialbackoff.IsExponentialBackoff errors, they are expected.
				// Log all other errors.
				glog.Errorf(
					"operationExecutor.DetachVolume failed to start for volume %q (spec.Name: %q) from node %q with err: %v",
					attachedVolume.VolumeName,
					attachedVolume.VolumeSpec.Name(),
					attachedVolume.NodeName,
					err)
			}
		}
	}

	// Ensure volumes that should be attached are attached.
	for _, volumeToAttach := range rc.desiredStateOfWorld.GetVolumesToAttach() {
		if rc.actualStateOfWorld.VolumeNodeExists(
			volumeToAttach.VolumeName, volumeToAttach.NodeName) {
			// Volume/Node exists, touch it to reset detachRequestedTime
			glog.V(5).Infof("Volume %q/Node %q is attached--touching.", volumeToAttach.VolumeName, volumeToAttach.NodeName)
			rc.actualStateOfWorld.ResetDetachRequestTime(volumeToAttach.VolumeName, volumeToAttach.NodeName)
		} else {
			// Volume/Node doesn't exist, spawn a goroutine to attach it
			glog.V(5).Infof("Attempting to start AttachVolume for volume %q to node %q", volumeToAttach.VolumeName, volumeToAttach.NodeName)
			err := rc.attacherDetacher.AttachVolume(volumeToAttach.VolumeToAttach, rc.actualStateOfWorld)
			if err == nil {
				glog.Infof("Started AttachVolume for volume %q to node %q", volumeToAttach.VolumeName, volumeToAttach.NodeName)
			}
			if err != nil &&
				!nestedpendingoperations.IsAlreadyExists(err) &&
				!exponentialbackoff.IsExponentialBackoff(err) {
				// Ignore nestedpendingoperations.IsAlreadyExists && exponentialbackoff.IsExponentialBackoff errors, they are expected.
				// Log all other errors.
				glog.Errorf(
					"operationExecutor.AttachVolume failed to start for volume %q (spec.Name: %q) to node %q with err: %v",
					volumeToAttach.VolumeName,
					volumeToAttach.VolumeSpec.Name(),
					volumeToAttach.NodeName,
					err)
			}
		}
	}

	// Update Node Status
	err := rc.nodeStatusUpdater.UpdateNodeStatuses()
	if err != nil {
		glog.Infof("UpdateNodeStatuses failed with: %v", err)
	}
}
