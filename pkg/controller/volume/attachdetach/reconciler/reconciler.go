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
	"fmt"
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/statusupdater"
	kevents "k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/util/goroutinemap/exponentialbackoff"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
)

// Reconciler runs a periodic loop to reconcile the desired state of the world with
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
	nodeStatusUpdater statusupdater.NodeStatusUpdater,
	recorder record.EventRecorder) Reconciler {
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
		recorder:                  recorder,
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
	recorder                  record.EventRecorder
}

func (rc *reconciler) Run(stopCh <-chan struct{}) {
	wait.Until(rc.reconciliationLoopFunc(), rc.loopPeriod, stopCh)
}

// reconciliationLoopFunc this can be disabled via cli option disableReconciliation.
// It periodically checks whether the attached volumes from actual state
// are still attached to the node and update the status if they are not.
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
	rc.attacherDetacher.VerifyVolumesAreAttached(volumesPerNode, rc.actualStateOfWorld)
}

// isMultiAttachForbidden checks if attaching this volume to multiple nodes is definitely not allowed/possible.
// In its current form, this function can only reliably say for which volumes it's definitely forbidden. If it returns
// false, it is not guaranteed that multi-attach is actually supported by the volume type and we must rely on the
// attacher to fail fast in such cases.
// Please see https://github.com/kubernetes/kubernetes/issues/40669 and https://github.com/kubernetes/kubernetes/pull/40148#discussion_r98055047
func (rc *reconciler) isMultiAttachForbidden(volumeSpec *volume.Spec) bool {
	if volumeSpec.Volume != nil {
		// Check for volume types which are known to fail slow or cause trouble when trying to multi-attach
		if volumeSpec.Volume.AzureDisk != nil ||
			volumeSpec.Volume.Cinder != nil {
			return true
		}
	}

	// Only if this volume is a persistent volume, we have reliable information on wether it's allowed or not to
	// multi-attach. We trust in the individual volume implementations to not allow unsupported access modes
	if volumeSpec.PersistentVolume != nil {
		// Check for persistent volume types which do not fail when trying to multi-attach
		if volumeSpec.PersistentVolume.Spec.VsphereVolume != nil {
			return false
		}

		if len(volumeSpec.PersistentVolume.Spec.AccessModes) == 0 {
			// No access mode specified so we don't know for sure. Let the attacher fail if needed
			return false
		}

		// check if this volume is allowed to be attached to multiple PODs/nodes, if yes, return false
		for _, accessMode := range volumeSpec.PersistentVolume.Spec.AccessModes {
			if accessMode == v1.ReadWriteMany || accessMode == v1.ReadOnlyMany {
				return false
			}
		}
		return true
	}

	// we don't know if it's supported or not and let the attacher fail later in cases it's not supported
	return false
}

func (rc *reconciler) reconcile() {
	// Detaches are triggered before attaches so that volumes referenced by
	// pods that are rescheduled to a different node are detached first.

	// Ensure volumes that should be detached are detached.
	for _, attachedVolume := range rc.actualStateOfWorld.GetAttachedVolumes() {
		if !rc.desiredStateOfWorld.VolumeExists(
			attachedVolume.VolumeName, attachedVolume.NodeName) {

			// Don't even try to start an operation if there is already one running
			// This check must be done before we do any other checks, as otherwise the other checks
			// may pass while at the same time the volume leaves the pending state, resulting in
			// double detach attempts
			if rc.attacherDetacher.IsOperationPending(attachedVolume.VolumeName, "") {
				glog.V(10).Infof("Operation for volume %q is already running. Can't start detach for %q", attachedVolume.VolumeName, attachedVolume.NodeName)
				continue
			}

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
				glog.V(12).Infof(attachedVolume.GenerateMsgDetailed("Cannot detach volume because it is still mounted", ""))
				continue
			}

			// Before triggering volume detach, mark volume as detached and update the node status
			// If it fails to update node status, skip detach volume
			err = rc.actualStateOfWorld.RemoveVolumeFromReportAsAttached(attachedVolume.VolumeName, attachedVolume.NodeName)
			if err != nil {
				glog.V(5).Infof("RemoveVolumeFromReportAsAttached failed while removing volume %q from node %q with: %v",
					attachedVolume.VolumeName,
					attachedVolume.NodeName,
					err)
			}

			// Update Node Status to indicate volume is no longer safe to mount.
			err = rc.nodeStatusUpdater.UpdateNodeStatuses()
			if err != nil {
				// Skip detaching this volume if unable to update node status
				glog.Errorf(attachedVolume.GenerateErrorDetailed("UpdateNodeStatuses failed while attempting to report volume as attached", err).Error())
				continue
			}

			// Trigger detach volume which requires verifing safe to detach step
			// If timeout is true, skip verifySafeToDetach check
			glog.V(5).Infof(attachedVolume.GenerateMsgDetailed("Starting attacherDetacher.DetachVolume", ""))
			verifySafeToDetach := !timeout
			err = rc.attacherDetacher.DetachVolume(attachedVolume.AttachedVolume, verifySafeToDetach, rc.actualStateOfWorld)
			if err == nil {
				if !timeout {
					glog.Infof(attachedVolume.GenerateMsgDetailed("attacherDetacher.DetachVolume started", ""))
				} else {
					glog.Warningf(attachedVolume.GenerateMsgDetailed("attacherDetacher.DetachVolume started", fmt.Sprintf("This volume is not safe to detach, but maxWaitForUnmountDuration %v expired, force detaching", rc.maxWaitForUnmountDuration)))
				}
			}
			if err != nil && !exponentialbackoff.IsExponentialBackoff(err) {
				// Ignore exponentialbackoff.IsExponentialBackoff errors, they are expected.
				// Log all other errors.
				glog.Errorf(attachedVolume.GenerateErrorDetailed("attacherDetacher.DetachVolume failed to start", err).Error())
			}
		}
	}

	// Ensure volumes that should be attached are attached.
	for _, volumeToAttach := range rc.desiredStateOfWorld.GetVolumesToAttach() {
		if rc.actualStateOfWorld.VolumeNodeExists(
			volumeToAttach.VolumeName, volumeToAttach.NodeName) {
			// Volume/Node exists, touch it to reset detachRequestedTime
			glog.V(5).Infof(volumeToAttach.GenerateMsgDetailed("Volume attached--touching", ""))
			rc.actualStateOfWorld.ResetDetachRequestTime(volumeToAttach.VolumeName, volumeToAttach.NodeName)
		} else {
			// Don't even try to start an operation if there is already one running
			if rc.attacherDetacher.IsOperationPending(volumeToAttach.VolumeName, "") {
				glog.V(10).Infof("Operation for volume %q is already running. Can't start attach for %q", volumeToAttach.VolumeName, volumeToAttach.NodeName)
				continue
			}

			if rc.isMultiAttachForbidden(volumeToAttach.VolumeSpec) {
				nodes := rc.actualStateOfWorld.GetNodesForVolume(volumeToAttach.VolumeName)
				if len(nodes) > 0 {
					if !volumeToAttach.MultiAttachErrorReported {
						simpleMsg, detailedMsg := volumeToAttach.GenerateMsg("Multi-Attach error", "Volume is already exclusively attached to one node and can't be attached to another")
						for _, pod := range volumeToAttach.ScheduledPods {
							rc.recorder.Eventf(pod, v1.EventTypeWarning, kevents.FailedAttachVolume, simpleMsg)
						}
						volumeToAttach.MultiAttachErrorReported = true
						glog.Warningf(detailedMsg)
					}
					continue
				}
			}

			// Volume/Node doesn't exist, spawn a goroutine to attach it
			glog.V(5).Infof(volumeToAttach.GenerateMsgDetailed("Starting attacherDetacher.AttachVolume", ""))
			err := rc.attacherDetacher.AttachVolume(volumeToAttach.VolumeToAttach, rc.actualStateOfWorld)
			if err == nil {
				glog.Infof(volumeToAttach.GenerateMsgDetailed("attacherDetacher.AttachVolume started", ""))
			}
			if err != nil && !exponentialbackoff.IsExponentialBackoff(err) {
				// Ignore exponentialbackoff.IsExponentialBackoff errors, they are expected.
				// Log all other errors.
				glog.Errorf(volumeToAttach.GenerateErrorDetailed("attacherDetacher.AttachVolume failed to start", err).Error())
			}
		}
	}

	// Update Node Status
	err := rc.nodeStatusUpdater.UpdateNodeStatuses()
	if err != nil {
		glog.Warningf("UpdateNodeStatuses failed with: %v", err)
	}
}
