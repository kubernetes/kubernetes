/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
// relevant actions (attach, detach, mount, unmount).
package reconciler

import (
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/kubelet/volume/cache"
	"k8s.io/kubernetes/pkg/util/goroutinemap"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
)

// Reconciler runs a periodic loop to reconcile the desired state of the world
// with the actual state of the world by triggering attach, detach, mount, and
// unmount operations.
// Note: This is distinct from the Reconciler implemented by the attach/detach
// controller. This reconciles state for the kubelet volume manager. That
// reconciles state for the attach/detach controller.
type Reconciler interface {
	// Starts running the reconciliation loop which executes periodically, checks
	// if volumes that should be mounted are mounted and volumes that should
	// be unmounted are unmounted. If not, it will trigger mount/unmount
	// operations to rectify.
	// If attach/detach management is enabled, the manager will also check if
	// volumes that should be attached are attached and volumes that should
	// be detached are detached and trigger attach/detach operations as needed.
	Run(stopCh <-chan struct{})
}

// NewReconciler returns a new instance of Reconciler.
//
// controllerAttachDetachEnabled - if true, indicates that the attach/detach
//   controller is responsible for managing the attach/detach operations for
//   this node, and therefore the volume manager should not
// loopSleepDuration - the amount of time the reconciler loop sleeps between
//   successive executions
// waitForAttachTimeout - the amount of time the Mount function will wait for
//   the volume to be attached
// hostName - the hostname for this node, used by Attach and Detach methods
// desiredStateOfWorld - cache containing the desired state of the world
// actualStateOfWorld - cache containing the actual state of the world
// operationExecutor - used to trigger attach/detach/mount/unmount operations
//   safely (prevents more than one operation from being triggered on the same
//   volume)
func NewReconciler(
	controllerAttachDetachEnabled bool,
	loopSleepDuration time.Duration,
	waitForAttachTimeout time.Duration,
	hostName string,
	desiredStateOfWorld cache.DesiredStateOfWorld,
	actualStateOfWorld cache.ActualStateOfWorld,
	operationExecutor operationexecutor.OperationExecutor) Reconciler {
	return &reconciler{
		controllerAttachDetachEnabled: controllerAttachDetachEnabled,
		loopSleepDuration:             loopSleepDuration,
		waitForAttachTimeout:          waitForAttachTimeout,
		hostName:                      hostName,
		desiredStateOfWorld:           desiredStateOfWorld,
		actualStateOfWorld:            actualStateOfWorld,
		operationExecutor:             operationExecutor,
	}
}

type reconciler struct {
	controllerAttachDetachEnabled bool
	loopSleepDuration             time.Duration
	waitForAttachTimeout          time.Duration
	hostName                      string
	desiredStateOfWorld           cache.DesiredStateOfWorld
	actualStateOfWorld            cache.ActualStateOfWorld
	operationExecutor             operationexecutor.OperationExecutor
}

func (rc *reconciler) Run(stopCh <-chan struct{}) {
	wait.Until(rc.reconciliationLoopFunc(), rc.loopSleepDuration, stopCh)
}

func (rc *reconciler) reconciliationLoopFunc() func() {
	return func() {
		// Unmounts are triggered before mounts so that a volume that was
		// referenced by a pod that was deleted and is now referenced by another
		// pod is unmounted from the first pod before being mounted to the new
		// pod.

		// Ensure volumes that should be unmounted are unmounted.
		for _, mountedVolume := range rc.actualStateOfWorld.GetMountedVolumes() {
			if !rc.desiredStateOfWorld.PodExistsInVolume(mountedVolume.PodName, mountedVolume.VolumeName) {
				// Volume is mounted, unmount it
				glog.V(12).Infof("Attempting to start UnmountVolume for volume %q (spec.Name: %q) from pod %q (UID: %q).",
					mountedVolume.VolumeName,
					mountedVolume.OuterVolumeSpecName,
					mountedVolume.PodName,
					mountedVolume.PodUID)
				err := rc.operationExecutor.UnmountVolume(
					mountedVolume.MountedVolume, rc.actualStateOfWorld)
				if err != nil && !goroutinemap.IsAlreadyExists(err) {
					// Ignore goroutinemap.IsAlreadyExists errors, they are expected.
					// Log all other errors.
					glog.Errorf(
						"operationExecutor.UnmountVolume failed for volume %q (spec.Name: %q) pod %q (UID: %q) controllerAttachDetachEnabled: %v with err: %v",
						mountedVolume.VolumeName,
						mountedVolume.OuterVolumeSpecName,
						mountedVolume.PodName,
						mountedVolume.PodUID,
						rc.controllerAttachDetachEnabled,
						err)
				}
				if err == nil {
					glog.Infof("UnmountVolume operation started for volume %q (spec.Name: %q) from pod %q (UID: %q).",
						mountedVolume.VolumeName,
						mountedVolume.OuterVolumeSpecName,
						mountedVolume.PodName,
						mountedVolume.PodUID)
				}
			}
		}

		// Ensure volumes that should be attached/mounted are attached/mounted.
		for _, volumeToMount := range rc.desiredStateOfWorld.GetVolumesToMount() {
			volMounted, err := rc.actualStateOfWorld.PodExistsInVolume(volumeToMount.PodName, volumeToMount.VolumeName)
			if cache.IsVolumeNotAttachedError(err) {
				// Volume is not attached, it should be
				if rc.controllerAttachDetachEnabled || !volumeToMount.PluginIsAttachable {
					// Kubelet not responsible for attaching or this volume has a non-attachable volume plugin,
					// so just add it to actualStateOfWorld without attach.
					markVolumeAttachErr := rc.actualStateOfWorld.MarkVolumeAsAttached(
						volumeToMount.VolumeSpec, rc.hostName)
					if markVolumeAttachErr != nil {
						glog.Errorf(
							"actualStateOfWorld.MarkVolumeAsAttached failed for volume %q (spec.Name: %q) pod %q (UID: %q) controllerAttachDetachEnabled: %v with err: %v",
							volumeToMount.VolumeName,
							volumeToMount.VolumeSpec.Name(),
							volumeToMount.PodName,
							volumeToMount.Pod.UID,
							rc.controllerAttachDetachEnabled,
							markVolumeAttachErr)
					} else {
						glog.V(12).Infof("actualStateOfWorld.MarkVolumeAsAttached succeeded for volume %q (spec.Name: %q) pod %q (UID: %q)",
							volumeToMount.VolumeName,
							volumeToMount.VolumeSpec.Name(),
							volumeToMount.PodName,
							volumeToMount.Pod.UID)
					}
				} else {
					// Volume is not attached to node, kubelet attach is enabled, volume implements an attacher,
					// so attach it
					volumeToAttach := operationexecutor.VolumeToAttach{
						VolumeName: volumeToMount.VolumeName,
						VolumeSpec: volumeToMount.VolumeSpec,
						NodeName:   rc.hostName,
					}
					glog.V(12).Infof("Attempting to start AttachVolume for volume %q (spec.Name: %q)  pod %q (UID: %q)",
						volumeToMount.VolumeName,
						volumeToMount.VolumeSpec.Name(),
						volumeToMount.PodName,
						volumeToMount.Pod.UID)
					err := rc.operationExecutor.AttachVolume(volumeToAttach, rc.actualStateOfWorld)
					if err != nil && !goroutinemap.IsAlreadyExists(err) {
						// Ignore goroutinemap.IsAlreadyExists errors, they are expected.
						// Log all other errors.
						glog.Errorf(
							"operationExecutor.AttachVolume failed for volume %q (spec.Name: %q) pod %q (UID: %q) controllerAttachDetachEnabled: %v with err: %v",
							volumeToMount.VolumeName,
							volumeToMount.VolumeSpec.Name(),
							volumeToMount.PodName,
							volumeToMount.Pod.UID,
							rc.controllerAttachDetachEnabled,
							err)
					}
					if err == nil {
						glog.Infof("AttachVolume operation started for volume %q (spec.Name: %q) pod %q (UID: %q)",
							volumeToMount.VolumeName,
							volumeToMount.VolumeSpec.Name(),
							volumeToMount.PodName,
							volumeToMount.Pod.UID)
					}
				}
			} else if !volMounted || cache.IsRemountRequiredError(err) {
				// Volume is not mounted, or is already mounted, but requires remounting
				remountingLogStr := ""
				if cache.IsRemountRequiredError(err) {
					remountingLogStr = "Volume is already mounted to pod, but remount was requested."
				}
				glog.V(12).Infof("Attempting to start MountVolume for volume %q (spec.Name: %q) to pod %q (UID: %q). %s",
					volumeToMount.VolumeName,
					volumeToMount.VolumeSpec.Name(),
					volumeToMount.PodName,
					volumeToMount.Pod.UID,
					remountingLogStr)
				err := rc.operationExecutor.MountVolume(
					rc.waitForAttachTimeout,
					volumeToMount.VolumeToMount,
					rc.actualStateOfWorld)
				if err != nil && !goroutinemap.IsAlreadyExists(err) {
					// Ignore goroutinemap.IsAlreadyExists errors, they are expected.
					// Log all other errors.
					glog.Errorf(
						"operationExecutor.MountVolume failed for volume %q (spec.Name: %q) pod %q (UID: %q) controllerAttachDetachEnabled: %v with err: %v",
						volumeToMount.VolumeName,
						volumeToMount.VolumeSpec.Name(),
						volumeToMount.PodName,
						volumeToMount.Pod.UID,
						rc.controllerAttachDetachEnabled,
						err)
				}
				if err == nil {
					glog.Infof("MountVolume operation started for volume %q (spec.Name: %q) to pod %q (UID: %q). %s",
						volumeToMount.VolumeName,
						volumeToMount.VolumeSpec.Name(),
						volumeToMount.PodName,
						volumeToMount.Pod.UID,
						remountingLogStr)
				}
			}
		}

		// Ensure devices that should be detached/unmounted are detached/unmounted.
		for _, attachedVolume := range rc.actualStateOfWorld.GetUnmountedVolumes() {
			if !rc.desiredStateOfWorld.VolumeExists(attachedVolume.VolumeName) {
				if attachedVolume.GloballyMounted {
					// Volume is globally mounted to device, unmount it
					glog.V(12).Infof("Attempting to start UnmountDevice for volume %q (spec.Name: %q)",
						attachedVolume.VolumeName,
						attachedVolume.VolumeSpec.Name())
					err := rc.operationExecutor.UnmountDevice(
						attachedVolume.AttachedVolume, rc.actualStateOfWorld)
					if err != nil && !goroutinemap.IsAlreadyExists(err) {
						// Ignore goroutinemap.IsAlreadyExists errors, they are expected.
						// Log all other errors.
						glog.Errorf(
							"operationExecutor.UnmountDevice failed for volume %q (spec.Name: %q) controllerAttachDetachEnabled: %v with err: %v",
							attachedVolume.VolumeName,
							attachedVolume.VolumeSpec.Name(),
							rc.controllerAttachDetachEnabled,
							err)
					}
					if err == nil {
						glog.Infof("UnmountDevice operation started for volume %q (spec.Name: %q)",
							attachedVolume.VolumeName,
							attachedVolume.VolumeSpec.Name())
					}
				} else {
					// Volume is attached to node, detach it
					if rc.controllerAttachDetachEnabled || !attachedVolume.PluginIsAttachable {
						// Kubelet not responsible for detaching or this volume has a non-attachable volume plugin,
						// so just remove it to actualStateOfWorld without attach.
						rc.actualStateOfWorld.MarkVolumeAsDetached(
							attachedVolume.VolumeName, rc.hostName)
					} else {
						// Only detach if kubelet detach is enabled
						glog.V(12).Infof("Attempting to start DetachVolume for volume %q (spec.Name: %q)",
							attachedVolume.VolumeName,
							attachedVolume.VolumeSpec.Name())
						err := rc.operationExecutor.DetachVolume(
							attachedVolume.AttachedVolume, rc.actualStateOfWorld)
						if err != nil && !goroutinemap.IsAlreadyExists(err) {
							// Ignore goroutinemap.IsAlreadyExists errors, they are expected.
							// Log all other errors.
							glog.Errorf(
								"operationExecutor.DetachVolume failed for volume %q (spec.Name: %q) controllerAttachDetachEnabled: %v with err: %v",
								attachedVolume.VolumeName,
								attachedVolume.VolumeSpec.Name(),
								rc.controllerAttachDetachEnabled,
								err)
						}
						if err == nil {
							glog.Infof("DetachVolume operation started for volume %q (spec.Name: %q)",
								attachedVolume.VolumeName,
								attachedVolume.VolumeSpec.Name())
						}
					}
				}
			}
		}
	}
}
