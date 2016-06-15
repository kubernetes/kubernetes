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
// actions.
package reconciler

import (
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/controller/volume/cache"
	"k8s.io/kubernetes/pkg/util/wait"
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
	desiredStateOfWorld cache.DesiredStateOfWorld,
	actualStateOfWorld cache.ActualStateOfWorld,
	attacherDetacher operationexecutor.OperationExecutor) Reconciler {
	return &reconciler{
		loopPeriod:                loopPeriod,
		maxWaitForUnmountDuration: maxWaitForUnmountDuration,
		desiredStateOfWorld:       desiredStateOfWorld,
		actualStateOfWorld:        actualStateOfWorld,
		attacherDetacher:          attacherDetacher,
	}
}

type reconciler struct {
	loopPeriod                time.Duration
	maxWaitForUnmountDuration time.Duration
	desiredStateOfWorld       cache.DesiredStateOfWorld
	actualStateOfWorld        cache.ActualStateOfWorld
	attacherDetacher          operationexecutor.OperationExecutor
}

func (rc *reconciler) Run(stopCh <-chan struct{}) {
	wait.Until(rc.reconciliationLoopFunc(), rc.loopPeriod, stopCh)
}

func (rc *reconciler) reconciliationLoopFunc() func() {
	return func() {
		// Detaches are triggered before attaches so that volumes referenced by
		// pods that are rescheduled to a different node are detached first.

		// Ensure volumes that should be detached are detached.
		for _, attachedVolume := range rc.actualStateOfWorld.GetAttachedVolumes() {
			if !rc.desiredStateOfWorld.VolumeExists(
				attachedVolume.VolumeName, attachedVolume.NodeName) {
				// Volume exists in actual state of world but not desired
				if !attachedVolume.MountedByNode {
					glog.V(5).Infof("Attempting to start DetachVolume for volume %q to node %q", attachedVolume.VolumeName, attachedVolume.NodeName)
					err := rc.attacherDetacher.DetachVolume(attachedVolume.AttachedVolume, rc.actualStateOfWorld)
					if err == nil {
						glog.Infof("Started DetachVolume for volume %q to node %q", attachedVolume.VolumeName, attachedVolume.NodeName)
					}
				} else {
					// If volume is not safe to detach (is mounted) wait a max amount of time before detaching any way.
					timeElapsed, err := rc.actualStateOfWorld.MarkDesireToDetach(attachedVolume.VolumeName, attachedVolume.NodeName)
					if err != nil {
						glog.Errorf("Unexpected error actualStateOfWorld.MarkDesireToDetach(): %v", err)
					}
					if timeElapsed > rc.maxWaitForUnmountDuration {
						glog.V(5).Infof("Attempting to start DetachVolume for volume %q to node %q. Volume is not safe to detach, but maxWaitForUnmountDuration expired.", attachedVolume.VolumeName, attachedVolume.NodeName)
						err := rc.attacherDetacher.DetachVolume(attachedVolume.AttachedVolume, rc.actualStateOfWorld)
						if err == nil {
							glog.Infof("Started DetachVolume for volume %q to node %q due to maxWaitForUnmountDuration expiry.", attachedVolume.VolumeName, attachedVolume.NodeName)
						}
					}
				}
			}
		}

		// Ensure volumes that should be attached are attached.
		for _, volumeToAttach := range rc.desiredStateOfWorld.GetVolumesToAttach() {
			if rc.actualStateOfWorld.VolumeNodeExists(
				volumeToAttach.VolumeName, volumeToAttach.NodeName) {
				// Volume/Node exists, touch it to reset detachRequestedTime
				glog.V(12).Infof("Volume %q/Node %q is attached--touching.", volumeToAttach.VolumeName, volumeToAttach.NodeName)
				_, err := rc.actualStateOfWorld.AddVolumeNode(
					volumeToAttach.VolumeSpec, volumeToAttach.NodeName)
				if err != nil {
					glog.Errorf("Unexpected error on actualStateOfWorld.AddVolumeNode(): %v", err)
				}
			} else {
				// Volume/Node doesn't exist, spawn a goroutine to attach it
				glog.V(5).Infof("Attempting to start AttachVolume for volume %q to node %q", volumeToAttach.VolumeName, volumeToAttach.NodeName)
				err := rc.attacherDetacher.AttachVolume(volumeToAttach.VolumeToAttach, rc.actualStateOfWorld)
				if err == nil {
					glog.Infof("Started AttachVolume for volume %q to node %q", volumeToAttach.VolumeName, volumeToAttach.NodeName)
				}
			}
		}
	}
}
