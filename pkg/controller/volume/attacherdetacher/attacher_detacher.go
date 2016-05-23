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

// Package attacherdetacher implements interfaces that enable triggering attach
// and detach operations on volumes.
package attacherdetacher

import (
	"fmt"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/controller/volume/cache"
	"k8s.io/kubernetes/pkg/util/goroutinemap"
	"k8s.io/kubernetes/pkg/volume"
)

// AttacherDetacher defines a set of operations for attaching or detaching a
// volume from a node.
type AttacherDetacher interface {
	// Spawns a new goroutine to execute volume-specific logic to attach the
	// volume to the node specified in the volumeToAttach.
	// Once attachment completes successfully, the actualStateOfWorld is updated
	// to indicate the volume is attached to the node.
	// If there is an error indicating the volume is already attached to the
	// specified node, attachment is assumed to be successful (plugins are
	// responsible for implmenting this behavior).
	// All other errors are logged and the goroutine terminates without updating
	// actualStateOfWorld (caller is responsible for retrying as needed).
	AttachVolume(volumeToAttach cache.VolumeToAttach, actualStateOfWorld cache.ActualStateOfWorld) error

	// Spawns a new goroutine to execute volume-specific logic to detach the
	// volume from the node specified in volumeToDetach.
	// Once detachment completes successfully, the actualStateOfWorld is updated
	// to remove the volume/node combo.
	// If there is an error indicating the volume is already detached from the
	// specified node, detachment is assumed to be successful (plugins are
	// responsible for implmenting this behavior).
	// All other errors are logged and the goroutine terminates without updating
	// actualStateOfWorld (caller is responsible for retrying as needed).
	DetachVolume(volumeToDetach cache.AttachedVolume, actualStateOfWorld cache.ActualStateOfWorld) error
}

// NewAttacherDetacher returns a new instance of AttacherDetacher.
func NewAttacherDetacher(volumePluginMgr *volume.VolumePluginMgr) AttacherDetacher {
	return &attacherDetacher{
		volumePluginMgr:   volumePluginMgr,
		pendingOperations: goroutinemap.NewGoRoutineMap(),
	}
}

type attacherDetacher struct {
	// volumePluginMgr is the volume plugin manager used to create volume
	// plugin objects.
	volumePluginMgr *volume.VolumePluginMgr
	// pendingOperations keeps track of pending attach and detach operations so
	// multiple operations are not started on the same volume
	pendingOperations goroutinemap.GoRoutineMap
}

func (ad *attacherDetacher) AttachVolume(
	volumeToAttach cache.VolumeToAttach,
	actualStateOfWorld cache.ActualStateOfWorld) error {
	attachFunc, err := ad.generateAttachVolumeFunc(volumeToAttach, actualStateOfWorld)
	if err != nil {
		return err
	}

	return ad.pendingOperations.Run(string(volumeToAttach.VolumeName), attachFunc)
}

func (ad *attacherDetacher) DetachVolume(
	volumeToDetach cache.AttachedVolume,
	actualStateOfWorld cache.ActualStateOfWorld) error {
	detachFunc, err := ad.generateDetachVolumeFunc(volumeToDetach, actualStateOfWorld)
	if err != nil {
		return err
	}

	return ad.pendingOperations.Run(string(volumeToDetach.VolumeName), detachFunc)
}

func (ad *attacherDetacher) generateAttachVolumeFunc(
	volumeToAttach cache.VolumeToAttach,
	actualStateOfWorld cache.ActualStateOfWorld) (func() error, error) {
	// Get attacher plugin
	attachableVolumePlugin, err := ad.volumePluginMgr.FindAttachablePluginBySpec(volumeToAttach.VolumeSpec)
	if err != nil || attachableVolumePlugin == nil {
		return nil, fmt.Errorf(
			"failed to get AttachablePlugin from volumeSpec for volume %q err=%v",
			volumeToAttach.VolumeSpec.Name(),
			err)
	}

	volumeAttacher, newAttacherErr := attachableVolumePlugin.NewAttacher()
	if newAttacherErr != nil {
		return nil, fmt.Errorf(
			"failed to get NewAttacher from volumeSpec for volume %q err=%v",
			volumeToAttach.VolumeSpec.Name(),
			newAttacherErr)
	}

	return func() error {
		// Execute attach
		attachErr := volumeAttacher.Attach(volumeToAttach.VolumeSpec, volumeToAttach.NodeName)

		if attachErr != nil {
			// On failure, just log and exit. The controller will retry
			glog.Errorf(
				"Attach operation for device %q to node %q failed with: %v",
				volumeToAttach.VolumeName, volumeToAttach.NodeName, attachErr)
			return attachErr
		}

		glog.Infof(
			"Successfully attached device %q to node %q. Will update actual state of world.",
			volumeToAttach.VolumeName, volumeToAttach.NodeName)

		// Update actual state of world
		_, addVolumeNodeErr := actualStateOfWorld.AddVolumeNode(volumeToAttach.VolumeSpec, volumeToAttach.NodeName)
		if addVolumeNodeErr != nil {
			// On failure, just log and exit. The controller will retry
			glog.Errorf(
				"Attach operation for device %q to node %q succeeded, but updating actualStateOfWorld failed with: %v",
				volumeToAttach.VolumeName, volumeToAttach.NodeName, addVolumeNodeErr)
			return addVolumeNodeErr
		}

		return nil
	}, nil
}

func (ad *attacherDetacher) generateDetachVolumeFunc(
	volumeToDetach cache.AttachedVolume,
	actualStateOfWorld cache.ActualStateOfWorld) (func() error, error) {
	// Get attacher plugin
	attachableVolumePlugin, err := ad.volumePluginMgr.FindAttachablePluginBySpec(volumeToDetach.VolumeSpec)
	if err != nil || attachableVolumePlugin == nil {
		return nil, fmt.Errorf(
			"failed to get AttachablePlugin from volumeSpec for volume %q err=%v",
			volumeToDetach.VolumeSpec.Name(),
			err)
	}

	deviceName, err := attachableVolumePlugin.GetDeviceName(volumeToDetach.VolumeSpec)
	if err != nil {
		return nil, fmt.Errorf(
			"failed to GetDeviceName from AttachablePlugin for volumeSpec %q err=%v",
			volumeToDetach.VolumeSpec.Name(),
			err)
	}

	volumeDetacher, err := attachableVolumePlugin.NewDetacher()
	if err != nil {
		return nil, fmt.Errorf(
			"failed to get NewDetacher from volumeSpec for volume %q err=%v",
			volumeToDetach.VolumeSpec.Name(),
			err)
	}

	return func() error {
		// Execute detach
		detachErr := volumeDetacher.Detach(deviceName, volumeToDetach.NodeName)

		if detachErr != nil {
			// On failure, just log and exit. The controller will retry
			glog.Errorf(
				"Detach operation for device %q from node %q failed with: %v",
				volumeToDetach.VolumeName, volumeToDetach.NodeName, detachErr)
			return detachErr
		}

		glog.Infof(
			"Successfully detached device %q from node %q. Will update actual state of world.",
			volumeToDetach.VolumeName, volumeToDetach.NodeName)

		// Update actual state of world
		actualStateOfWorld.DeleteVolumeNode(volumeToDetach.VolumeName, volumeToDetach.NodeName)

		return nil
	}, nil
}
