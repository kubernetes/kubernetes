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

package cinder

import (
	"time"
	
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

type cinderManager interface {
	GetName() string

	// Attaches the volume specified by the given spec to the node with the given Name.
	// On success, returns the device path where the device was attached on the
	// node.
	Attach(spec *volume.Spec, nodeName types.NodeName) (string, error)
	// VolumesAreAttached checks whether the list of volumes still attached to the specified
	// the node. It returns a map which maps from the volume spec to the checking result.
	// If an error is occurred during checking, the error will be returned.
	VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error)
	// WaitForAttach blocks until the device is attached to this
	// node. If it successfully attaches, the path to the device
	// is returned. Otherwise, if the device does not attach after
	// the given timeout period, an error will be returned.
	WaitForAttach(spec *volume.Spec, devicePath string, timeout time.Duration) (string, error)
	// Detach the given device from the node with the given Name.
	Detach(spec *volume.Spec, deviceName string, nodeName types.NodeName) error
	// WaitForDetach blocks until the device is detached from this
	// node. If the device does not detach within the given timeout
	// period an error is returned.
	WaitForDetach(spec *volume.Spec, devicePath string, timeout time.Duration) error
	// UnmountDevice unmounts the global mount of the disk. This
	// should only be called once all bind mounts have been
	// unmounted.
	UnmountDevice(spec *volume.Spec, deviceMountPath string, mounter mount.Interface) error

	// CreateVolume provisions a volume.
	CreateVolume(provisioner *cinderVolumeProvisioner) (volumeID string, volumeSizeGB int, secretRef *v1.LocalObjectReference, err error)
	// DeleteVolume deletes a provisioned volume.
	DeleteVolume(deleter *cinderVolumeDeleter) error
}
