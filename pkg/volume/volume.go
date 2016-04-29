/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package volume

import (
	"io/ioutil"
	"os"
	"path"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/util/mount"
)

// Volume represents a directory used by pods or hosts on a node.
// All method implementations of methods in the volume interface must be idempotent.
type Volume interface {
	// GetPath returns the path to which the volume should be
	// mounted for the pod.
	GetPath() string

	// MetricsProvider embeds methods for exposing metrics (e.g. used,available space).
	MetricsProvider
}

// MetricsProvider exposes metrics (e.g. used,available space) related to a Volume.
type MetricsProvider interface {
	// GetMetrics returns the Metrics for the Volume.  Maybe expensive for some implementations.
	GetMetrics() (*Metrics, error)
}

// Metrics represents the used and available bytes of the Volume.
type Metrics struct {
	// Used represents the total bytes used by the Volume.
	// Note: For block devices this maybe more than the total size of the files.
	Used *resource.Quantity

	// Capacity represents the total capacity (bytes) of the volume's underlying storage.
	// For Volumes that share a filesystem with the host (e.g. emptydir, hostpath) this is the size
	// of the underlying storage, and will not equal Used + Available as the fs is shared.
	Capacity *resource.Quantity

	// Available represents the storage space available (bytes) for the Volume.
	// For Volumes that share a filesystem with the host (e.g. emptydir, hostpath), this is the available
	// space on the underlying storage, and is shared with host processes and other Volumes.
	Available *resource.Quantity
}

// Attributes represents the attributes of this mounter.
type Attributes struct {
	ReadOnly        bool
	Managed         bool
	SupportsSELinux bool
}

// Mounter interface provides methods to set up/mount the volume.
type Mounter interface {
	// Uses Interface to provide the path for Docker binds.
	Volume
	// SetUp prepares and mounts/unpacks the volume to a
	// self-determined directory path. The mount point and its
	// content should be owned by 'fsGroup' so that it can be
	// accessed by the pod. This may be called more than once, so
	// implementations must be idempotent.
	SetUp(fsGroup *int64) error
	// SetUpAt prepares and mounts/unpacks the volume to the
	// specified directory path, which may or may not exist yet.
	// The mount point and its content should be owned by
	// 'fsGroup' so that it can be accessed by the pod. This may
	// be called more than once, so implementations must be
	// idempotent.
	SetUpAt(dir string, fsGroup *int64) error
	// GetAttributes returns the attributes of the mounter.
	GetAttributes() Attributes
}

// Unmounter interface provides methods to cleanup/unmount the volumes.
type Unmounter interface {
	Volume
	// TearDown unmounts the volume from a self-determined directory and
	// removes traces of the SetUp procedure.
	TearDown() error
	// TearDown unmounts the volume from the specified directory and
	// removes traces of the SetUp procedure.
	TearDownAt(dir string) error
}

// Recycler provides methods to reclaim the volume resource.
type Recycler interface {
	Volume
	// Recycle reclaims the resource.  Calls to this method should block until the recycling task is complete.
	// Any error returned indicates the volume has failed to be reclaimed.  A nil return indicates success.
	Recycle() error
}

// Provisioner is an interface that creates templates for PersistentVolumes and can create the volume
// as a new resource in the infrastructure provider.
type Provisioner interface {
	// Provision creates the resource by allocating the underlying volume in a storage system.
	// This method should block until completion.
	Provision(*api.PersistentVolume) error
	// NewPersistentVolumeTemplate creates a new PersistentVolume to be used as a template before saving.
	// The provisioner will want to tweak its properties, assign correct annotations, etc.
	// This func should *NOT* persist the PV in the API.  That is left to the caller.
	NewPersistentVolumeTemplate() (*api.PersistentVolume, error)
}

// Deleter removes the resource from the underlying storage provider.  Calls to this method should block until
// the deletion is complete. Any error returned indicates the volume has failed to be reclaimed.
// A nil return indicates success.
type Deleter interface {
	Volume
	// This method should block until completion.
	Delete() error
}

// Attacher can attach a volume to a node.
type Attacher interface {
	Volume

	// Attach the volume specified by the given spec to the given host
	Attach(spec *Spec, hostName string) error

	// WaitForAttach blocks until the device is attached to this
	// node. If it successfully attaches, the path to the device
	// is returned. Otherwise, if the device does not attach after
	// the given timeout period, an error will be returned.
	WaitForAttach(spec *Spec, timeout time.Duration) (string, error)

	// GetDeviceMountPath returns a path where the device should
	// be mounted after it is attached. This is a global mount
	// point which should be bind mounted for individual volumes.
	GetDeviceMountPath(spec *Spec) string

	// MountDevice mounts the disk to a global path which
	// individual pods can then bind mount
	MountDevice(devicePath string, deviceMountPath string, mounter mount.Interface) error
}

// Detacher can detach a volume from a node.
type Detacher interface {

	// Detach the given volume from the given host.
	Detach(deviceMountPath string, hostName string) error

	// WaitForDetach blocks until the device is detached from this
	// node. If the device does not detach within the given timout
	// period an error is returned.
	WaitForDetach(devicePath string, timout time.Duration) error

	// UnmountDevice unmounts the global mount of the disk. This
	// should only be called once all bind mounts have been
	// unmounted.
	UnmountDevice(globalMountPath string, mounter mount.Interface) error
}

func RenameDirectory(oldPath, newName string) (string, error) {
	newPath, err := ioutil.TempDir(path.Dir(oldPath), newName)
	if err != nil {
		return "", err
	}
	err = os.Rename(oldPath, newPath)
	if err != nil {
		return "", err
	}
	return newPath, nil
}
