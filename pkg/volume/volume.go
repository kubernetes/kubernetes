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
	"k8s.io/kubernetes/pkg/api"
	"os"
	"path"
)

// Volume represents a directory used by pods or hosts on a node.
// All method implementations of methods in the volume interface must be idempotent.
type Volume interface {
	// GetPath returns the directory path the volume is mounted to.
	GetPath() string
}

// Builder interface provides methods to set up/mount the volume.
type Builder interface {
	// Uses Interface to provide the path for Docker binds.
	Volume
	// SetUp prepares and mounts/unpacks the volume to a self-determined
	// directory path.  This may be called more than once, so
	// implementations must be idempotent.
	SetUp() error
	// SetUpAt prepares and mounts/unpacks the volume to the specified
	// directory path, which may or may not exist yet.  This may be called
	// more than once, so implementations must be idempotent.
	SetUpAt(dir string) error
	// IsReadOnly is a flag that gives the builder's ReadOnly attribute.
	// All persistent volumes have a private readOnly flag in their builders.
	IsReadOnly() bool
	// SupportsOwnershipManagement returns whether this builder wants
	// ownership management for the volume.  If this method returns true,
	// the Kubelet will:
	//
	// 1. Make the volume owned by group FSGroup
	// 2. Set the setgid bit is set (new files created in the volume will be owned by FSGroup)
	// 3. Logical OR the permission bits with rw-rw----
	SupportsOwnershipManagement() bool
}

// Cleaner interface provides methods to cleanup/unmount the volumes.
type Cleaner interface {
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

// Create adds a new resource in the storage provider and creates a PersistentVolume for the new resource.
// Calls to Create should block until complete.
type Creater interface {
	Create() (*api.PersistentVolume, error)
}

// Delete removes the resource from the underlying storage provider.  Calls to this method should block until
// the deletion is complete. Any error returned indicates the volume has failed to be reclaimed.
// A nil return indicates success.
type Deleter interface {
	Volume
	Delete() error
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
