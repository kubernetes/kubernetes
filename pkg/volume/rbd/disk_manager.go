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

//
// diskManager interface and diskSetup/TearDown functions abtract commonly used procedures to setup a block volume
// rbd volume implements diskManager, calls diskSetup when creating a volume, and calls diskTearDown inside volume cleaner.
// TODO: consolidate, refactor, and share diskManager among iSCSI, GCE PD, and RBD
//

package rbd

import (
	"os"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/mount"
)

// Abstract interface to disk operations.
type diskManager interface {
	MakeGlobalPDName(disk rbd) string
	// Attaches the disk to the kubelet's host machine.
	AttachDisk(disk rbdBuilder) error
	// Detaches the disk from the kubelet's host machine.
	DetachDisk(disk rbdCleaner, mntPath string) error
}

// utility to mount a disk based filesystem
func diskSetUp(manager diskManager, b rbdBuilder, volPath string, mounter mount.Interface) error {
	globalPDPath := manager.MakeGlobalPDName(*b.rbd)
	// TODO: handle failed mounts here.
	mountpoint, err := mounter.IsMountPoint(volPath)

	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("cannot validate mountpoint: %s", volPath)
		return err
	}
	if mountpoint {
		return nil
	}
	if err := manager.AttachDisk(b); err != nil {
		glog.Errorf("failed to attach disk")
		return err
	}

	if err := os.MkdirAll(volPath, 0750); err != nil {
		glog.Errorf("failed to mkdir:%s", volPath)
		return err
	}
	// Perform a bind mount to the full path to allow duplicate mounts of the same disk.
	options := []string{"bind"}
	if b.IsReadOnly() {
		options = append(options, "ro")
	}
	err = mounter.Mount(globalPDPath, volPath, "", options)
	if err != nil {
		glog.Errorf("failed to bind mount:%s", globalPDPath)
		return err
	}
	return nil
}

// utility to tear down a disk based filesystem
func diskTearDown(manager diskManager, c rbdCleaner, volPath string, mounter mount.Interface) error {
	mountpoint, err := mounter.IsMountPoint(volPath)
	if err != nil {
		glog.Errorf("cannot validate mountpoint %s", volPath)
		return err
	}
	if !mountpoint {
		return os.Remove(volPath)
	}

	refs, err := mount.GetMountRefs(mounter, volPath)
	if err != nil {
		glog.Errorf("failed to get reference count %s", volPath)
		return err
	}
	if err := mounter.Unmount(volPath); err != nil {
		glog.Errorf("failed to umount %s", volPath)
		return err
	}
	// If len(refs) is 1, then all bind mounts have been removed, and the
	// remaining reference is the global mount. It is safe to detach.
	if len(refs) == 1 {
		mntPath := refs[0]
		if err := manager.DetachDisk(c, mntPath); err != nil {
			glog.Errorf("failed to detach disk from %s", mntPath)
			return err
		}
	}

	mountpoint, mntErr := mounter.IsMountPoint(volPath)
	if mntErr != nil {
		glog.Errorf("isMountpoint check failed: %v", mntErr)
		return err
	}
	if !mountpoint {
		if err := os.Remove(volPath); err != nil {
			return err
		}
	}
	return nil

}
