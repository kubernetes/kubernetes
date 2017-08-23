/*
Copyright 2015 The Kubernetes Authors.

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

package fc

import (
	"os"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

// Abstract interface to disk operations.
type diskManager interface {
	MakeGlobalPDName(disk fcDisk) string
	// Attaches the disk to the kubelet's host machine.
	AttachDisk(b fcDiskMounter) (string, error)
	// Detaches the disk from the kubelet's host machine.
	DetachDisk(disk fcDiskUnmounter, devName string) error
}

// utility to mount a disk based filesystem
func diskSetUp(manager diskManager, b fcDiskMounter, volPath string, mounter mount.Interface, fsGroup *int64) error {
	globalPDPath := manager.MakeGlobalPDName(*b.fcDisk)
	// TODO: handle failed mounts here.
	noMnt, err := mounter.IsLikelyNotMountPoint(volPath)

	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("cannot validate mountpoint: %s", volPath)
		return err
	}
	if !noMnt {
		return nil
	}
	if err := os.MkdirAll(volPath, 0750); err != nil {
		glog.Errorf("failed to mkdir:%s", volPath)
		return err
	}
	// Perform a bind mount to the full path to allow duplicate mounts of the same disk.
	options := []string{"bind"}
	if b.readOnly {
		options = append(options, "ro")
	}
	err = mounter.Mount(globalPDPath, volPath, "", options)
	if err != nil {
		glog.Errorf("failed to bind mount:%s", globalPDPath)
		return err
	}

	if !b.readOnly {
		volume.SetVolumeOwnership(&b, fsGroup)
	}

	return nil
}

// utility to tear down a disk based filesystem
func diskTearDown(manager diskManager, c fcDiskUnmounter, volPath string, mounter mount.Interface) error {
	noMnt, err := mounter.IsLikelyNotMountPoint(volPath)
	if err != nil {
		glog.Errorf("cannot validate mountpoint %s", volPath)
		return err
	}
	if noMnt {
		return os.Remove(volPath)
	}

	if err := mounter.Unmount(volPath); err != nil {
		glog.Errorf("failed to unmount %s", volPath)
		return err
	}

	noMnt, mntErr := mounter.IsLikelyNotMountPoint(volPath)
	if mntErr != nil {
		glog.Errorf("isMountpoint check failed: %v", mntErr)
		return err
	}
	if noMnt {
		if err := os.Remove(volPath); err != nil {
			return err
		}
	}
	return nil

}
