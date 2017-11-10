/*
Copyright 2014 The Kubernetes Authors.

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
// diskManager interface and diskSetup/TearDown functions abstract commonly used procedures to setup a block volume
// rbd volume implements diskManager, calls diskSetup when creating a volume, and calls diskTearDown inside volume unmounter.
// TODO: consolidate, refactor, and share diskManager among iSCSI, GCE PD, and RBD
//

package rbd

import (
	"fmt"
	"os"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

// Abstract interface to disk operations.
type diskManager interface {
	// MakeGlobalPDName creates global persistent disk path.
	MakeGlobalPDName(disk rbd) string
	// Attaches the disk to the kubelet's host machine.
	// If it successfully attaches, the path to the device
	// is returned. Otherwise, an error will be returned.
	AttachDisk(disk rbdMounter) (string, error)
	// Detaches the disk from the kubelet's host machine.
	DetachDisk(plugin *rbdPlugin, deviceMountPath string, device string) error
	// Creates a rbd image.
	CreateImage(provisioner *rbdVolumeProvisioner) (r *v1.RBDPersistentVolumeSource, volumeSizeGB int, err error)
	// Deletes a rbd image.
	DeleteImage(deleter *rbdVolumeDeleter) error
}

// utility to mount a disk based filesystem
func diskSetUp(manager diskManager, b rbdMounter, volPath string, mounter mount.Interface, fsGroup *int64) error {
	globalPDPath := manager.MakeGlobalPDName(*b.rbd)
	notMnt, err := mounter.IsLikelyNotMountPoint(globalPDPath)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("cannot validate mountpoint: %s", globalPDPath)
		return err
	}
	if notMnt {
		return fmt.Errorf("no device is mounted at %s", globalPDPath)
	}

	notMnt, err = mounter.IsLikelyNotMountPoint(volPath)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("cannot validate mountpoint: %s", volPath)
		return err
	}
	if !notMnt {
		return nil
	}

	if err := os.MkdirAll(volPath, 0750); err != nil {
		glog.Errorf("failed to mkdir:%s", volPath)
		return err
	}
	// Perform a bind mount to the full path to allow duplicate mounts of the same disk.
	options := []string{"bind"}
	if (&b).GetAttributes().ReadOnly {
		options = append(options, "ro")
	}
	mountOptions := volume.JoinMountOptions(b.mountOptions, options)
	err = mounter.Mount(globalPDPath, volPath, "", mountOptions)
	if err != nil {
		glog.Errorf("failed to bind mount:%s", globalPDPath)
		return err
	}
	glog.V(3).Infof("rbd: successfully bind mount %s to %s with options %v", globalPDPath, volPath, mountOptions)

	if !b.ReadOnly {
		volume.SetVolumeOwnership(&b, fsGroup)
	}

	return nil
}

// utility to tear down a disk based filesystem
func diskTearDown(manager diskManager, c rbdUnmounter, volPath string, mounter mount.Interface) error {
	notMnt, err := mounter.IsLikelyNotMountPoint(volPath)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("cannot validate mountpoint: %s", volPath)
		return err
	}
	if notMnt {
		glog.V(3).Infof("volume path %s is not a mountpoint, deleting", volPath)
		return os.Remove(volPath)
	}

	// Unmount the bind-mount inside this pod.
	if err := mounter.Unmount(volPath); err != nil {
		glog.Errorf("failed to umount %s", volPath)
		return err
	}

	notMnt, mntErr := mounter.IsLikelyNotMountPoint(volPath)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
		return err
	}
	if notMnt {
		if err := os.Remove(volPath); err != nil {
			glog.V(2).Info("Error removing mountpoint ", volPath, ": ", err)
			return err
		}
	}
	return nil
}
