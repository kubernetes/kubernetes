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
	"os"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

// Abstract interface to disk operations.
type diskManager interface {
	// Attaches the disk to the kubelet's host machine.
	AttachDisk(disk rbdMounter) (string, error)
	// Detaches the disk from the kubelet's host machine.
	DetachDisk(plugin *rbdPlugin, device string) error
	// Creates a rbd image
	CreateImage(provisioner *rbdVolumeProvisioner) (r *api.RBDVolumeSource, volumeSizeGB int, err error)
	// Deletes a rbd image
	DeleteImage(deleter *rbdVolumeDeleter) error
}

// utility to mount a disk based filesystem
func diskSetUp(manager diskManager, b rbdMounter, volPath string, mounter mount.Interface, fsGroup *int64) error {
	// TODO: handle failed mounts here.
	notMnt, err := mounter.IsLikelyNotMountPoint(volPath)

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

	globalPDPath := makePDNameInternal(b.plugin.host, b.Pool, b.Image)
	// Perform a bind mount to the full path to allow duplicate mounts of the same disk.
	options := []string{"bind"}
	if b.ReadOnly {
		options = append(options, "ro")
	}
	err = mounter.Mount(globalPDPath, volPath, "", options)
	if err != nil {
		glog.Errorf("failed to bind mount:%s", globalPDPath)
		return err
	}

	if !b.ReadOnly {
		volume.SetVolumeOwnership(&b, fsGroup)
	}

	return nil
}

// utility to tear down a disk based filesystem
func diskTearDown(manager diskManager, c rbdUnmounter, volPath string, mounter mount.Interface) error {
	notMnt, err := mounter.IsLikelyNotMountPoint(volPath)
	if err != nil {
		glog.V(2).Info("Error checking if mountpoint ", volPath, ": ", err)
		return err
	}
	if notMnt {
		glog.V(2).Info("Not mountpoint, deleting")
		return os.Remove(volPath)
	}

	// Unmount the bind-mount inside this pod
	if err := mounter.Unmount(volPath); err != nil {
		glog.V(2).Info("Error unmounting ", volPath, ": ", err)
		return err
	}
	notMnt, mntErr := mounter.IsLikelyNotMountPoint(volPath)
	if mntErr != nil {
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
