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

package iscsi

import (
	"os"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/utils/mount"

	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

// Abstract interface to disk operations.
type diskManager interface {
	MakeGlobalPDName(disk iscsiDisk) string
	MakeGlobalVDPDName(disk iscsiDisk) string
	// Attaches the disk to the kubelet's host machine.
	AttachDisk(b iscsiDiskMounter) (string, error)
	// Detaches the disk from the kubelet's host machine.
	DetachDisk(disk iscsiDiskUnmounter, mntPath string) error
	// Detaches the block disk from the kubelet's host machine.
	DetachBlockISCSIDisk(disk iscsiDiskUnmapper, mntPath string) error
}

// utility to mount a disk based filesystem
// globalPDPath: global mount path like, /var/lib/kubelet/plugins/kubernetes.io/iscsi/{ifaceName}/{portal-some_iqn-lun-lun_id}
// volPath: pod volume dir path like, /var/lib/kubelet/pods/{podUID}/volumes/kubernetes.io~iscsi/{volumeName}
func diskSetUp(manager diskManager, b iscsiDiskMounter, volPath string, mounter mount.Interface, fsGroup *int64, fsGroupChangePolicy *v1.PodFSGroupChangePolicy) error {
	notMnt, err := mounter.IsLikelyNotMountPoint(volPath)
	if err != nil && !os.IsNotExist(err) {
		klog.Errorf("cannot validate mountpoint: %s", volPath)
		return err
	}
	if !notMnt {
		return nil
	}

	if err := os.MkdirAll(volPath, 0750); err != nil {
		klog.Errorf("failed to mkdir:%s", volPath)
		return err
	}
	// Perform a bind mount to the full path to allow duplicate mounts of the same disk.
	options := []string{"bind"}
	if b.readOnly {
		options = append(options, "ro")
	}
	if b.iscsiDisk.InitiatorName != "" {
		// new iface name is <target portal>:<volume name>
		b.iscsiDisk.Iface = b.iscsiDisk.Portals[0] + ":" + b.iscsiDisk.VolName
	}
	globalPDPath := manager.MakeGlobalPDName(*b.iscsiDisk)
	mountOptions := util.JoinMountOptions(b.mountOptions, options)
	err = mounter.Mount(globalPDPath, volPath, "", mountOptions)
	if err != nil {
		klog.Errorf("Failed to bind mount: source:%s, target:%s, err:%v", globalPDPath, volPath, err)
		noMnt, mntErr := b.mounter.IsLikelyNotMountPoint(volPath)
		if mntErr != nil {
			klog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
			return err
		}
		if !noMnt {
			if mntErr = b.mounter.Unmount(volPath); mntErr != nil {
				klog.Errorf("Failed to unmount: %v", mntErr)
				return err
			}
			noMnt, mntErr = b.mounter.IsLikelyNotMountPoint(volPath)
			if mntErr != nil {
				klog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
				return err
			}
			if !noMnt {
				//  will most likely retry on next sync loop.
				klog.Errorf("%s is still mounted, despite call to unmount().  Will try again next sync loop.", volPath)
				return err
			}
		}
		os.Remove(volPath)
		return err
	}

	if !b.readOnly {
		volume.SetVolumeOwnership(&b, fsGroup, fsGroupChangePolicy)
	}

	return nil
}
