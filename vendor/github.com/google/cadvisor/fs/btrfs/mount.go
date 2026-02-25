// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build linux

package btrfs

import (
	"fmt"
	"syscall"

	mount "github.com/moby/sys/mountinfo"
	"k8s.io/klog/v2"
)

// major extracts the major device number from a device number.
func major(devNumber uint64) uint {
	return uint((devNumber >> 8) & 0xfff)
}

// minor extracts the minor device number from a device number.
func minor(devNumber uint64) uint {
	return uint((devNumber & 0xff) | ((devNumber >> 12) & 0xfff00))
}

// GetBtrfsMajorMinorIds gets the major and minor device IDs for a btrfs mount point.
// This is a workaround for wrong btrfs Major and Minor Ids reported in /proc/self/mountinfo.
// Instead of using values from /proc/self/mountinfo we use stat to get Ids from btrfs mount point.
func GetBtrfsMajorMinorIds(mnt *mount.Info) (int, int, error) {
	buf := new(syscall.Stat_t)
	err := syscall.Stat(mnt.Source, buf)
	if err != nil {
		err = fmt.Errorf("stat failed on %s with error: %s", mnt.Source, err)
		return 0, 0, err
	}

	klog.V(4).Infof("btrfs mount %#v", mnt)
	if buf.Mode&syscall.S_IFMT == syscall.S_IFBLK {
		err := syscall.Stat(mnt.Mountpoint, buf)
		if err != nil {
			err = fmt.Errorf("stat failed on %s with error: %s", mnt.Mountpoint, err)
			return 0, 0, err
		}

		// The type Dev and Rdev in Stat_t are 32bit on mips.
		klog.V(4).Infof("btrfs dev major:minor %d:%d\n", int(major(uint64(buf.Dev))), int(minor(uint64(buf.Dev))))    // nolint: unconvert
		klog.V(4).Infof("btrfs rdev major:minor %d:%d\n", int(major(uint64(buf.Rdev))), int(minor(uint64(buf.Rdev)))) // nolint: unconvert

		return int(major(uint64(buf.Dev))), int(minor(uint64(buf.Dev))), nil // nolint: unconvert
	}
	return 0, 0, fmt.Errorf("%s is not a block device", mnt.Source)
}
