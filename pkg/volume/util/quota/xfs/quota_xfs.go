// +build linux

/*
Copyright 2018 The Kubernetes Authors.

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

package xfs

import (
	"fmt"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/volume/util/quota/common"
)

const (
	linuxXfsMagic = 0x58465342
	// Documented in man xfs_quota(8); not necessarily the same
	// as the filesystem blocksize
	quotaBsize        = 512
	bitsPerWord       = 32 << (^uint(0) >> 63) // either 32 or 64
	maxQuota    int64 = 1<<(bitsPerWord-1) - 1 // either 1<<31 - 1 or 1<<63 - 1
)

// DiskQuota -- shadowing fs_disk_quota_t from <linux/dqblk_xfs.h>
type diskQuota struct {
	version      int8
	flags        int8
	fieldmask    uint16
	id           uint32
	blkhardlimit uint64
	blksoftlimit uint64
	inohardlimit uint64
	inosoftlimit uint64
	bcount       uint64
	icount       uint64
	itimer       int32
	btimer       int32
	iwarns       uint16
	bwarns       uint16
	padding2     int32
	rtbhardlimit uint64
	rtbsoftlimit uint64
	rtbcount     uint64
	rtbtimer     int32
	rtbwarns     uint16
	padding3     int16
	padding4     [4]byte
}

// VolumeProvider supplies a quota applier to the generic code.
type VolumeProvider struct {
}

// GetQuotaApplier -- does this backing device support quotas that
// can be applied to directories?
func (*VolumeProvider) GetQuotaApplier(mountpoint string, backingDev string) common.LinuxVolumeQuotaApplier {
	if common.IsFilesystemOfType(mountpoint, backingDev, linuxXfsMagic) {
		return xfsVolumeQuota{backingDev}
	}
	return nil
}

type xfsVolumeQuota struct {
	backingDev string
}

// GetQuotaOnDir -- get the quota ID that applies to this directory.
func (v xfsVolumeQuota) GetQuotaOnDir(path string) (common.QuotaID, error) {
	return common.GetQuotaOnDir(path)
}

// SetQuotaOnDir -- apply the specified quota to the directory.  If
// bytes is not greater than zero, the quota should be applied in a
// way that is non-enforcing (either explicitly so or by setting a
// quota larger than anything the user may possibly create)
func (v xfsVolumeQuota) SetQuotaOnDir(path string, id common.QuotaID, bytes int64) error {
	klog.V(3).Infof("xfsSetQuotaOn %s ID %v bytes %v", path, id, bytes)
	if bytes < 0 || bytes > maxQuota {
		bytes = maxQuota
	}

	var d diskQuota
	d.version = common.FSDQuotVersion
	d.id = uint32(id)
	d.flags = common.XFSProjQuota

	d.fieldmask = common.FSDQBHard
	d.blkhardlimit = uint64(bytes / quotaBsize)
	d.blksoftlimit = d.blkhardlimit

	cs := append([]byte(v.backingDev), 0)

	_, _, errno := unix.Syscall6(unix.SYS_QUOTACTL, common.QXSetPQLim,
		uintptr(unsafe.Pointer(&cs[0])), uintptr(d.id),
		uintptr(unsafe.Pointer(&d)), 0, 0)
	if errno != 0 {
		return fmt.Errorf("Failed to set quota limit for ID %d on %s: %v",
			id, path, errno.Error())
	}
	return common.ApplyProjectToDir(path, id)
}

func (v xfsVolumeQuota) getQuotaInfo(path string, id common.QuotaID) (diskQuota, syscall.Errno) {
	var d diskQuota

	cs := append([]byte(v.backingDev), 0)

	_, _, errno := unix.Syscall6(unix.SYS_QUOTACTL, common.QXGetPQuota,
		uintptr(unsafe.Pointer(&cs[0])), uintptr(uint32(id)),
		uintptr(unsafe.Pointer(&d)), 0, 0)
	return d, errno
}

// QuotaIDIsInUse -- determine whether the quota ID is already in use.
func (v xfsVolumeQuota) QuotaIDIsInUse(path string, id common.QuotaID) (bool, error) {
	_, errno := v.getQuotaInfo(path, id)
	return errno == 0, nil
}

// GetConsumption -- retrieve the consumption (in bytes) of the directory
func (v xfsVolumeQuota) GetConsumption(path string, id common.QuotaID) (int64, error) {
	d, errno := v.getQuotaInfo(path, id)
	if errno != 0 {
		return 0, fmt.Errorf("Failed to get quota for %s: %s", path, errno.Error())
	}
	klog.V(3).Infof("Consumption for %s is %v", path, d.bcount*quotaBsize)
	return int64(d.bcount) * quotaBsize, nil
}

// GetInodes -- retrieve the number of inodes in use under the directory
func (v xfsVolumeQuota) GetInodes(path string, id common.QuotaID) (int64, error) {
	d, errno := v.getQuotaInfo(path, id)
	if errno != 0 {
		return 0, fmt.Errorf("Failed to get quota for %s: %s", path, errno.Error())
	}
	klog.V(3).Infof("Inode consumption for %s is %v", path, d.icount)
	return int64(d.icount), nil
}
