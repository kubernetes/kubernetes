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

package extfs

import (
	"fmt"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/volume/util/quota/common"
)

//  ext4fs empirically has a maximum quota size of 2^48 - 1 1KiB blocks (256 petabytes)
const (
	linuxExtfsMagic       = 0xef53
	quotaBsize            = 1024                                   // extfs specific
	bitsPerWord           = 32 << (^uint(0) >> 63)                 // either 32 or 64
	maxQuota        int64 = (1<<(bitsPerWord-1) - 1) & (1<<58 - 1) // either 1<<31 - 1 or 1<<58 - 1
)

// dqblk -- shadowing if_dqblk from <linux/quota.h>
type dqblk struct {
	bhardlimit uint64
	bsoftlimit uint64
	curspace   uint64
	ihardlimit uint64
	isoftlimit uint64
	curinodes  uint64
	btime      uint64
	itime      uint64
	valid      uint32
}

// VolumeProvider supplies a quota applier to the generic code.
type VolumeProvider struct {
}

// GetQuotaApplier -- does this backing device support quotas that
// can be applied to directories?
func (*VolumeProvider) GetQuotaApplier(mountpoint string, backingDev string) common.LinuxVolumeQuotaApplier {
	if common.IsFilesystemOfType(mountpoint, backingDev, linuxExtfsMagic) {
		return extfsVolumeQuota{backingDev}
	}
	return nil
}

type extfsVolumeQuota struct {
	backingDev string
}

// GetQuotaOnDir -- get the quota ID that applies to this directory.

func (v extfsVolumeQuota) GetQuotaOnDir(path string) (common.QuotaID, error) {
	return common.GetQuotaOnDir(path)
}

// SetQuotaOnDir -- apply the specified quota to the directory.  If
// bytes is not greater than zero, the quota should be applied in a
// way that is non-enforcing (either explicitly so or by setting a
// quota larger than anything the user may possibly create)
func (v extfsVolumeQuota) SetQuotaOnDir(path string, id common.QuotaID, bytes int64) error {
	klog.V(3).Infof("extfsSetQuotaOn %s ID %v bytes %v", path, id, bytes)
	if bytes < 0 || bytes > maxQuota {
		bytes = maxQuota
	}

	var d dqblk

	d.bhardlimit = uint64(bytes / quotaBsize)
	d.bsoftlimit = d.bhardlimit
	d.ihardlimit = 0
	d.isoftlimit = 0
	d.valid = common.QIfLimits

	cs := append([]byte(v.backingDev), 0)

	_, _, errno := unix.Syscall6(unix.SYS_QUOTACTL, common.QSetPQuota,
		uintptr(unsafe.Pointer(&cs[0])), uintptr(id),
		uintptr(unsafe.Pointer(&d)), 0, 0)
	if errno != 0 {
		return fmt.Errorf("Failed to set quota limit for ID %d on %s: %v",
			id, path, errno.Error())
	}
	return common.ApplyProjectToDir(path, id)
}

func (v extfsVolumeQuota) getQuotaInfo(path string, id common.QuotaID) (dqblk, syscall.Errno) {
	var d dqblk

	cs := append([]byte(v.backingDev), 0)

	_, _, errno := unix.Syscall6(unix.SYS_QUOTACTL, common.QGetPQuota,
		uintptr(unsafe.Pointer(&cs[0])), uintptr(uint32(id)),
		uintptr(unsafe.Pointer(&d)), 0, 0)
	return d, errno
}

// QuotaIDIsInUse -- determine whether the quota ID is already in use.
func (v extfsVolumeQuota) QuotaIDIsInUse(path string, id common.QuotaID) (bool, error) {
	d, errno := v.getQuotaInfo(path, id)
	isInUse := !(d.bhardlimit == 0 && d.bsoftlimit == 0 && d.curspace == 0 &&
		d.ihardlimit == 0 && d.isoftlimit == 0 && d.curinodes == 0 &&
		d.btime == 0 && d.itime == 0)
	return errno == 0 && isInUse, nil
}

// GetConsumption -- retrieve the consumption (in bytes) of the directory
// Note that with ext[[:digit:]]fs the quota consumption is in bytes
// per man quotactl
func (v extfsVolumeQuota) GetConsumption(path string, id common.QuotaID) (int64, error) {
	d, errno := v.getQuotaInfo(path, id)
	if errno != 0 {
		return 0, fmt.Errorf("Failed to get quota for %s: %s", path, errno.Error())
	}
	klog.V(3).Infof("Consumption for %s is %v", path, d.curspace)
	return int64(d.curspace), nil
}

// GetInodes -- retrieve the number of inodes in use under the directory
func (v extfsVolumeQuota) GetInodes(path string, id common.QuotaID) (int64, error) {
	d, errno := v.getQuotaInfo(path, id)
	if errno != 0 {
		return 0, fmt.Errorf("Failed to get quota for %s: %s", path, errno.Error())
	}
	klog.V(3).Infof("Inode consumption for %s is %v", path, d.curinodes)
	return int64(d.curinodes), nil
}
