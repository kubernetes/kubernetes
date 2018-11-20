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

package common

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
	"k8s.io/klog"
)

const (
	// FSDQuotVersion -- quota version
	FSDQuotVersion = 0x00000001
	// FSDQBHard -- manipulate hard limit
	FSDQBHard = 0x00000008
	// FSDQBSoft -- manipulate soft limit
	FSDQBSoft = 0x00000004
	// FSIocGetXAttr -- get extended file attributes
	FSIocGetXAttr = 0x801c581f
	// FSIocSetXAttr -- set extended file attributes
	FSIocSetXAttr = 0x401c5820
	// FSQuotaPDQAcct -- quotas for accounting
	FSQuotaPDQAcct = 0x00000010
	// FSQuotaPDQEnfd -- quotas for enforcement (currently not used)
	// Not currently using them, but a filesystem with enforcing quotas set
	// is useful for our purpose.
	FSQuotaPDQEnfd = 0x00000020
	// FSXFlagProjInherit -- project quotas are inherited by subdirs and files
	FSXFlagProjInherit = 0x00000200
	// XGetQStatPrjQuota -- manipulate project quotas with quotactl(2)
	XGetQStatPrjQuota = 0x00580502
	// QIfLimits -- Needed for setting quota limits
	QIfLimits = 0x00000005
	// QGetPQuota -- Get project quotas for foreign (non-XFS) filesystems)
	QGetPQuota = 0x80000702
	// QSetPQuota -- Set project quotas for foreign (non-XFS) filesystems)
	QSetPQuota = 0x80000802
	// QXGetPQuota -- Get project quotas for XFS filesystems
	QXGetPQuota = 0x00580302
	// QXSetPQLim -- Set project quotas for XFS filesystems
	QXSetPQLim = 0x00580402
	// XFSProjQuota -- use of project (vs. user or group) quotas
	XFSProjQuota = 0x00000002
)

// Shadowing struct fsxattr from <linux.fs.h>
type fsxattr struct {
	xflags     uint32
	extsize    uint32
	nextents   uint32
	projid     uint32
	cowextsize uint32
	pad        [8]byte
}

// Shadowing fs_quota_stat_t from <linux/dqblk_xfs.h>
type fsqfilestat struct {
	fsino     uint64
	fsnblks   uint64
	fsextents uint64
}

type fsquotastat struct {
	version      int8
	flags        uint16
	pad          int8
	uquota       fsqfilestat
	gquota       fsqfilestat
	incoredqs    uint32
	btimelimit   int32
	itimelimit   int32
	rtbtimelimit int32
	bwarnlimit   int16
	iwarnlimit   int16
}

// IsFilesystemOfType determines whether the filesystem specified is of the type
// specified by the magic number
func IsFilesystemOfType(mountpoint string, backingDev string, magic int64) bool {
	var buf syscall.Statfs_t
	err := syscall.Statfs(mountpoint, &buf)
	if err != nil {
		klog.V(3).Infof("Extfs Unable to statfs %s: %v", mountpoint, err)
		return false
	}
	// Per https://golang.org/pkg/syscall/#Statfs_t buf.Type is int64, but
	// typecheck complains about this on i386, ARM (int32) and s390x (uint32).
	if int64(buf.Type) != magic {
		return false
	}

	var qstat fsquotastat
	CPath := append([]byte(backingDev), 0)

	_, _, errno := unix.Syscall6(unix.SYS_QUOTACTL, uintptr(XGetQStatPrjQuota), uintptr(unsafe.Pointer(&CPath[0])), 0, uintptr(unsafe.Pointer(&qstat)), 0, 0)
	return errno == 0 && qstat.flags&FSQuotaPDQEnfd > 0 && qstat.flags&FSQuotaPDQAcct > 0
}

func openDir(path string) (*os.File, error) {

	dir, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("Can't open %s: %v", path, err)
	}
	s, err := dir.Stat()
	if err != nil {
		dir.Close()
		return nil, fmt.Errorf("Can't stat %s: %v", path, err)
	}
	if !s.IsDir() {
		dir.Close()
		return nil, fmt.Errorf("%s: not a directory", path)
	}
	return dir, nil
}

func closeDir(dir *os.File) {
	if dir != nil {
		dir.Close()
	}
}

// GetQuotaOnDir retrieves the quota ID (if any) associated with the specified directory
func GetQuotaOnDir(path string) (QuotaID, error) {
	dir, err := openDir(path)
	if err != nil {
		klog.V(3).Infof("Can't open directory %s: %#+v", path, err)
		return BadQuotaID, err
	}
	defer closeDir(dir)
	var fsx fsxattr
	_, _, errno := unix.Syscall(unix.SYS_IOCTL, dir.Fd(), FSIocGetXAttr,
		uintptr(unsafe.Pointer(&fsx)))
	if errno != 0 {
		return BadQuotaID, fmt.Errorf("Failed to get quota ID for %s: %v", path, errno.Error())
	}
	if fsx.projid == 0 {
		return BadQuotaID, fmt.Errorf("Failed to get quota ID for %s: %s", path, "no applicable quota")
	}
	return QuotaID(fsx.projid), nil
}

// ApplyProjectToDir applies the specified quota ID to the specified directory
func ApplyProjectToDir(path string, id QuotaID) error {
	dir, err := openDir(path)
	if err != nil {
		return err
	}
	defer closeDir(dir)

	var fsx fsxattr
	_, _, errno := unix.Syscall(unix.SYS_IOCTL, dir.Fd(), FSIocGetXAttr,
		uintptr(unsafe.Pointer(&fsx)))
	if errno != 0 {
		return fmt.Errorf("Failed to get quota ID for %s: %v", path, errno.Error())
	}

	fsx.projid = uint32(id)
	fsx.xflags |= FSXFlagProjInherit
	_, _, errno = unix.Syscall(unix.SYS_IOCTL, dir.Fd(), FSIocSetXAttr,
		uintptr(unsafe.Pointer(&fsx)))
	if errno != 0 {
		return fmt.Errorf("Failed to set quota ID for %s: %v", path, errno.Error())
	}
	return nil
}
