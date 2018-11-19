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

/*
#include <stdlib.h>
#include <dirent.h>
#include <linux/fs.h>
#include <linux/quota.h>
#include <linux/dqblk_xfs.h>
#include <errno.h>

#ifndef FS_XFLAG_PROJINHERIT
struct fsxattr {
	__u32		fsx_xflags;
	__u32		fsx_extsize;
	__u32		fsx_nextents;
	__u32		fsx_projid;
	unsigned char	fsx_pad[12];
};
#define FS_XFLAG_PROJINHERIT	0x00000200
#endif
#ifndef FS_IOC_FSGETXATTR
#define FS_IOC_FSGETXATTR		_IOR ('X', 31, struct fsxattr)
#endif
#ifndef FS_IOC_FSSETXATTR
#define FS_IOC_FSSETXATTR		_IOW ('X', 32, struct fsxattr)
#endif

#ifndef PRJQUOTA
#define PRJQUOTA	2
#endif
#ifndef Q_XGETQSTAT_PRJQUOTA
#define Q_XGETQSTAT_PRJQUOTA QCMD(Q_XGETQSTAT, PRJQUOTA)
#endif
*/
import "C"

import (
	"fmt"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
	"k8s.io/klog"
)

// IsFilesystemOfType determines whether the filesystem specified is of the type
// specified by the magic number
func IsFilesystemOfType(mountpoint string, backingDev string, magic int64) bool {
	var buf syscall.Statfs_t
	err := syscall.Statfs(mountpoint, &buf)
	if err != nil {
		klog.V(3).Infof("Extfs Unable to statfs %s: %v", mountpoint, err)
		return false
	}
	if buf.Type != magic {
		return false
	}

	var qstat C.fs_quota_stat_t
	CPath := C.CString(backingDev)
	defer free(CPath)

	_, _, errno := unix.Syscall6(unix.SYS_QUOTACTL, uintptr(C.Q_XGETQSTAT_PRJQUOTA), uintptr(unsafe.Pointer(CPath)), 0, uintptr(unsafe.Pointer(&qstat)), 0, 0)
	return errno == 0 && qstat.qs_flags&C.FS_QUOTA_PDQ_ENFD > 0 && qstat.qs_flags&C.FS_QUOTA_PDQ_ACCT > 0
}

func free(p *C.char) {
	C.free(unsafe.Pointer(p))
}

func openDir(path string) (*C.DIR, error) {
	Cpath := C.CString(path)
	defer free(Cpath)

	dir := C.opendir(Cpath)
	if dir == nil {
		return nil, fmt.Errorf("Can't open dir")
	}
	return dir, nil
}

func closeDir(dir *C.DIR) {
	if dir != nil {
		C.closedir(dir)
	}
}

func getDirFd(dir *C.DIR) uintptr {
	return uintptr(C.dirfd(dir))
}

// GetQuotaOnDir retrieves the quota ID (if any) associated with the specified directory
func GetQuotaOnDir(path string) (QuotaID, error) {
	dir, err := openDir(path)
	if err != nil {
		klog.V(3).Infof("Can't open directory %s: %#+v", path, err)
		return BadQuotaID, err
	}
	defer closeDir(dir)
	var fsx C.struct_fsxattr
	_, _, errno := unix.Syscall(unix.SYS_IOCTL, getDirFd(dir), C.FS_IOC_FSGETXATTR,
		uintptr(unsafe.Pointer(&fsx)))
	if errno != 0 {
		return BadQuotaID, fmt.Errorf("Failed to get quota ID for %s: %v", path, errno.Error())
	}
	if fsx.fsx_projid == 0 {
		return BadQuotaID, fmt.Errorf("Failed to get quota ID for %s: %s", path, "no applicable quota")
	}
	return QuotaID(fsx.fsx_projid), nil
}

// ApplyProjectToDir applies the specified quota ID to the specified directory
func ApplyProjectToDir(path string, id QuotaID) error {
	dir, err := openDir(path)
	if err != nil {
		return err
	}
	defer closeDir(dir)

	var fsx C.struct_fsxattr
	_, _, errno := unix.Syscall(unix.SYS_IOCTL, getDirFd(dir), C.FS_IOC_FSGETXATTR,
		uintptr(unsafe.Pointer(&fsx)))
	if errno != 0 {
		return fmt.Errorf("Failed to get quota ID for %s: %v", path, errno.Error())
	}
	fsx.fsx_projid = C.__u32(id)
	fsx.fsx_xflags |= C.FS_XFLAG_PROJINHERIT
	_, _, errno = unix.Syscall(unix.SYS_IOCTL, getDirFd(dir), C.FS_IOC_FSSETXATTR,
		uintptr(unsafe.Pointer(&fsx)))
	if errno != 0 {
		return fmt.Errorf("Failed to set quota ID for %s: %v", path, errno.Error())
	}
	return nil
}
