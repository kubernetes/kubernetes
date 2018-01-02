// Copyright 2016 The rkt Authors
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

package common

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/coreos/rkt/pkg/fs"
	rktlog "github.com/coreos/rkt/pkg/log"
	"github.com/coreos/rkt/pkg/mountinfo"
)

// ChrootPrivateUnmount cleans up in a safe way all mountpoints existing under
// `targetPath`. This requires multiple steps:
//  1. take handles to the current rootdir and workdir (to restore at the end)
//  2. parse /proc/self/mountinfo to get a list of all mount targets, and filter
//     out those outside of targetPath
//  3. chroot into target path, so that all mounts and symlinks can be properly
//     de-referenced as they appear inside the rootfs
//  4. mark all mounts as private, so that further operations are not propagated
//     outside of this rootfs - in descending nest order (parent first)
//  5. unmount all mount targets - in ascending nest order (children first).
//     If unmount fails, lazy-detach the mount target so that the kernel can
//     still clean it up once it ceases to be busy
//  6. chdir and chroot back to the original state
func ChrootPrivateUnmount(targetPath string, log *rktlog.Logger, diagf func(string, ...interface{})) error {
	mounter := fs.NewLoggingMounter(
		fs.MounterFunc(syscall.Mount),
		fs.UnmounterFunc(syscall.Unmount),
		diagf,
	)

	// getFd checks if dirFile is a directory and returns its fd
	getFd := func(dirFile *os.File) (int, error) {
		dirInfo, err := dirFile.Stat()
		if err != nil {
			return 0, fmt.Errorf("error getting info on %q: %s", dirFile.Name(), err)
		}
		if !dirInfo.IsDir() {
			return 0, fmt.Errorf("%q is not a directory", dirFile.Name())
		}
		return int(dirFile.Fd()), nil
	}

	// 1a. remember current workdir
	cwdString, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("error getting current workdir: %s", err)
	}

	// 1b. take the fd for current rootdir
	rootFile, err := os.Open("/")
	if err != nil {
		return fmt.Errorf("error opening current rootdir: %s", err)
	}
	defer rootFile.Close()
	rootFd, err := getFd(rootFile)
	if err != nil {
		return err
	}

	// 2. list all mounts and keeps only those in targetPath
	//    (this needs to be done here as /proc may not be available after chroot)
	mnts, err := mountinfo.ParseMounts(0)
	if err != nil {
		return fmt.Errorf("error getting mountinfo: %s", err)
	}
	mnts = mnts.Filter(mountinfo.HasPrefix(targetPath))

	// 2. chdir to / (to avoid keeping a dir busy) and chroot to target
	//    (defers in reverse order to escape the chroot on return)
	_ = syscall.Fchdir(rootFd)
	defer os.Chdir(cwdString)
	err = syscall.Chroot(targetPath)
	if err != nil {
		return fmt.Errorf("error chroot-ing to %q: %s", targetPath, err)
	}
	defer syscall.Chroot(".")
	defer syscall.Fchdir(rootFd)

	// 4. mark mounts private, top to bottom
	//    (mountinfo list contains full-paths captured outside of chroot, prefix-stripping needed)
	for i := len(mnts) - 1; i >= 0; i-- {
		mnt := mnts[i]
		if mnt.NeedsRemountPrivate() {
			relPath := strings.TrimPrefix(mnt.MountPoint, targetPath)
			mntPath := filepath.Join("/", relPath)
			err := mounter.Mount("", mntPath, "", syscall.MS_PRIVATE|syscall.MS_REC, "")
			if err != nil {
				log.Printf("skipping %q, not marked as private: %v", mntPath, err)
			}
		}
	}
	// 5. unmount all targets, bottom to top. If busy, still mark them as detached for later cleanups.
	//    (mountinfo list contains full-paths captured outside of chroot, prefix-stripping needed)
	for _, mnt := range mnts {
		relPath := strings.TrimPrefix(mnt.MountPoint, targetPath)
		mntPath := filepath.Join("/", relPath)
		err := mounter.Unmount(mntPath, 0)
		if err == syscall.EBUSY {
			log.Printf("mount %q is busy, marking for lazy detach", mntPath)
			_ = mounter.Unmount(mntPath, syscall.MNT_DETACH)
		}
	}
	return nil
}
