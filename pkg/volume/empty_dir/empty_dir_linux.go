/*
Copyright 2015 Google Inc. All rights reserved.

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

package empty_dir

import (
	"fmt"
	"syscall"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/docker/libcontainer/selinux"
)

// Defined by Linux - the type number for tmpfs mounts.
const linuxTmpfsMagic = 0x01021994

// realMountDetector implements mountDetector in terms of syscalls.
type realMountDetector struct{}

// GetMountMedium returns the storgeMedium for the given path and a
// bool indicating whether or not the path is a mountpoint, or an error.
func (m *realMountDetector) GetMountMedium(path string) (storageMedium, bool, error) {
	isMnt, err := mount.IsMountPoint(path)
	if err != nil {
		return 0, false, fmt.Errorf("IsMountPoint(%q): %v", path, err)
	}
	buf := syscall.Statfs_t{}
	if err := syscall.Statfs(path, &buf); err != nil {
		return 0, false, fmt.Errorf("statfs(%q): %v", path, err)
	}
	if buf.Type == linuxTmpfsMagic {
		return mediumMemory, isMnt, nil
	}
	return mediumUnknown, isMnt, nil
}

// getTmpfsMountOptions returns the option string to be passed to the mount
// system call in order to set the SELinux context of the tmpfs mount to
// the same as the kubelet root directory. The rootcontext mount option
// is used so that files created within the mount will have the same
// SELinux context by default as the mountpoint itself.
//
// If SELinux is not enabled, returns an empty string.
func (ed *emptyDir) getTmpfsMountOptions() (string, error) {
	// If SELinux is not enabled, rootcontext is not a valid option;
	// return an empty string.
	if !selinux.SelinuxEnabled() {
		return "", nil
	}

	// Get the SELinux context of the Kubelet rootDir.
	rootContext, err := selinux.Getfilecon(ed.rootDir)
	if err != nil {
		return "", err
	}

	// There is a libcontainer bug where the null byte is not stripped from
	// the result of reading some selinux xattrs; strip it.
	//
	// TODO: remove when https://github.com/docker/libcontainer/issues/499
	// is fixed
	rootContext = rootContext[:len(rootContext)-1]

	// The rootcontext mount option sets the context of the mount and
	// the default context for all files created within it.
	return fmt.Sprintf("rootcontext=\"%v\"", rootContext), nil
}
