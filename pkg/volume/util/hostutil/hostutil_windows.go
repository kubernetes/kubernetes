//go:build windows
// +build windows

/*
Copyright 2017 The Kubernetes Authors.

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

package hostutil

import (
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
	"syscall"

	"golang.org/x/sys/windows"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/util/filesystem"
	"k8s.io/mount-utils"
	utilpath "k8s.io/utils/path"
)

// HostUtil implements HostUtils for Windows platforms.
type HostUtil struct{}

// NewHostUtil returns a struct that implements HostUtils on Windows platforms
func NewHostUtil() *HostUtil {
	return &HostUtil{}
}

// GetDeviceNameFromMount given a mnt point, find the device
func (hu *HostUtil) GetDeviceNameFromMount(mounter mount.Interface, mountPath, pluginMountDir string) (string, error) {
	return getDeviceNameFromMount(mounter, mountPath, pluginMountDir)
}

// getDeviceNameFromMount find the device(drive) name in which
// the mount path reference should match the given plugin mount directory. In case no mount path reference
// matches, returns the volume name taken from its given mountPath
func getDeviceNameFromMount(mounter mount.Interface, mountPath, pluginMountDir string) (string, error) {
	refs, err := mounter.GetMountRefs(mountPath)
	if err != nil {
		klog.V(4).Infof("GetMountRefs failed for mount path %q: %v", mountPath, err)
		return "", err
	}
	if len(refs) == 0 {
		return "", fmt.Errorf("directory %s is not mounted", mountPath)
	}
	basemountPath := mount.NormalizeWindowsPath(pluginMountDir)
	for _, ref := range refs {
		if strings.Contains(ref, basemountPath) {
			volumeID, err := filepath.Rel(mount.NormalizeWindowsPath(basemountPath), ref)
			if err != nil {
				klog.Errorf("Failed to get volume id from mount %s - %v", mountPath, err)
				return "", err
			}
			return volumeID, nil
		}
	}

	return path.Base(mountPath), nil
}

// DeviceOpened determines if the device is in use elsewhere
func (hu *HostUtil) DeviceOpened(pathname string) (bool, error) {
	return false, nil
}

// PathIsDevice determines if a path is a device.
func (hu *HostUtil) PathIsDevice(pathname string) (bool, error) {
	return false, nil
}

// MakeRShared checks that given path is on a mount with 'rshared' mount
// propagation. Empty implementation here.
func (hu *HostUtil) MakeRShared(path string) error {
	return nil
}

func isSystemCannotAccessErr(err error) bool {
	if fserr, ok := err.(*fs.PathError); ok {
		errno, ok := fserr.Err.(syscall.Errno)
		return ok && errno == windows.ERROR_CANT_ACCESS_FILE
	}

	return false
}

// GetFileType checks for sockets/block/character devices
func (hu *(HostUtil)) GetFileType(pathname string) (FileType, error) {
	filetype, err := getFileType(pathname)

	// os.Stat will return a 1920 error (windows.ERROR_CANT_ACCESS_FILE) if we use it on a Unix Socket
	// on Windows. In this case, we need to use a different method to check if it's a Unix Socket.
	if isSystemCannotAccessErr(err) {
		if isSocket, errSocket := filesystem.IsUnixDomainSocket(pathname); errSocket == nil && isSocket {
			return FileTypeSocket, nil
		}
	}

	return filetype, err
}

// PathExists checks whether the path exists
func (hu *HostUtil) PathExists(pathname string) (bool, error) {
	return utilpath.Exists(utilpath.CheckFollowSymlink, pathname)
}

// EvalHostSymlinks returns the path name after evaluating symlinks
func (hu *HostUtil) EvalHostSymlinks(pathname string) (string, error) {
	return filepath.EvalSymlinks(pathname)
}

// GetOwner returns the integer ID for the user and group of the given path
// Note that on windows, it always returns 0. We actually don't set Group on
// windows platform, see SetVolumeOwnership implementation.
func (hu *HostUtil) GetOwner(pathname string) (int64, int64, error) {
	return -1, -1, nil
}

// GetSELinuxSupport returns a boolean indicating support for SELinux.
// Windows does not support SELinux.
func (hu *HostUtil) GetSELinuxSupport(pathname string) (bool, error) {
	return false, nil
}

// GetMode returns permissions of the path.
func (hu *HostUtil) GetMode(pathname string) (os.FileMode, error) {
	info, err := os.Stat(pathname)
	if err != nil {
		return 0, err
	}
	return info.Mode(), nil
}

// GetSELinuxMountContext returns value of -o context=XYZ mount option on
// given mount point.
func (hu *HostUtil) GetSELinuxMountContext(pathname string) (string, error) {
	return "", nil
}
