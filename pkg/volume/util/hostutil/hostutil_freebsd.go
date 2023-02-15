//go:build freebsd
// +build freebsd

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

package hostutil

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
	"syscall"

	"golang.org/x/sys/unix"
	"k8s.io/klog/v2"
	"k8s.io/mount-utils"
	utilpath "k8s.io/utils/path"
)

const (
	// Location of the mountinfo file
	procMountInfoPath = "/proc/self/mountinfo"
)

// HostUtil implements HostUtils for Linux platforms.
type HostUtil struct {
}

// NewHostUtil returns a struct that implements the HostUtils interface on
// linux platforms
func NewHostUtil() *HostUtil {
	return &HostUtil{}
}

// DeviceOpened checks if block device in use by calling Open with O_EXCL flag.
// If pathname is not a device, log and return false with nil error.
// If open returns errno EBUSY, return true with nil error.
// If open returns nil, return false with nil error.
// Otherwise, return false with error
func (hu *HostUtil) DeviceOpened(pathname string) (bool, error) {
	return ExclusiveOpenFailsOnDevice(pathname)
}

// PathIsDevice uses FileInfo returned from os.Stat to check if path refers
// to a device.
func (hu *HostUtil) PathIsDevice(pathname string) (bool, error) {
	pathType, err := hu.GetFileType(pathname)
	isDevice := pathType == FileTypeCharDev || pathType == FileTypeBlockDev
	return isDevice, err
}

// ExclusiveOpenFailsOnDevice is shared with NsEnterMounter
func ExclusiveOpenFailsOnDevice(pathname string) (bool, error) {
	var isDevice bool
	finfo, err := os.Stat(pathname)
	if os.IsNotExist(err) {
		isDevice = false
	}
	// err in call to os.Stat
	if err != nil {
		return false, fmt.Errorf(
			"PathIsDevice failed for path %q: %v",
			pathname,
			err)
	}
	// path refers to a device
	if finfo.Mode()&os.ModeDevice != 0 {
		isDevice = true
	}

	if !isDevice {
		klog.Errorf("Path %q is not referring to a device.", pathname)
		return false, nil
	}
	fd, errno := unix.Open(pathname, unix.O_RDONLY|unix.O_EXCL|unix.O_CLOEXEC, 0)
	// If the device is in use, open will return an invalid fd.
	// When this happens, it is expected that Close will fail and throw an error.
	defer unix.Close(fd)
	if errno == nil {
		// device not in use
		return false, nil
	} else if errno == unix.EBUSY {
		// device is in use
		return true, nil
	}
	// error during call to Open
	return false, errno
}

// GetDeviceNameFromMount given a mount point, find the device name from its global mount point
func (hu *HostUtil) GetDeviceNameFromMount(mounter mount.Interface, mountPath, pluginMountDir string) (string, error) {
	return getDeviceNameFromMount(mounter, mountPath, pluginMountDir)
}

// getDeviceNameFromMountLinux find the device name from /proc/mounts in which
// the mount path reference should match the given plugin mount directory. In case no mount path reference
// matches, returns the volume name taken from its given mountPath
func getDeviceNameFromMount(mounter mount.Interface, mountPath, pluginMountDir string) (string, error) {
	refs, err := mounter.GetMountRefs(mountPath)
	if err != nil {
		klog.V(4).Infof("GetMountRefs failed for mount path %q: %v", mountPath, err)
		return "", err
	}
	if len(refs) == 0 {
		klog.V(4).Infof("Directory %s is not mounted", mountPath)
		return "", fmt.Errorf("directory %s is not mounted", mountPath)
	}
	for _, ref := range refs {
		if strings.HasPrefix(ref, pluginMountDir) {
			volumeID, err := filepath.Rel(pluginMountDir, ref)
			if err != nil {
				klog.Errorf("Failed to get volume id from mount %s - %v", mountPath, err)
				return "", err
			}
			return volumeID, nil
		}
	}

	return path.Base(mountPath), nil
}

// MakeRShared checks that given path is on a mount with 'rshared' mount
// propagation. Empty implementation here.
func (hu *HostUtil) MakeRShared(path string) error {
	return nil
}

// GetFileType checks for file/directory/socket/block/character devices.
func (hu *HostUtil) GetFileType(pathname string) (FileType, error) {
	return getFileType(pathname)
}

// PathExists tests if the given path already exists
// Error is returned on any other error than "file not found".
func (hu *HostUtil) PathExists(pathname string) (bool, error) {
	return utilpath.Exists(utilpath.CheckFollowSymlink, pathname)
}

// EvalHostSymlinks returns the path name after evaluating symlinks.
// TODO once the nsenter implementation is removed, this method can be removed
// from the interface and filepath.EvalSymlinks used directly
func (hu *HostUtil) EvalHostSymlinks(pathname string) (string, error) {
	return filepath.EvalSymlinks(pathname)
}

// FindMountInfo returns the mount info on the given path.
func (hu *HostUtil) FindMountInfo(path string) (mount.MountInfo, error) {
	return findMountInfo(path, procMountInfoPath)
}

func findMountInfo(path, mountInfoPath string) (mount.MountInfo, error) {
	infos, err := mount.ParseMountInfo(mountInfoPath)
	if err != nil {
		return mount.MountInfo{}, err
	}

	// process /proc/xxx/mountinfo in backward order and find the first mount
	// point that is prefix of 'path' - that's the mount where path resides
	var info *mount.MountInfo
	for i := len(infos) - 1; i >= 0; i-- {
		if mount.PathWithinBase(path, infos[i].MountPoint) {
			info = &infos[i]
			break
		}
	}
	if info == nil {
		return mount.MountInfo{}, fmt.Errorf("cannot find mount point for %q", path)
	}
	return *info, nil
}

// selinux.SELinuxEnabled implementation for unit tests
type seLinuxEnabledFunc func() bool

// GetSELinux is not supported on FreeBSD
func GetSELinux(path string, mountInfoFilename string, selinuxEnabled seLinuxEnabledFunc) (bool, error) {
	return false, nil
}

// GetSELinuxSupport return false on FreeBSD
func (hu *HostUtil) GetSELinuxSupport(pathname string) (bool, error) {
	return false, nil
}

// GetOwner returns the integer ID for the user and group of the given path
func (hu *HostUtil) GetOwner(pathname string) (int64, int64, error) {
	realpath, err := filepath.EvalSymlinks(pathname)
	if err != nil {
		return -1, -1, err
	}

	info, err := os.Stat(realpath)
	if err != nil {
		return -1, -1, err
	}
	stat := info.Sys().(*syscall.Stat_t)
	return int64(stat.Uid), int64(stat.Gid), nil
}

// GetMode returns permissions of the path.
func (hu *HostUtil) GetMode(pathname string) (os.FileMode, error) {
	info, err := os.Stat(pathname)
	if err != nil {
		return 0, err
	}
	return info.Mode(), nil
}

// GetSELinuxMountContext is not supported on FreeBSD
func (hu *HostUtil) GetSELinuxMountContext(pathname string) (string, error) {
	return "", nil
}

