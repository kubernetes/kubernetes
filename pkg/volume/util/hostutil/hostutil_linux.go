//go:build linux
// +build linux

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
	"k8s.io/kubernetes/pkg/util/selinux"
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
// propagation. If not, it bind-mounts the path as rshared.
func (hu *HostUtil) MakeRShared(path string) error {
	return DoMakeRShared(path, procMountInfoPath)
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

// isShared returns true, if given path is on a mount point that has shared
// mount propagation.
func isShared(mount string, mountInfoPath string) (bool, error) {
	info, err := findMountInfo(mount, mountInfoPath)
	if err != nil {
		return false, err
	}

	// parse optional parameters
	for _, opt := range info.OptionalFields {
		if strings.HasPrefix(opt, "shared:") {
			return true, nil
		}
	}
	return false, nil
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

// DoMakeRShared is common implementation of MakeRShared on Linux. It checks if
// path is shared and bind-mounts it as rshared if needed. mountCmd and
// mountArgs are expected to contain mount-like command, DoMakeRShared will add
// '--bind <path> <path>' and '--make-rshared <path>' to mountArgs.
func DoMakeRShared(path string, mountInfoFilename string) error {
	shared, err := isShared(path, mountInfoFilename)
	if err != nil {
		return err
	}
	if shared {
		klog.V(4).Infof("Directory %s is already on a shared mount", path)
		return nil
	}

	klog.V(2).Infof("Bind-mounting %q with shared mount propagation", path)
	// mount --bind /var/lib/kubelet /var/lib/kubelet
	if err := syscall.Mount(path, path, "" /*fstype*/, syscall.MS_BIND, "" /*data*/); err != nil {
		return fmt.Errorf("failed to bind-mount %s: %v", path, err)
	}

	// mount --make-rshared /var/lib/kubelet
	if err := syscall.Mount(path, path, "" /*fstype*/, syscall.MS_SHARED|syscall.MS_REC, "" /*data*/); err != nil {
		return fmt.Errorf("failed to make %s rshared: %v", path, err)
	}

	return nil
}

// selinux.SELinuxEnabled implementation for unit tests
type seLinuxEnabledFunc func() bool

// GetSELinux is common implementation of GetSELinuxSupport on Linux.
func GetSELinux(path string, mountInfoFilename string, selinuxEnabled seLinuxEnabledFunc) (bool, error) {
	// Skip /proc/mounts parsing if SELinux is disabled.
	if !selinuxEnabled() {
		return false, nil
	}

	info, err := findMountInfo(path, mountInfoFilename)
	if err != nil {
		return false, err
	}

	// "seclabel" can be both in mount options and super options.
	for _, opt := range info.SuperOptions {
		if opt == "seclabel" {
			return true, nil
		}
	}
	for _, opt := range info.MountOptions {
		if opt == "seclabel" {
			return true, nil
		}
	}
	return false, nil
}

// GetSELinuxSupport returns true if given path is on a mount that supports
// SELinux.
func (hu *HostUtil) GetSELinuxSupport(pathname string) (bool, error) {
	return GetSELinux(pathname, procMountInfoPath, selinux.SELinuxEnabled)
}

// GetOwner returns the integer ID for the user and group of the given path
func (hu *HostUtil) GetOwner(pathname string) (int64, int64, error) {
	realpath, err := filepath.EvalSymlinks(pathname)
	if err != nil {
		return -1, -1, err
	}
	return GetOwnerLinux(realpath)
}

// GetMode returns permissions of the path.
func (hu *HostUtil) GetMode(pathname string) (os.FileMode, error) {
	return GetModeLinux(pathname)
}

// GetOwnerLinux is shared between Linux and NsEnterMounter
// pathname must already be evaluated for symlinks
func GetOwnerLinux(pathname string) (int64, int64, error) {
	info, err := os.Stat(pathname)
	if err != nil {
		return -1, -1, err
	}
	stat := info.Sys().(*syscall.Stat_t)
	return int64(stat.Uid), int64(stat.Gid), nil
}

// GetModeLinux is shared between Linux and NsEnterMounter
func GetModeLinux(pathname string) (os.FileMode, error) {
	info, err := os.Stat(pathname)
	if err != nil {
		return 0, err
	}
	return info.Mode(), nil
}
