//go:build windows
// +build windows

/*
Copyright 2019 The Kubernetes Authors.

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

package mount

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"
	"time"

	"k8s.io/klog/v2"
)

// following failure codes are from https://docs.microsoft.com/en-us/windows/desktop/debug/system-error-codes--1300-1699-
// ERROR_BAD_NETPATH                 = 53
// ERROR_NETWORK_BUSY                = 54
// ERROR_UNEXP_NET_ERR               = 59
// ERROR_NETNAME_DELETED             = 64
// ERROR_NETWORK_ACCESS_DENIED       = 65
// ERROR_BAD_DEV_TYPE                = 66
// ERROR_BAD_NET_NAME                = 67
// ERROR_SESSION_CREDENTIAL_CONFLICT = 1219
// ERROR_LOGON_FAILURE               = 1326
// WSAEHOSTDOWN                      = 10064
var errorNoList = [...]int{53, 54, 59, 64, 65, 66, 67, 1219, 1326, 10064}

// IsCorruptedMnt return true if err is about corrupted mount point
func IsCorruptedMnt(err error) bool {
	if err == nil {
		return false
	}

	var underlyingError error
	switch pe := err.(type) {
	case nil:
		return false
	case *os.PathError:
		underlyingError = pe.Err
	case *os.LinkError:
		underlyingError = pe.Err
	case *os.SyscallError:
		underlyingError = pe.Err
	}

	if ee, ok := underlyingError.(syscall.Errno); ok {
		for _, errno := range errorNoList {
			if int(ee) == errno {
				klog.Warningf("IsCorruptedMnt failed with error: %v, error code: %v", err, errno)
				return true
			}
		}
	}

	return false
}

// CleanupMountPoint unmounts the given path and deletes the remaining directory
// if successful. If extensiveMountPointCheck is true IsNotMountPoint will be
// called instead of IsLikelyNotMountPoint. IsNotMountPoint is more expensive
// but properly handles bind mounts within the same fs.
func CleanupMountPoint(mountPath string, mounter Interface, extensiveMountPointCheck bool) error {
	pathExists, pathErr := PathExists(mountPath)
	if !pathExists && pathErr == nil {
		klog.Warningf("Warning: Unmount skipped because path does not exist: %v", mountPath)
		return nil
	}
	corruptedMnt := IsCorruptedMnt(pathErr)
	if pathErr != nil && !corruptedMnt {
		return fmt.Errorf("Error checking path: %v", pathErr)
	}
	return doCleanupMountPoint(mountPath, mounter, extensiveMountPointCheck, corruptedMnt)
}

func CleanupMountWithForce(mountPath string, mounter MounterForceUnmounter, extensiveMountPointCheck bool, umountTimeout time.Duration) error {
	pathExists, pathErr := PathExists(mountPath)
	if !pathExists && pathErr == nil {
		klog.Warningf("Warning: Unmount skipped because path does not exist: %v", mountPath)
		return nil
	}
	corruptedMnt := IsCorruptedMnt(pathErr)
	if pathErr != nil && !corruptedMnt {
		return fmt.Errorf("Error checking path: %v", pathErr)
	}
	var notMnt bool
	var err error
	if !corruptedMnt {
		notMnt, err = removePathIfNotMountPoint(mountPath, mounter, extensiveMountPointCheck)
		// if mountPath was not a mount point - we would have attempted to remove mountPath
		// and hence return errors if any.
		if err != nil || notMnt {
			return err
		}
	}

	// Unmount the mount path
	klog.V(4).Infof("%q is a mountpoint, unmounting", mountPath)
	if err := mounter.UnmountWithForce(mountPath, umountTimeout); err != nil {
		return err
	}

	notMnt, err = removePathIfNotMountPoint(mountPath, mounter, extensiveMountPointCheck)
	// mountPath is not a mount point we should return whatever error we saw
	if notMnt {
		return err
	}
	return fmt.Errorf("Failed to unmount path %v", mountPath)
}

// doCleanupMountPoint unmounts the given path and
// deletes the remaining directory if successful.
// if extensiveMountPointCheck is true
// IsNotMountPoint will be called instead of IsLikelyNotMountPoint.
// IsNotMountPoint is more expensive but properly handles bind mounts within the same fs.
// if corruptedMnt is true, it means that the mountPath is a corrupted mountpoint, and the mount point check
// will be skipped
func doCleanupMountPoint(mountPath string, mounter Interface, extensiveMountPointCheck bool, corruptedMnt bool) error {
	var notMnt bool
	var err error
	if !corruptedMnt {
		notMnt, err = removePathIfNotMountPoint(mountPath, mounter, extensiveMountPointCheck)
		// if mountPath was not a mount point - we would have attempted to remove mountPath
		// and hence return errors if any.
		if err != nil || notMnt {
			return err
		}
	}

	// Unmount the mount path
	klog.V(4).Infof("%q is a mountpoint, unmounting", mountPath)
	if err := mounter.Unmount(mountPath); err != nil {
		return err
	}

	notMnt, err = removePathIfNotMountPoint(mountPath, mounter, extensiveMountPointCheck)
	// mountPath is not a mount point we should return whatever error we saw
	if notMnt {
		return err
	}
	return fmt.Errorf("Failed to unmount path %v", mountPath)
}

// removePathIfNotMountPoint verifies if given mountPath is a mount point if not it attempts
// to remove the directory. Returns true and nil if directory was not a mount point and removed.
func removePathIfNotMountPoint(mountPath string, mounter Interface, extensiveMountPointCheck bool) (bool, error) {
	var notMnt bool
	var err error

	if extensiveMountPointCheck {
		notMnt, err = IsNotMountPoint(mounter, mountPath)
	} else {
		notMnt, err = mounter.IsLikelyNotMountPoint(mountPath)
	}

	if err != nil {
		if os.IsNotExist(err) {
			klog.V(4).Infof("%q does not exist", mountPath)
			return true, nil
		}
		return notMnt, err
	}

	if notMnt {
		klog.Warningf("Warning: %q is not a mountpoint, deleting", mountPath)
		return notMnt, os.Remove(mountPath)
	}
	return notMnt, nil
}

// NormalizeWindowsPath makes sure the given path is a valid path on Windows
// systems by making sure all instances of `/` are replaced with `\\`, and the
// path beings with `c:`
func NormalizeWindowsPath(path string) string {
	normalizedPath := strings.Replace(path, "/", "\\", -1)
	if strings.HasPrefix(normalizedPath, "\\") {
		normalizedPath = "c:" + normalizedPath
	}
	return normalizedPath
}

// ValidateDiskNumber : disk number should be a number in [0, 99]
func ValidateDiskNumber(disk string) error {
	if _, err := strconv.Atoi(disk); err != nil {
		return fmt.Errorf("wrong disk number format: %q, err: %v", disk, err)
	}
	return nil
}

// isMountPointMatch determines if the mountpoint matches the dir
func isMountPointMatch(mp MountPoint, dir string) bool {
	return mp.Path == dir
}

// PathExists returns true if the specified path exists.
// TODO: clean this up to use pkg/util/file/FileExists
func PathExists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	} else if os.IsNotExist(err) {
		return false, nil
	} else if IsCorruptedMnt(err) {
		return true, err
	}
	return false, err
}
