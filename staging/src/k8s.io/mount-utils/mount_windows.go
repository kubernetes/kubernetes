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

package mount

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
	"golang.org/x/sys/windows"
	"k8s.io/klog/v2"
	"k8s.io/utils/keymutex"
)

// Mounter provides the default implementation of mount.Interface
// for the windows platform.  This implementation assumes that the
// kubelet is running in the host's root mount namespace.
type Mounter struct {
	mounterPath string
}

// New returns a mount.Interface for the current system.
// It provides options to override the default mounter behavior.
// mounterPath allows using an alternative to `/bin/mount` for mounting.
func New(mounterPath string) Interface {
	return &Mounter{
		mounterPath: mounterPath,
	}
}

// acquire lock for smb mount
var getSMBMountMutex = keymutex.NewHashed(0)

// Mount : mounts source to target with given options.
// currently only supports cifs(smb), bind mount(for disk)
func (mounter *Mounter) Mount(source string, target string, fstype string, options []string) error {
	return mounter.MountSensitive(source, target, fstype, options, nil /* sensitiveOptions */)
}

// MountSensitiveWithoutSystemd is the same as MountSensitive() but disable using ssytemd mount.
// Windows not supported systemd mount, this function degrades to MountSensitive().
func (mounter *Mounter) MountSensitiveWithoutSystemd(source string, target string, fstype string, options []string, sensitiveOptions []string) error {
	return mounter.MountSensitive(source, target, fstype, options, sensitiveOptions /* sensitiveOptions */)
}

// MountSensitiveWithoutSystemdWithMountFlags is the same as MountSensitiveWithoutSystemd with additional mount flags
// Windows not supported systemd mount, this function degrades to MountSensitive().
func (mounter *Mounter) MountSensitiveWithoutSystemdWithMountFlags(source string, target string, fstype string, options []string, sensitiveOptions []string, mountFlags []string) error {
	return mounter.MountSensitive(source, target, fstype, options, sensitiveOptions /* sensitiveOptions */)
}

// MountSensitive is the same as Mount() but this method allows
// sensitiveOptions to be passed in a separate parameter from the normal
// mount options and ensures the sensitiveOptions are never logged. This
// method should be used by callers that pass sensitive material (like
// passwords) as mount options.
func (mounter *Mounter) MountSensitive(source string, target string, fstype string, options []string, sensitiveOptions []string) error {
	target = NormalizeWindowsPath(target)
	sanitizedOptionsForLogging := sanitizedOptionsForLogging(options, sensitiveOptions)

	if source == "tmpfs" {
		klog.V(3).Infof("mounting source (%q), target (%q), with options (%q)", source, target, sanitizedOptionsForLogging)
		return os.MkdirAll(target, 0o755)
	}

	parentDir := filepath.Dir(target)
	if err := os.MkdirAll(parentDir, 0o755); err != nil {
		return err
	}

	klog.V(4).Infof("mount options(%q) source:%q, target:%q, fstype:%q, begin to mount",
		sanitizedOptionsForLogging, source, target, fstype)
	bindSource := source

	if bind, _, _, _ := MakeBindOptsSensitive(options, sensitiveOptions); bind {
		bindSource = NormalizeWindowsPath(source)
	} else {
		allOptions := []string{}
		allOptions = append(allOptions, options...)
		allOptions = append(allOptions, sensitiveOptions...)
		if len(allOptions) < 2 {
			return fmt.Errorf("mount options(%q) should have at least 2 options, current number:%d, source:%q, target:%q",
				sanitizedOptionsForLogging, len(allOptions), source, target)
		}

		// currently only cifs mount is supported
		if strings.ToLower(fstype) != "cifs" {
			return fmt.Errorf("only cifs mount is supported now, fstype: %q, mounting source (%q), target (%q), with options (%q)", fstype, source, target, sanitizedOptionsForLogging)
		}

		// lock smb mount for the same source
		getSMBMountMutex.LockKey(source)
		defer getSMBMountMutex.UnlockKey(source)

		username := allOptions[0]
		password := allOptions[1]
		cimOps := cimOperations{}
		if err := cimOps.NewSMBMapping(username, password, source); err != nil {
			klog.Warningf("SMB Mapping(%s) returned with error(%v)", source, err)
			exist, err := cimOps.IsSMBMappingExist(source)
			if err != nil {
				return fmt.Errorf("IsSMBMappingExist failed: %v", err)
			}
			if exist {
				valid, err := isValidPath(source)
				if !valid {
					if err == nil || isAccessDeniedError(err) {
						klog.V(2).Infof("SMB Mapping(%s) already exists while it's not valid, return error: %v, now begin to remove and remount", source, err)
						if err = cimOps.RemoveSMBMapping(source); err != nil {
							return fmt.Errorf("removeSMBMapping failed: %v", err)
						}
						if err := cimOps.NewSMBMapping(username, password, source); err != nil {
							return fmt.Errorf("newSMBMapping(%s) failed: %v", source, err)
						}
					}
				} else {
					klog.V(2).Infof("SMB Mapping(%s) already exists and is still valid, skip error(%v)", source, err)
				}
			} else {
				return fmt.Errorf("newSMBMapping(%s) failed: %v", source, err)
			}
		}
	}

	// There is an issue in golang where EvalSymlinks fails on Windows when passed a
	// UNC share root path without a trailing backslash.
	// Ex: \\SERVER\share will fail to resolve but \\SERVER\share\ will resolve
	// containerD on Windows calls EvalSymlinks so we'll add the backslash when making the symlink if it is missing.
	// https://github.com/golang/go/pull/42096 fixes this issue in golang but a fix will not be available until
	// golang v1.16
	mklinkSource := bindSource
	if !strings.HasSuffix(mklinkSource, "\\") {
		mklinkSource = mklinkSource + "\\"
	}

	err := os.Symlink(mklinkSource, target)
	if err != nil {
		klog.Errorf("symlink failed: %v, source(%q) target(%q)", err, mklinkSource, target)
		return err
	}
	klog.V(2).Infof("symlink source(%q) on target(%q) successfully", mklinkSource, target)

	return nil
}

// check whether remotepath is valid
// return (true, nil) if remotepath is valid
func isValidPath(path string) (bool, error) {
	pathString, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return false, fmt.Errorf("invalid path: %w", err)
	}

	attrs, err := windows.GetFileAttributes(pathString)
	if err != nil {
		if errors.Is(err, windows.ERROR_PATH_NOT_FOUND) || errors.Is(err, windows.ERROR_FILE_NOT_FOUND) || errors.Is(err, windows.ERROR_INVALID_NAME) {
			return false, nil
		}

		// GetFileAttribute returns user or password incorrect for a disconnected SMB connection after the password is changed
		return false, fmt.Errorf("failed to get path %s attribute: %w", path, err)
	}

	klog.V(6).Infof("Path %s attribute: %d", path, attrs)
	return attrs != windows.INVALID_FILE_ATTRIBUTES, nil
}

func isAccessDeniedError(err error) bool {
	return errors.Is(err, windows.ERROR_ACCESS_DENIED)
}

// Unmount unmounts the target.
func (mounter *Mounter) Unmount(target string) error {
	klog.V(4).Infof("Unmount target (%q)", target)
	target = NormalizeWindowsPath(target)

	if err := os.Remove(target); err != nil {
		klog.Errorf("removing directory %s failed: %v", target, err)
		return err
	}
	return nil
}

// List returns a list of all mounted filesystems. todo
func (mounter *Mounter) List() ([]MountPoint, error) {
	return []MountPoint{}, nil
}

// IsLikelyNotMountPoint determines if a directory is not a mountpoint.
func (mounter *Mounter) IsLikelyNotMountPoint(file string) (bool, error) {
	isMnt, err := mounter.IsMountPoint(file)
	if err != nil {
		return false, err
	}
	return !isMnt, nil
}

// CanSafelySkipMountPointCheck always returns false on Windows
func (mounter *Mounter) CanSafelySkipMountPointCheck() bool {
	return false
}

// IsMountPoint determines if a directory is a mountpoint.
func (mounter *Mounter) IsMountPoint(path string) (bool, error) {
	return IsMountedFolder(path)
}

// GetMountRefs : empty implementation here since there is no place to query all mount points on Windows
func (mounter *Mounter) GetMountRefs(pathname string) ([]string, error) {
	windowsPath := NormalizeWindowsPath(pathname)
	pathExists, pathErr := PathExists(windowsPath)
	if !pathExists {
		return []string{}, nil
	} else if IsCorruptedMnt(pathErr) {
		klog.Warningf("GetMountRefs found corrupted mount at %s, treating as unmounted path", windowsPath)
		return []string{}, nil
	} else if pathErr != nil {
		return nil, fmt.Errorf("error checking path %s: %v", windowsPath, pathErr)
	}
	return []string{pathname}, nil
}

func (mounter *SafeFormatAndMount) formatAndMountSensitive(source string, target string, fstype string, options []string, sensitiveOptions []string, formatOptions []string) error {
	// Try to mount the disk
	klog.V(4).Infof("Attempting to formatAndMount disk: %s %s %s", fstype, source, target)

	sourceDiskNumber, err := ValidateDiskNumber(source)
	if err != nil {
		klog.Errorf("diskMount: formatAndMount failed, err: %v", err)
		return err
	}

	if len(fstype) == 0 {
		// Use 'NTFS' as the default
		fstype = "NTFS"
	}

	if len(formatOptions) > 0 {
		return fmt.Errorf("diskMount: formatOptions are not supported on Windows")
	}

	cimOps := cimOperations{}
	err = cimOps.PartitionDisk(sourceDiskNumber)
	if err != nil {
		return fmt.Errorf("diskMount: format disk failed, error: %v, output: %q", err)
	}
	klog.V(4).Infof("diskMount: Disk successfully formatted, disk: %q, fstype: %q", source, fstype)

	volumeIds, err := cimOps.ListVolumesOnDisk(sourceDiskNumber)
	if err != nil {
		return err
	}
	driverPath := volumeIds[0]
	return mounter.MountSensitive(driverPath, target, fstype, options, sensitiveOptions)
}
