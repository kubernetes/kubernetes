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
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"

	"golang.org/x/sys/windows"
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

func IsPathValid(path string) (bool, error) {
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

	klog.V(6).Infof("Path %s attribute: %O", path, attrs)
	return attrs != windows.INVALID_FILE_ATTRIBUTES, nil
}

// IsMountedFolder checks whether the `path` is a mounted folder.
func IsMountedFolder(path string) (bool, error) {
	// https://learn.microsoft.com/en-us/windows/win32/fileio/determining-whether-a-directory-is-a-volume-mount-point
	utf16Path, _ := windows.UTF16PtrFromString(path)
	attrs, err := windows.GetFileAttributes(utf16Path)
	if err != nil {
		return false, err
	}

	if (attrs & windows.FILE_ATTRIBUTE_REPARSE_POINT) == 0 {
		return false, nil
	}

	var findData windows.Win32finddata
	findHandle, err := windows.FindFirstFile(utf16Path, &findData)
	if err != nil && !errors.Is(err, windows.ERROR_NO_MORE_FILES) {
		return false, err
	}

	for err == nil {
		if findData.Reserved0&windows.IO_REPARSE_TAG_MOUNT_POINT != 0 {
			return true, nil
		}

		err = windows.FindNextFile(findHandle, &findData)
		if err != nil && !errors.Is(err, windows.ERROR_NO_MORE_FILES) {
			return false, err
		}
	}

	return false, nil
}
