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

	"k8s.io/utils/mount"
)

// FileType enumerates the known set of possible file types.
type FileType string

const (
	// FileTypeBlockDev defines a constant for the block device FileType.
	FileTypeBlockDev FileType = "BlockDevice"
	// FileTypeCharDev defines a constant for the character device FileType.
	FileTypeCharDev FileType = "CharDevice"
	// FileTypeDirectory defines a constant for the directory FileType.
	FileTypeDirectory FileType = "Directory"
	// FileTypeFile defines a constant for the file FileType.
	FileTypeFile FileType = "File"
	// FileTypeSocket defines a constant for the socket FileType.
	FileTypeSocket FileType = "Socket"
	// FileTypeUnknown defines a constant for an unknown FileType.
	FileTypeUnknown FileType = ""
)

// HostUtils defines the set of methods for interacting with paths on a host.
type HostUtils interface {
	// DeviceOpened determines if the device (e.g. /dev/sdc) is in use elsewhere
	// on the system, i.e. still mounted.
	DeviceOpened(pathname string) (bool, error)
	// PathIsDevice determines if a path is a device.
	PathIsDevice(pathname string) (bool, error)
	// GetDeviceNameFromMount finds the device name by checking the mount path
	// to get the global mount path within its plugin directory.
	GetDeviceNameFromMount(mounter mount.Interface, mountPath, pluginMountDir string) (string, error)
	// MakeRShared checks that given path is on a mount with 'rshared' mount
	// propagation. If not, it bind-mounts the path as rshared.
	MakeRShared(path string) error
	// GetFileType checks for file/directory/socket/block/character devices.
	GetFileType(pathname string) (FileType, error)
	// PathExists tests if the given path already exists
	// Error is returned on any other error than "file not found".
	PathExists(pathname string) (bool, error)
	// EvalHostSymlinks returns the path name after evaluating symlinks.
	EvalHostSymlinks(pathname string) (string, error)
	// GetOwner returns the integer ID for the user and group of the given path
	GetOwner(pathname string) (int64, int64, error)
	// GetSELinuxSupport returns true if given path is on a mount that supports
	// SELinux.
	GetSELinuxSupport(pathname string) (bool, error)
	// GetMode returns permissions of the path.
	GetMode(pathname string) (os.FileMode, error)
}

// Compile-time check to ensure all HostUtil implementations satisfy
// the Interface.
var _ HostUtils = &HostUtil{}

// getFileType checks for file/directory/socket and block/character devices.
func getFileType(pathname string) (FileType, error) {
	var pathType FileType
	info, err := os.Stat(pathname)
	if os.IsNotExist(err) {
		return pathType, fmt.Errorf("path %q does not exist", pathname)
	}
	// err in call to os.Stat
	if err != nil {
		return pathType, err
	}

	// checks whether the mode is the target mode.
	isSpecificMode := func(mode, targetMode os.FileMode) bool {
		return mode&targetMode == targetMode
	}

	mode := info.Mode()
	if mode.IsDir() {
		return FileTypeDirectory, nil
	} else if mode.IsRegular() {
		return FileTypeFile, nil
	} else if isSpecificMode(mode, os.ModeSocket) {
		return FileTypeSocket, nil
	} else if isSpecificMode(mode, os.ModeDevice) {
		if isSpecificMode(mode, os.ModeCharDevice) {
			return FileTypeCharDev, nil
		}
		return FileTypeBlockDev, nil
	}

	return pathType, fmt.Errorf("only recognise file, directory, socket, block device and character device")
}
