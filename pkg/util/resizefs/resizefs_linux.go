// +build linux

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

package resizefs

import (
	"fmt"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/mount"
	utilexec "k8s.io/utils/exec"
)

const (
	// 'fsck' found errors and corrected them
	fsckErrorsCorrected = 1
	// 'fsck' found errors but exited without correcting them
	fsckErrorsUncorrected = 4
)

// ResizeFs Provides support for resizing file systems
type ResizeFs struct {
	mounter *mount.SafeFormatAndMount
}

// NewResizeFs returns new instance of resizer
func NewResizeFs(mounter *mount.SafeFormatAndMount) *ResizeFs {
	return &ResizeFs{mounter: mounter}
}

// Resize perform resize of file system
func (resizefs *ResizeFs) Resize(devicePath string) (bool, error) {
	format, err := resizefs.mounter.GetDiskFormat(devicePath)

	if err != nil {
		formatErr := fmt.Errorf("error checking format for device %s: %v", devicePath, err)
		return false, formatErr
	}

	// If disk has no format, there is no need to resize the disk because mkfs.*
	// by default will use whole disk anyways.
	if format == "" {
		return false, nil
	}

	deviceOpened, err := resizefs.mounter.DeviceOpened(devicePath)

	if err != nil {
		deviceOpenErr := fmt.Errorf("error verifying if device %s is open: %v", devicePath, err)
		return false, deviceOpenErr
	}

	if deviceOpened {
		deviceAlreadyOpenErr := fmt.Errorf("the device %s is already in use", devicePath)
		return false, deviceAlreadyOpenErr
	}

	switch format {
	case "ext3", "ext4":
		fsckErr := resizefs.extFsck(devicePath, format)
		if fsckErr != nil {
			return false, fsckErr
		}
		return resizefs.extResize(devicePath)
	case "xfs":
		fsckErr := resizefs.fsckDevice(devicePath)
		if fsckErr != nil {
			return false, fsckErr
		}
		return resizefs.xfsResize(devicePath)
	}
	return false, fmt.Errorf("resize of format %s is not supported for device %s", format, devicePath)
}

func (resizefs *ResizeFs) fsckDevice(devicePath string) error {
	glog.V(4).Infof("Checking for issues with fsck on device: %s", devicePath)
	args := []string{"-a", devicePath}
	out, err := resizefs.mounter.Exec.Run("fsck", args...)
	if err != nil {
		ee, isExitError := err.(utilexec.ExitError)
		switch {
		case err == utilexec.ErrExecutableNotFound:
			glog.Warningf("'fsck' not found on system; continuing resizing without running 'fsck'.")
		case isExitError && ee.ExitStatus() == fsckErrorsCorrected:
			glog.V(2).Infof("Device %s has errors which were corrected by fsck: %s", devicePath, string(out))
		case isExitError && ee.ExitStatus() == fsckErrorsUncorrected:
			return fmt.Errorf("'fsck' found errors on device %s but could not correct them: %s", devicePath, string(out))
		case isExitError && ee.ExitStatus() > fsckErrorsUncorrected:
			glog.Infof("`fsck` error %s", string(out))
		}
	}
	return nil
}

func (resizefs *ResizeFs) extFsck(devicePath string, fsType string) error {
	glog.V(4).Infof("Checking for issues with fsck.%s on device: %s", fsType, devicePath)
	args := []string{"-f", "-y", devicePath}
	out, err := resizefs.mounter.Run("fsck."+fsType, args...)
	if err != nil {
		return fmt.Errorf("running fsck.%s failed on %s with error: %v\n Output: %s", fsType, devicePath, err, string(out))
	}
	return nil
}

func (resizefs *ResizeFs) extResize(devicePath string) (bool, error) {
	output, err := resizefs.mounter.Exec.Run("resize2fs", devicePath)
	if err == nil {
		glog.V(2).Infof("Device %s resized successfully", devicePath)
		return true, nil
	}

	resizeError := fmt.Errorf("resize of device %s failed: %v. resize2fs output: %s", devicePath, err, string(output))
	return false, resizeError

}

func (resizefs *ResizeFs) xfsResize(devicePath string) (bool, error) {
	args := []string{"-d", devicePath}
	output, err := resizefs.mounter.Exec.Run("xfs_growfs", args...)

	if err == nil {
		glog.V(2).Infof("Device %s resized successfully", devicePath)
		return true, nil
	}

	resizeError := fmt.Errorf("resize of device %s failed: %v. xfs_growfs output: %s", devicePath, err, string(output))
	return false, resizeError
}
