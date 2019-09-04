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
	"io/ioutil"
	"path/filepath"
	"strings"

	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/util/mount"
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
func (resizefs *ResizeFs) Resize(devicePath string, deviceMountPath string) (bool, error) {
	format, err := resizefs.mounter.GetDiskFormat(devicePath)

	if err != nil {
		formatErr := fmt.Errorf("ResizeFS.Resize - error checking format for device %s: %v", devicePath, err)
		return false, formatErr
	}

	// If disk has no format, there is no need to resize the disk because mkfs.*
	// by default will use whole disk anyways.
	if format == "" {
		return false, nil
	}

	// don't fail if resolving doesn't work
	if blockDeviceRescanPath, err := findBlockDeviceRescanPath(devicePath); err != nil {
		klog.V(0).Infof("ResizeFS.Resize - error resolving block device path from %q: %v", devicePath, err)
	} else {
		klog.V(3).Infof("ResizeFS.Resize - resolved block device path from %q to %q", devicePath, blockDeviceRescanPath)

		klog.V(3).Infof("ResizeFS.Resize - polling %q block device geometry", devicePath)
		err = ioutil.WriteFile(blockDeviceRescanPath, []byte{'1'}, 0666)
		if err != nil {
			klog.V(0).Infof("ResizeFS.Resize - error polling new block device geometry: %v", err)
		}
	}

	klog.V(3).Infof("ResizeFS.Resize - Expanding mounted volume %s", devicePath)
	switch format {
	case "ext3", "ext4":
		return resizefs.extResize(devicePath)
	case "xfs":
		return resizefs.xfsResize(deviceMountPath)
	}
	return false, fmt.Errorf("ResizeFS.Resize - resize of format %s is not supported for device %s mounted at %s", format, devicePath, deviceMountPath)
}

func (resizefs *ResizeFs) extResize(devicePath string) (bool, error) {
	output, err := resizefs.mounter.Exec.Run("resize2fs", devicePath)
	if err == nil {
		klog.V(2).Infof("Device %s resized successfully", devicePath)
		return true, nil
	}

	resizeError := fmt.Errorf("resize of device %s failed: %v. resize2fs output: %s", devicePath, err, string(output))
	return false, resizeError

}

func (resizefs *ResizeFs) xfsResize(deviceMountPath string) (bool, error) {
	args := []string{"-d", deviceMountPath}
	output, err := resizefs.mounter.Exec.Run("xfs_growfs", args...)

	if err == nil {
		klog.V(2).Infof("Device %s resized successfully", deviceMountPath)
		return true, nil
	}

	resizeError := fmt.Errorf("resize of device %s failed: %v. xfs_growfs output: %s", deviceMountPath, err, string(output))
	return false, resizeError
}

// findBlockDeviceRescanPath Find the underlaying disk for a linked path such as /dev/disk/by-path/XXXX or /dev/mapper/XXXX
// will return /sys/devices/pci0000:00/0000:00:15.0/0000:03:00.0/host0/target0:0:1/0:0:1:0/rescan
func findBlockDeviceRescanPath(path string) (string, error) {
	devicePath, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", err
	}
	// if path /dev/hdX split into "", "dev", "hdX" then we will
	// return just the last part
	parts := strings.Split(devicePath, "/")
	if len(parts) == 3 && strings.HasPrefix(parts[1], "dev") {
		blockPath := "/sys/block" + "/" + parts[2] + "/device/rescan"
		return filepath.EvalSymlinks(blockPath)
	}
	return "", fmt.Errorf("Illegal path for device " + devicePath)
}
