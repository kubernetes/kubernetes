//go:build linux
// +build linux

/*
Copyright 2021 The Kubernetes Authors.

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
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	utilexec "k8s.io/utils/exec"
)

const (
	blockDev = "blockdev"
)

// ResizeFs Provides support for resizing file systems
type ResizeFs struct {
	exec utilexec.Interface
}

// NewResizeFs returns new instance of resizer
func NewResizeFs(exec utilexec.Interface) *ResizeFs {
	return &ResizeFs{exec: exec}
}

// Resize perform resize of file system
func (resizefs *ResizeFs) Resize(devicePath string, deviceMountPath string) (bool, error) {
	format, err := getDiskFormat(resizefs.exec, devicePath)
	if err != nil {
		formatErr := fmt.Errorf("ResizeFS.Resize - error checking format for device %s: %v", devicePath, err)
		return false, formatErr
	}

	// If disk has no format, there is no need to resize the disk because mkfs.*
	// by default will use whole disk anyways.
	if format == "" {
		return false, nil
	}

	klog.V(3).Infof("ResizeFS.Resize - Expanding mounted volume %s", devicePath)
	switch format {
	case "ext3", "ext4":
		return resizefs.extResize(devicePath)
	case "xfs":
		return resizefs.xfsResize(deviceMountPath)
	case "btrfs":
		return resizefs.btrfsResize(deviceMountPath)
	}
	return false, fmt.Errorf("ResizeFS.Resize - resize of format %s is not supported for device %s mounted at %s", format, devicePath, deviceMountPath)
}

func (resizefs *ResizeFs) extResize(devicePath string) (bool, error) {
	output, err := resizefs.exec.Command("resize2fs", devicePath).CombinedOutput()
	if err == nil {
		klog.V(2).Infof("Device %s resized successfully", devicePath)
		return true, nil
	}

	resizeError := fmt.Errorf("resize of device %s failed: %v. resize2fs output: %s", devicePath, err, string(output))
	return false, resizeError
}

func (resizefs *ResizeFs) xfsResize(deviceMountPath string) (bool, error) {
	args := []string{"-d", deviceMountPath}
	output, err := resizefs.exec.Command("xfs_growfs", args...).CombinedOutput()

	if err == nil {
		klog.V(2).Infof("Device %s resized successfully", deviceMountPath)
		return true, nil
	}

	resizeError := fmt.Errorf("resize of device %s failed: %v. xfs_growfs output: %s", deviceMountPath, err, string(output))
	return false, resizeError
}

func (resizefs *ResizeFs) btrfsResize(deviceMountPath string) (bool, error) {
	args := []string{"filesystem", "resize", "max", deviceMountPath}
	output, err := resizefs.exec.Command("btrfs", args...).CombinedOutput()

	if err == nil {
		klog.V(2).Infof("Device %s resized successfully", deviceMountPath)
		return true, nil
	}

	resizeError := fmt.Errorf("resize of device %s failed: %v. btrfs output: %s", deviceMountPath, err, string(output))
	return false, resizeError
}

func (resizefs *ResizeFs) NeedResize(devicePath string, deviceMountPath string) (bool, error) {
	// Do nothing if device is mounted as readonly
	readonly, err := resizefs.getDeviceRO(devicePath)
	if err != nil {
		return false, err
	}

	if readonly {
		klog.V(3).Infof("ResizeFs.needResize - no resize possible since filesystem %s is readonly", devicePath)
		return false, nil
	}

	format, err := getDiskFormat(resizefs.exec, devicePath)
	if err != nil {
		formatErr := fmt.Errorf("ResizeFS.Resize - error checking format for device %s: %v", devicePath, err)
		return false, formatErr
	}

	// If disk has no format, there is no need to resize the disk because mkfs.*
	// by default will use whole disk anyways.
	if format == "" {
		return false, nil
	}

	supportedFormats := sets.New("ext3", "ext4", "xfs", "btrfs")
	if !supportedFormats.Has(format) {
		return false, fmt.Errorf("could not parse fs info of given filesystem format: %s. Supported fs types are: xfs, ext3, ext4", format)
	}
	return true, nil
}

func (resizefs *ResizeFs) getDeviceRO(devicePath string) (bool, error) {
	output, err := resizefs.exec.Command(blockDev, "--getro", devicePath).CombinedOutput()
	outStr := strings.TrimSpace(string(output))
	if err != nil {
		return false, fmt.Errorf("failed to get readonly bit from device %s: %s: %s", devicePath, err, outStr)
	}
	switch outStr {
	case "0":
		return false, nil
	case "1":
		return true, nil
	default:
		return false, fmt.Errorf("failed readonly device check. Expected 1 or 0, got '%s'", outStr)
	}
}
