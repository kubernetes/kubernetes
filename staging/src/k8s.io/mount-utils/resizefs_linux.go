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
	"strconv"
	"strings"

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

	switch format {
	case "ext3", "ext4", "xfs":
		// For ext3/ext4/xfs, recommendation received from linux filesystem folks is to let
		// resize2fs/xfs_growfs do the check for us. So we will not do any check here.
		return true, nil
	case "btrfs":
		deviceSize, err := resizefs.getDeviceSize(devicePath)
		if err != nil {
			return false, err
		}
		blockSize, fsSize, err := resizefs.getBtrfsSize(devicePath)
		klog.V(5).Infof("Btrfs size: filesystem size=%d, block size=%d, err=%v", fsSize, blockSize, err)
		if err != nil {
			return false, err
		}
		if deviceSize <= fsSize+blockSize {
			return false, nil
		}
		return true, nil
	default:
		return false, fmt.Errorf("could not parse fs info of given filesystem format: %s. Supported fs types are: xfs, ext3, ext4", format)
	}
}

func (resizefs *ResizeFs) getDeviceSize(devicePath string) (uint64, error) {
	output, err := resizefs.exec.Command(blockDev, "--getsize64", devicePath).CombinedOutput()
	outStr := strings.TrimSpace(string(output))
	if err != nil {
		return 0, fmt.Errorf("failed to read size of device %s: %s: %s", devicePath, err, outStr)
	}
	size, err := strconv.ParseUint(outStr, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("failed to parse size of device %s %s: %s", devicePath, outStr, err)
	}
	return size, nil
}

func (resizefs *ResizeFs) getBtrfsSize(devicePath string) (uint64, uint64, error) {
	output, err := resizefs.exec.Command("btrfs", "inspect-internal", "dump-super", "-f", devicePath).CombinedOutput()
	if err != nil {
		return 0, 0, fmt.Errorf("failed to read size of filesystem on %s: %s: %s", devicePath, err, string(output))
	}

	blockSize, totalBytes, _ := resizefs.parseBtrfsInfoOutput(string(output), "sectorsize", "total_bytes")

	if blockSize == 0 {
		return 0, 0, fmt.Errorf("could not find block size of device %s", devicePath)
	}
	if totalBytes == 0 {
		return 0, 0, fmt.Errorf("could not find total size of device %s", devicePath)
	}
	return blockSize, totalBytes, nil
}

func (resizefs *ResizeFs) parseBtrfsInfoOutput(cmdOutput string, blockSizeKey string, totalBytesKey string) (uint64, uint64, error) {
	lines := strings.Split(cmdOutput, "\n")
	var blockSize, blockCount uint64
	var err error

	for _, line := range lines {
		tokens := strings.Fields(line)
		if len(tokens) != 2 {
			continue
		}
		key, value := strings.ToLower(strings.TrimSpace(tokens[0])), strings.ToLower(strings.TrimSpace(tokens[1]))

		if key == blockSizeKey {
			blockSize, err = strconv.ParseUint(value, 10, 64)
			if err != nil {
				return 0, 0, fmt.Errorf("failed to parse block size %s: %s", value, err)
			}
		}
		if key == totalBytesKey {
			blockCount, err = strconv.ParseUint(value, 10, 64)
			if err != nil {
				return 0, 0, fmt.Errorf("failed to parse total size %s: %s", value, err)
			}
		}
	}
	return blockSize, blockCount, err
}

func (resizefs *ResizeFs) getDeviceRO(devicePath string) (bool, error) {
	output, err := resizefs.exec.Command(blockDev, "--getro", devicePath).CombinedOutput()
	outStr := strings.TrimSpace(string(output))
	if err != nil {
		return false, fmt.Errorf("failed to get readonly bit from device %s: %w: %s", devicePath, err, outStr)
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
