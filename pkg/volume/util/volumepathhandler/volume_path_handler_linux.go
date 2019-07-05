// +build linux

/*
Copyright 2018 The Kubernetes Authors.

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

package volumepathhandler

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"k8s.io/klog"
)

// AttachFileDevice takes a path to a regular file and makes it available as an
// attached block device.
func (v VolumePathHandler) AttachFileDevice(path string) (string, error) {
	blockDevicePath, err := v.GetLoopDevice(path)
	if err != nil && err.Error() != ErrDeviceNotFound {
		return "", err
	}

	// If no existing loop device for the path, create one
	if blockDevicePath == "" {
		klog.V(4).Infof("Creating device for path: %s", path)
		blockDevicePath, err = makeLoopDevice(path)
		if err != nil {
			return "", err
		}
	}
	return blockDevicePath, nil
}

// GetLoopDevice returns the full path to the loop device associated with the given path.
func (v VolumePathHandler) GetLoopDevice(path string) (string, error) {
	_, err := os.Stat(path)
	if os.IsNotExist(err) {
		return "", errors.New(ErrDeviceNotFound)
	}
	if err != nil {
		return "", fmt.Errorf("not attachable: %v", err)
	}

	args := []string{"-j", path}
	cmd := exec.Command(losetupPath, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		klog.V(2).Infof("Failed device discover command for path %s: %v %s", path, err, out)
		return "", err
	}
	return parseLosetupOutputForDevice(out)
}

func makeLoopDevice(path string) (string, error) {
	args := []string{"-f", "--show", path}
	cmd := exec.Command(losetupPath, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		klog.V(2).Infof("Failed device create command for path: %s %v %s ", path, err, out)
		return "", err
	}
	return parseLosetupOutputForDevice(out)
}

// RemoveLoopDevice removes specified loopback device
func (v VolumePathHandler) RemoveLoopDevice(device string) error {
	args := []string{"-d", device}
	cmd := exec.Command(losetupPath, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		if _, err := os.Stat(device); os.IsNotExist(err) {
			return nil
		}
		klog.V(2).Infof("Failed to remove loopback device: %s: %v %s", device, err, out)
		return err
	}
	return nil
}

func parseLosetupOutputForDevice(output []byte) (string, error) {
	if len(output) == 0 {
		return "", errors.New(ErrDeviceNotFound)
	}

	// losetup returns device in the format:
	// /dev/loop1: [0073]:148662 (/dev/sda)
	device := strings.TrimSpace(strings.SplitN(string(output), ":", 2)[0])
	if len(device) == 0 {
		return "", errors.New(ErrDeviceNotFound)
	}
	return device, nil
}
