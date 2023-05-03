//go:build linux
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
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"golang.org/x/sys/unix"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
)

// AttachFileDevice takes a path to a regular file and makes it available as an
// attached block device.
func (v VolumePathHandler) AttachFileDevice(path string) (string, error) {
	blockDevicePath, err := v.GetLoopDevice(path)
	if err != nil && err.Error() != ErrDeviceNotFound {
		return "", fmt.Errorf("GetLoopDevice failed for path %s: %v", path, err)
	}

	// If no existing loop device for the path, create one
	if blockDevicePath == "" {
		klog.V(4).Infof("Creating device for path: %s", path)
		blockDevicePath, err = makeLoopDevice(path)
		if err != nil {
			return "", fmt.Errorf("makeLoopDevice failed for path %s: %v", path, err)
		}
	}
	return blockDevicePath, nil
}

// DetachFileDevice takes a path to the attached block device and
// detach it from block device.
func (v VolumePathHandler) DetachFileDevice(path string) error {
	loopPath, err := v.GetLoopDevice(path)
	if err != nil {
		if err.Error() == ErrDeviceNotFound {
			klog.Warningf("couldn't find loopback device which takes file descriptor lock. Skip detaching device. device path: %q", path)
		} else {
			return fmt.Errorf("GetLoopDevice failed for path %s: %v", path, err)
		}
	} else {
		if len(loopPath) != 0 {
			err = removeLoopDevice(loopPath)
			if err != nil {
				return fmt.Errorf("removeLoopDevice failed for path %s: %v", path, err)
			}
		}
	}
	return nil
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

	return getLoopDeviceFromSysfs(path)
}

func makeLoopDevice(path string) (string, error) {
	args := []string{"-f", path}
	cmd := exec.Command(losetupPath, args...)

	out, err := cmd.CombinedOutput()
	if err != nil {
		klog.V(2).Infof("Failed device create command for path: %s %v %s", path, err, out)
		return "", fmt.Errorf("losetup %s failed: %v", strings.Join(args, " "), err)
	}

	return getLoopDeviceFromSysfs(path)
}

// removeLoopDevice removes specified loopback device
func removeLoopDevice(device string) error {
	args := []string{"-d", device}
	cmd := exec.Command(losetupPath, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		if _, err := os.Stat(device); os.IsNotExist(err) {
			return nil
		}
		klog.V(2).Infof("Failed to remove loopback device: %s: %v %s", device, err, out)
		return fmt.Errorf("losetup -d %s failed: %v", device, err)
	}
	return nil
}

// getLoopDeviceFromSysfs finds the backing file for a loop
// device from sysfs via "/sys/block/loop*/loop/backing_file".
func getLoopDeviceFromSysfs(path string) (string, error) {
	// If the file is a symlink.
	realPath, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", fmt.Errorf("failed to evaluate path %s: %s", path, err)
	}

	devices, err := filepath.Glob("/sys/block/loop*")
	if err != nil {
		return "", fmt.Errorf("failed to list loop devices in sysfs: %s", err)
	}

	for _, device := range devices {
		backingFile := fmt.Sprintf("%s/loop/backing_file", device)

		// The contents of this file is the absolute path of "path".
		data, err := ioutil.ReadFile(backingFile)
		if err != nil {
			continue
		}

		// Return the first match.
		backingFilePath := cleanBackingFilePath(string(data))
		if backingFilePath == path || backingFilePath == realPath {
			return fmt.Sprintf("/dev/%s", filepath.Base(device)), nil
		}
	}

	return "", errors.New(ErrDeviceNotFound)
}

// cleanPath remove any trailing substrings that are not part of the backing file path.
func cleanBackingFilePath(path string) string {
	// If the block device was deleted, the path will contain a "(deleted)" suffix
	path = strings.TrimSpace(path)
	path = strings.TrimSuffix(path, "(deleted)")
	return strings.TrimSpace(path)
}

// FindGlobalMapPathUUIDFromPod finds {pod uuid} bind mount under globalMapPath
// corresponding to map path symlink, and then return global map path with pod uuid.
// (See pkg/volume/volume.go for details on a global map path and a pod device map path.)
// ex. mapPath symlink: pods/{podUid}}/{DefaultKubeletVolumeDevicesDirName}/{escapeQualifiedPluginName}/{volumeName} -> /dev/sdX
//
//	globalMapPath/{pod uuid} bind mount: plugins/kubernetes.io/{PluginName}/{DefaultKubeletVolumeDevicesDirName}/{volumePluginDependentPath}/{pod uuid} -> /dev/sdX
func (v VolumePathHandler) FindGlobalMapPathUUIDFromPod(pluginDir, mapPath string, podUID types.UID) (string, error) {
	var globalMapPathUUID string
	// Find symbolic link named pod uuid under plugin dir
	err := filepath.Walk(pluginDir, func(path string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if (fi.Mode()&os.ModeDevice == os.ModeDevice) && (fi.Name() == string(podUID)) {
			klog.V(5).Infof("FindGlobalMapPathFromPod: path %s, mapPath %s", path, mapPath)
			if res, err := compareBindMountAndSymlinks(path, mapPath); err == nil && res {
				globalMapPathUUID = path
			}
		}
		return nil
	})
	if err != nil {
		return "", fmt.Errorf("FindGlobalMapPathUUIDFromPod failed: %v", err)
	}
	klog.V(5).Infof("FindGlobalMapPathFromPod: globalMapPathUUID %s", globalMapPathUUID)
	// Return path contains global map path + {pod uuid}
	return globalMapPathUUID, nil
}

// compareBindMountAndSymlinks returns if global path (bind mount) and
// pod path (symlink) are pointing to the same device.
// If there is an error in checking it returns error.
func compareBindMountAndSymlinks(global, pod string) (bool, error) {
	// To check if bind mount and symlink are pointing to the same device,
	// we need to check if they are pointing to the devices that have same major/minor number.

	// Get the major/minor number for global path
	devNumGlobal, err := getDeviceMajorMinor(global)
	if err != nil {
		return false, fmt.Errorf("getDeviceMajorMinor failed for path %s: %v", global, err)
	}

	// Get the symlinked device from the pod path
	devPod, err := os.Readlink(pod)
	if err != nil {
		return false, fmt.Errorf("failed to readlink path %s: %v", pod, err)
	}
	// Get the major/minor number for the symlinked device from the pod path
	devNumPod, err := getDeviceMajorMinor(devPod)
	if err != nil {
		return false, fmt.Errorf("getDeviceMajorMinor failed for path %s: %v", devPod, err)
	}
	klog.V(5).Infof("CompareBindMountAndSymlinks: devNumGlobal %s, devNumPod %s", devNumGlobal, devNumPod)

	// Check if the major/minor number are the same
	if devNumGlobal == devNumPod {
		return true, nil
	}
	return false, nil
}

// getDeviceMajorMinor returns major/minor number for the path with below format:
// major:minor (in hex)
// ex)
//
//	fc:10
func getDeviceMajorMinor(path string) (string, error) {
	var stat unix.Stat_t

	if err := unix.Stat(path, &stat); err != nil {
		return "", fmt.Errorf("failed to stat path %s: %v", path, err)
	}

	devNumber := uint64(stat.Rdev)
	major := unix.Major(devNumber)
	minor := unix.Minor(devNumber)

	return fmt.Sprintf("%x:%x", major, minor), nil
}
