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
	"bufio"
	"errors"
	"fmt"
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

	args := []string{"-j", path}
	cmd := exec.Command(losetupPath, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		klog.V(2).Infof("Failed device discover command for path %s: %v %s", path, err, out)
		return "", fmt.Errorf("losetup -j %s failed: %v", path, err)
	}
	return parseLosetupOutputForDevice(out, path)
}

func makeLoopDevice(path string) (string, error) {
	args := []string{"-f", "--show", path}
	cmd := exec.Command(losetupPath, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		klog.V(2).Infof("Failed device create command for path: %s %v %s ", path, err, out)
		return "", fmt.Errorf("losetup -f --show %s failed: %v", path, err)
	}

	// losetup -f --show {path} returns device in the format:
	// /dev/loop1
	if len(out) == 0 {
		return "", errors.New(ErrDeviceNotFound)
	}

	return strings.TrimSpace(string(out)), nil
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

func parseLosetupOutputForDevice(output []byte, path string) (string, error) {
	if len(output) == 0 {
		return "", errors.New(ErrDeviceNotFound)
	}

	// losetup -j {path} returns device in the format:
	// /dev/loop1: [0073]:148662 ({path})
	// /dev/loop2: [0073]:148662 (/dev/sdX)
	//
	// losetup -j shows all the loop device for the same device that has the same
	// major/minor number, by resolving symlink and matching major/minor number.
	// Therefore, there will be other path than {path} in output, as shown in above output.
	s := string(output)
	// Find the line that exact matches to the path, or "({path})"
	var matched string
	scanner := bufio.NewScanner(strings.NewReader(s))
	for scanner.Scan() {
		if strings.HasSuffix(scanner.Text(), "("+path+")") {
			matched = scanner.Text()
			break
		}
	}
	if len(matched) == 0 {
		return "", errors.New(ErrDeviceNotFound)
	}
	s = matched

	// Get device name, or the 0th field of the output separated with ":".
	// We don't need 1st field or later to be splitted, so passing 2 to SplitN.
	device := strings.TrimSpace(strings.SplitN(s, ":", 2)[0])
	if len(device) == 0 {
		return "", errors.New(ErrDeviceNotFound)
	}
	return device, nil
}

// FindGlobalMapPathUUIDFromPod finds {pod uuid} bind mount under globalMapPath
// corresponding to map path symlink, and then return global map path with pod uuid.
// (See pkg/volume/volume.go for details on a global map path and a pod device map path.)
// ex. mapPath symlink: pods/{podUid}}/{DefaultKubeletVolumeDevicesDirName}/{escapeQualifiedPluginName}/{volumeName} -> /dev/sdX
//     globalMapPath/{pod uuid} bind mount: plugins/kubernetes.io/{PluginName}/{DefaultKubeletVolumeDevicesDirName}/{volumePluginDependentPath}/{pod uuid} -> /dev/sdX
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
//     fc:10
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
