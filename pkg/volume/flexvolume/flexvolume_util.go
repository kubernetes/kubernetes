/*
Copyright 2015 The Kubernetes Authors.

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

package flexvolume

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/util/exec"
)

const (
	initCmd    = "init"
	attachCmd  = "attach"
	detachCmd  = "detach"
	mountCmd   = "mount"
	unmountCmd = "unmount"

	optionFSType    = "kubernetes.io/fsType"
	optionReadWrite = "kubernetes.io/readwrite"
	optionKeySecret = "kubernetes.io/secret"
)

const (
	// StatusSuccess represents the successful completion of command.
	StatusSuccess = "Success"
	// StatusFailed represents that the command failed.
	StatusFailure = "Failed"
	// StatusNotSupported represents that the command is not supported.
	StatusNotSupported = "Not supported"
)

// FlexVolumeDriverStatus represents the return value of the driver callout.
type FlexVolumeDriverStatus struct {
	// Status of the callout. One of "Success" or "Failure".
	Status string
	// Message is the reason for failure.
	Message string
	// Device assigned by the driver.
	Device string `json:"device"`
}

// flexVolumeUtil is the utility structure to setup and teardown devices from
// the host.
type flexVolumeUtil struct{}

// isCmdNotSupportedErr checks if the error corresponds to command not supported by
// driver.
func isCmdNotSupportedErr(err error) bool {
	if err.Error() == StatusNotSupported {
		return true
	}

	return false
}

// handleCmdResponse processes the command output and returns the appropriate
// error code or message.
func handleCmdResponse(cmd string, output []byte) (*FlexVolumeDriverStatus, error) {
	var status FlexVolumeDriverStatus
	if err := json.Unmarshal(output, &status); err != nil {
		glog.Errorf("Failed to unmarshal output for command: %s, output: %s, error: %s", cmd, output, err.Error())
		return nil, err
	} else if status.Status == StatusNotSupported {
		glog.V(5).Infof("%s command is not supported by the driver", cmd)
		return nil, errors.New(status.Status)
	} else if status.Status != StatusSuccess {
		errMsg := fmt.Sprintf("%s command failed, status: %s, reason: %s", cmd, status.Status, status.Message)
		glog.Errorf(errMsg)
		return nil, fmt.Errorf("%s", errMsg)
	}

	return &status, nil
}

// init initializes the plugin.
func (u *flexVolumeUtil) init(plugin *flexVolumePlugin) error {
	// call the init script
	output, err := exec.New().Command(plugin.getExecutable(), initCmd).CombinedOutput()
	if err != nil {
		glog.Errorf("Failed to init driver: %s, error: %s", plugin.driverName, err.Error())
		_, err := handleCmdResponse(initCmd, output)
		return err
	}

	glog.V(5).Infof("Successfully initialized driver %s", plugin.driverName)
	return nil
}

// Attach exposes a volume on the host.
func (u *flexVolumeUtil) attach(f *flexVolumeMounter) (string, error) {
	execPath := f.execPath

	var options string
	if f.options != nil {
		out, err := json.Marshal(f.options)
		if err != nil {
			glog.Errorf("Failed to marshal plugin options, error: %s", err.Error())
			return "", err
		}
		if len(out) != 0 {
			options = string(out)
		} else {
			options = ""
		}
	}

	cmd := f.runner.Command(execPath, attachCmd, options)
	output, err := cmd.CombinedOutput()
	if err != nil {
		glog.Errorf("Failed to attach volume %s, output: %s, error: %s", f.volName, output, err.Error())
		_, err := handleCmdResponse(attachCmd, output)
		return "", err
	}

	status, err := handleCmdResponse(attachCmd, output)
	if err != nil {
		return "", err
	}

	glog.Infof("Successfully attached volume %s on device: %s", f.volName, status.Device)

	return status.Device, nil
}

// Detach detaches a volume from the host.
func (u *flexVolumeUtil) detach(f *flexVolumeUnmounter, mntDevice string) error {
	execPath := f.execPath

	// Executable provider command.
	cmd := f.runner.Command(execPath, detachCmd, mntDevice)
	output, err := cmd.CombinedOutput()
	if err != nil {
		glog.Errorf("Failed to detach volume %s, output: %s, error: %s", f.volName, output, err.Error())
		_, err := handleCmdResponse(detachCmd, output)
		return err
	}

	_, err = handleCmdResponse(detachCmd, output)
	if err != nil {
		return err
	}

	glog.Infof("Successfully detached volume %s on device: %s", f.volName, mntDevice)
	return nil
}

// Mount mounts the volume on the host.
func (u *flexVolumeUtil) mount(f *flexVolumeMounter, mntDevice, dir string) error {
	execPath := f.execPath

	var options string
	if f.options != nil {
		out, err := json.Marshal(f.options)
		if err != nil {
			glog.Errorf("Failed to marshal plugin options, error: %s", err.Error())
			return err
		}
		if len(out) != 0 {
			options = string(out)
		} else {
			options = ""
		}
	}

	// Executable provider command.
	cmd := f.runner.Command(execPath, mountCmd, dir, mntDevice, options)
	output, err := cmd.CombinedOutput()
	if err != nil {
		glog.Errorf("Failed to mount volume %s, output: %s, error: %s", f.volName, output, err.Error())
		_, err := handleCmdResponse(mountCmd, output)
		return err
	}

	_, err = handleCmdResponse(mountCmd, output)
	if err != nil {
		return err
	}

	glog.Infof("Successfully mounted volume %s on dir: %s", f.volName, dir)
	return nil
}

// Unmount unmounts the volume on the host.
func (u *flexVolumeUtil) unmount(f *flexVolumeUnmounter, dir string) error {
	execPath := f.execPath

	// Executable provider command.
	cmd := f.runner.Command(execPath, unmountCmd, dir)
	output, err := cmd.CombinedOutput()
	if err != nil {
		glog.Errorf("Failed to unmount volume %s, output: %s, error: %s", f.volName, output, err.Error())
		_, err := handleCmdResponse(unmountCmd, output)
		return err
	}

	_, err = handleCmdResponse(unmountCmd, output)
	if err != nil {
		return err
	}

	glog.Infof("Successfully unmounted volume %s on dir: %s", f.volName, dir)
	return nil
}
