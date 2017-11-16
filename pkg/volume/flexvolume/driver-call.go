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

package flexvolume

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/volume"
)

const (
	// Driver calls
	initCmd          = "init"
	getVolumeNameCmd = "getvolumename"

	isAttached = "isattached"

	attachCmd        = "attach"
	waitForAttachCmd = "waitforattach"
	mountDeviceCmd   = "mountdevice"

	detachCmd        = "detach"
	waitForDetachCmd = "waitfordetach"
	unmountDeviceCmd = "unmountdevice"

	mountCmd   = "mount"
	unmountCmd = "unmount"

	// Option keys
	optionFSType         = "kubernetes.io/fsType"
	optionReadWrite      = "kubernetes.io/readwrite"
	optionKeySecret      = "kubernetes.io/secret"
	optionFSGroup        = "kubernetes.io/fsGroup"
	optionMountsDir      = "kubernetes.io/mountsDir"
	optionPVorVolumeName = "kubernetes.io/pvOrVolumeName"

	optionKeyPodName      = "kubernetes.io/pod.name"
	optionKeyPodNamespace = "kubernetes.io/pod.namespace"
	optionKeyPodUID       = "kubernetes.io/pod.uid"

	optionKeyServiceAccountName = "kubernetes.io/serviceAccount.name"
)

const (
	// StatusSuccess represents the successful completion of command.
	StatusSuccess = "Success"
	// StatusNotSupported represents that the command is not supported.
	StatusNotSupported = "Not supported"
)

var (
	TimeoutError = fmt.Errorf("Timeout")
)

// DriverCall implements the basic contract between FlexVolume and its driver.
// The caller is responsible for providing the required args.
type DriverCall struct {
	Command string
	Timeout time.Duration
	plugin  *flexVolumePlugin
	args    []string
}

func (plugin *flexVolumePlugin) NewDriverCall(command string) *DriverCall {
	return plugin.NewDriverCallWithTimeout(command, 0)
}

func (plugin *flexVolumePlugin) NewDriverCallWithTimeout(command string, timeout time.Duration) *DriverCall {
	return &DriverCall{
		Command: command,
		Timeout: timeout,
		plugin:  plugin,
		args:    []string{command},
	}
}

func (dc *DriverCall) Append(arg string) {
	dc.args = append(dc.args, arg)
}

func (dc *DriverCall) AppendSpec(spec *volume.Spec, host volume.VolumeHost, extraOptions map[string]string) error {
	optionsForDriver, err := NewOptionsForDriver(spec, host, extraOptions)
	if err != nil {
		return err
	}

	jsonBytes, err := json.Marshal(optionsForDriver)
	if err != nil {
		return fmt.Errorf("Failed to marshal spec, error: %s", err.Error())
	}

	dc.Append(string(jsonBytes))
	return nil
}

func (dc *DriverCall) Run() (*DriverStatus, error) {
	if dc.plugin.isUnsupported(dc.Command) {
		return nil, errors.New(StatusNotSupported)
	}
	execPath := dc.plugin.getExecutable()

	cmd := dc.plugin.runner.Command(execPath, dc.args...)

	timeout := false
	if dc.Timeout > 0 {
		timer := time.AfterFunc(dc.Timeout, func() {
			timeout = true
			cmd.Stop()
		})
		defer timer.Stop()
	}

	output, execErr := cmd.CombinedOutput()
	if execErr != nil {
		if timeout {
			return nil, TimeoutError
		}
		_, err := handleCmdResponse(dc.Command, output)
		if err == nil {
			glog.Errorf("FlexVolume: driver bug: %s: exec error (%s) but no error in response.", execPath, execErr)
			return nil, execErr
		}
		if isCmdNotSupportedErr(err) {
			dc.plugin.unsupported(dc.Command)
		} else {
			glog.Warningf("FlexVolume: driver call failed: executable: %s, args: %s, error: %s, output: %q", execPath, dc.args, execErr.Error(), output)
		}
		return nil, err
	}

	status, err := handleCmdResponse(dc.Command, output)
	if err != nil {
		if isCmdNotSupportedErr(err) {
			dc.plugin.unsupported(dc.Command)
		}
		return nil, err
	}

	return status, nil
}

// OptionsForDriver represents the spec given to the driver.
type OptionsForDriver map[string]string

func NewOptionsForDriver(spec *volume.Spec, host volume.VolumeHost, extraOptions map[string]string) (OptionsForDriver, error) {
	volSource, readOnly := getVolumeSource(spec)
	options := map[string]string{}

	options[optionFSType] = volSource.FSType

	if readOnly {
		options[optionReadWrite] = "ro"
	} else {
		options[optionReadWrite] = "rw"
	}

	options[optionPVorVolumeName] = spec.Name()

	for key, value := range extraOptions {
		options[key] = value
	}

	for key, value := range volSource.Options {
		options[key] = value
	}

	return OptionsForDriver(options), nil
}

// DriverStatus represents the return value of the driver callout.
type DriverStatus struct {
	// Status of the callout. One of "Success", "Failure" or "Not supported".
	Status string `json:"status"`
	// Reason for success/failure.
	Message string `json:"message,omitempty"`
	// Path to the device attached. This field is valid only for attach calls.
	// ie: /dev/sdx
	DevicePath string `json:"device,omitempty"`
	// Cluster wide unique name of the volume.
	VolumeName string `json:"volumeName,omitempty"`
	// Represents volume is attached on the node
	Attached bool `json:"attached,omitempty"`
	// Returns capabilities of the driver.
	// By default we assume all the capabilities are supported.
	// If the plugin does not support a capability, it can return false for that capability.
	Capabilities *DriverCapabilities `json:",omitempty"`
}

type DriverCapabilities struct {
	Attach         bool `json:"attach"`
	SELinuxRelabel bool `json:"selinuxRelabel"`
}

func defaultCapabilities() *DriverCapabilities {
	return &DriverCapabilities{
		Attach:         true,
		SELinuxRelabel: true,
	}
}

// isCmdNotSupportedErr checks if the error corresponds to command not supported by
// driver.
func isCmdNotSupportedErr(err error) bool {
	if err != nil && err.Error() == StatusNotSupported {
		return true
	}

	return false
}

// handleCmdResponse processes the command output and returns the appropriate
// error code or message.
func handleCmdResponse(cmd string, output []byte) (*DriverStatus, error) {
	status := DriverStatus{
		Capabilities: defaultCapabilities(),
	}
	if err := json.Unmarshal(output, &status); err != nil {
		glog.Errorf("Failed to unmarshal output for command: %s, output: %q, error: %s", cmd, string(output), err.Error())
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
