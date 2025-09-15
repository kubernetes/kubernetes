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

	"k8s.io/klog/v2"

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
	unmountDeviceCmd = "unmountdevice"

	mountCmd   = "mount"
	unmountCmd = "unmount"

	expandVolumeCmd = "expandvolume"
	expandFSCmd     = "expandfs"

	// Option keys
	optionFSType         = "kubernetes.io/fsType"
	optionReadWrite      = "kubernetes.io/readwrite"
	optionKeySecret      = "kubernetes.io/secret"
	optionFSGroup        = "kubernetes.io/mounterArgs.FsGroup"
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
	errTimeout = fmt.Errorf("timeout")
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

// Append appends arg into driver call argument list
func (dc *DriverCall) Append(arg string) {
	dc.args = append(dc.args, arg)
}

// AppendSpec appends volume spec to driver call argument list
func (dc *DriverCall) AppendSpec(spec *volume.Spec, host volume.VolumeHost, extraOptions map[string]string) error {
	optionsForDriver, err := NewOptionsForDriver(spec, host, extraOptions)
	if err != nil {
		return err
	}

	jsonBytes, err := json.Marshal(optionsForDriver)
	if err != nil {
		return fmt.Errorf("failed to marshal spec, error: %s", err.Error())
	}

	dc.Append(string(jsonBytes))
	return nil
}

// Run executes the driver call
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
			return nil, errTimeout
		}
		_, err := handleCmdResponse(dc.Command, output)
		if err == nil {
			klog.Errorf("FlexVolume: driver bug: %s: exec error (%s) but no error in response.", execPath, execErr)
			return nil, execErr
		}
		if isCmdNotSupportedErr(err) {
			dc.plugin.unsupported(dc.Command)
		} else {
			klog.Warningf("FlexVolume: driver call failed: executable: %s, args: %s, error: %s, output: %q", execPath, dc.args, execErr.Error(), output)
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

// NewOptionsForDriver create driver options given volume spec
func NewOptionsForDriver(spec *volume.Spec, host volume.VolumeHost, extraOptions map[string]string) (OptionsForDriver, error) {

	volSourceFSType, err := getFSType(spec)
	if err != nil {
		return nil, err
	}

	readOnly, err := getReadOnly(spec)
	if err != nil {
		return nil, err
	}

	volSourceOptions, err := getOptions(spec)
	if err != nil {
		return nil, err
	}

	options := map[string]string{}

	options[optionFSType] = volSourceFSType

	if readOnly {
		options[optionReadWrite] = "ro"
	} else {
		options[optionReadWrite] = "rw"
	}

	options[optionPVorVolumeName] = spec.Name()

	for key, value := range extraOptions {
		options[key] = value
	}

	for key, value := range volSourceOptions {
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
	// Returns the actual size of the volume after resizing is done, the size is in bytes.
	ActualVolumeSize int64 `json:"volumeNewSize,omitempty"`
}

// DriverCapabilities represents what driver can do
type DriverCapabilities struct {
	Attach           bool `json:"attach"`
	SELinuxRelabel   bool `json:"selinuxRelabel"`
	SupportsMetrics  bool `json:"supportsMetrics"`
	FSGroup          bool `json:"fsGroup"`
	RequiresFSResize bool `json:"requiresFSResize"`
}

func defaultCapabilities() *DriverCapabilities {
	return &DriverCapabilities{
		Attach:           true,
		SELinuxRelabel:   true,
		SupportsMetrics:  false,
		FSGroup:          true,
		RequiresFSResize: true,
	}
}

// isCmdNotSupportedErr checks if the error corresponds to command not supported by
// driver.
func isCmdNotSupportedErr(err error) bool {
	return err != nil && err.Error() == StatusNotSupported
}

// handleCmdResponse processes the command output and returns the appropriate
// error code or message.
func handleCmdResponse(cmd string, output []byte) (*DriverStatus, error) {
	status := DriverStatus{
		Capabilities: defaultCapabilities(),
	}
	if err := json.Unmarshal(output, &status); err != nil {
		klog.Errorf("Failed to unmarshal output for command: %s, output: %q, error: %s", cmd, string(output), err.Error())
		return nil, err
	} else if status.Status == StatusNotSupported {
		klog.V(5).Infof("%s command is not supported by the driver", cmd)
		return nil, errors.New(status.Status)
	} else if status.Status != StatusSuccess {
		errMsg := fmt.Sprintf("%s command failed, status: %s, reason: %s", cmd, status.Status, status.Message)
		klog.Error(errMsg)
		return nil, fmt.Errorf("%s", errMsg)
	}

	return &status, nil
}
