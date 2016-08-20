/*
Copyright 2016 The Kubernetes Authors.

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
	"path"
	"time"

	"k8s.io/kubernetes/pkg/volume"
)

type flexVolumeAttacher struct {
	plugin *flexVolumePlugin
}

var _ volume.Attacher = &flexVolumeAttacher{}

// Attach is part of the volume.Attacher interface
func (a *flexVolumeAttacher) Attach(spec *volume.Spec, hostName string) (string, error) {
	call := a.plugin.NewDriverCall(attachCmd)
	call.AppendSpec(spec, a.plugin.host, nil)
	call.Append(hostName)

	status, err := call.Run()
	if isCmdNotSupportedErr(err) {
		return (*attacherDefaults)(a).Attach(spec, hostName)
	} else if err != nil {
		return "", err
	}
	return status.Device, err
}

// WaitForAttach is part of the volume.Attacher interface
func (a *flexVolumeAttacher) WaitForAttach(spec *volume.Spec, devicePath string, timeout time.Duration) (string, error) {
	call := a.plugin.NewDriverCallWithTimeout(waitForAttachCmd, timeout)
	call.AppendSpec(spec, a.plugin.host, nil)
	call.Append(devicePath)

	status, err := call.Run()
	if isCmdNotSupportedErr(err) {
		return (*attacherDefaults)(a).WaitForAttach(spec, devicePath, timeout)
	} else if err != nil {
		return "", err
	}
	return status.Device, nil
}

// GetDeviceMountPath is part of the volume.Attacher interface
func (a *flexVolumeAttacher) GetDeviceMountPath(spec *volume.Spec) (string, error) {
	mountsDir := path.Join(a.plugin.host.GetPluginDir(flexVolumePluginName), a.plugin.driverName, "mounts")

	call := a.plugin.NewDriverCall(getDeviceMountPathCmd)
	call.AppendSpec(spec, a.plugin.host, map[string]string{
		optionMountsDir: mountsDir,
	})

	status, err := call.Run()
	if isCmdNotSupportedErr(err) {
		return (*attacherDefaults)(a).GetDeviceMountPath(spec, mountsDir)
	} else if err != nil {
		return "", err
	}
	return status.Path, nil
}

// MountDevice is part of the volume.Attacher interface
func (a *flexVolumeAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	// Mount only once.
	alreadyMounted, err := prepareForMount(a.plugin.host.GetMounter(), deviceMountPath)
	if err != nil {
		return err
	}
	if alreadyMounted {
		return nil
	}

	call := a.plugin.NewDriverCall(mountDeviceCmd)
	call.AppendSpec(spec, a.plugin.host, nil)
	call.Append(devicePath)
	call.Append(deviceMountPath)

	_, err = call.Run()
	if isCmdNotSupportedErr(err) {
		return (*attacherDefaults)(a).MountDevice(spec, devicePath, deviceMountPath, a.plugin.host.GetMounter())
	}
	return err
}
