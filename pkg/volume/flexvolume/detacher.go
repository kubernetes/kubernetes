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
	"time"

	"k8s.io/kubernetes/pkg/volume"
)

type flexVolumeDetacher struct {
	plugin *flexVolumePlugin
}

var _ volume.Detacher = &flexVolumeDetacher{}

// Detach is part of the volume.Detacher interface.
func (d *flexVolumeDetacher) Detach(deviceName, hostName string) error {
	call := d.plugin.NewDriverCall(detachCmd)
	call.Append(deviceName)
	call.Append(hostName)

	_, err := call.Run()
	if isCmdNotSupportedErr(err) {
		return (*detacherDefaults)(d).Detach(deviceName, hostName)
	}
	return err
}

// WaitForDetach is part of the volume.Detacher interface.
func (d *flexVolumeDetacher) WaitForDetach(devicePath string, timeout time.Duration) error {
	call := d.plugin.NewDriverCallWithTimeout(waitForDetachCmd, timeout)
	call.Append(devicePath)

	_, err := call.Run()
	if isCmdNotSupportedErr(err) {
		return (*detacherDefaults)(d).WaitForDetach(devicePath, timeout)
	}
	return err
}

// UnmountDevice is part of the volume.Detacher interface.
func (d *flexVolumeDetacher) UnmountDevice(deviceMountPath string) error {
	call := d.plugin.NewDriverCall(unmountDeviceCmd)
	call.Append(deviceMountPath)

	_, err := call.Run()
	if isCmdNotSupportedErr(err) {
		err = (*detacherDefaults)(d).UnmountDevice(deviceMountPath)
	}
	if err != nil {
		return err
	}
	return removeMountPath(d.plugin.host.GetMounter(), deviceMountPath)
}
