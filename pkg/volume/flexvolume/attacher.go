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
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
)

type flexVolumeAttacher struct {
	plugin *flexVolumeAttachablePlugin
}

var _ volume.Attacher = &flexVolumeAttacher{}

// Attach is part of the volume.Attacher interface
func (a *flexVolumeAttacher) Attach(spec *volume.Spec, hostName types.NodeName) (string, error) {

	call := a.plugin.NewDriverCall(attachCmd)
	call.AppendSpec(spec, a.plugin.host, nil)
	call.Append(string(hostName))

	status, err := call.Run()
	if isCmdNotSupportedErr(err) {
		return (*attacherDefaults)(a).Attach(spec, hostName)
	} else if err != nil {
		return "", err
	}
	return status.DevicePath, err
}

// WaitForAttach is part of the volume.Attacher interface
func (a *flexVolumeAttacher) WaitForAttach(spec *volume.Spec, devicePath string, _ *v1.Pod, timeout time.Duration) (string, error) {
	call := a.plugin.NewDriverCallWithTimeout(waitForAttachCmd, timeout)
	call.Append(devicePath)
	call.AppendSpec(spec, a.plugin.host, nil)

	status, err := call.Run()
	if isCmdNotSupportedErr(err) {
		return (*attacherDefaults)(a).WaitForAttach(spec, devicePath, timeout)
	} else if err != nil {
		return "", err
	}
	return status.DevicePath, nil
}

// GetDeviceMountPath is part of the volume.Attacher interface
func (a *flexVolumeAttacher) GetDeviceMountPath(spec *volume.Spec) (string, error) {
	return a.plugin.getDeviceMountPath(spec)
}

// MountDevice is part of the volume.Attacher interface
func (a *flexVolumeAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	// Mount only once.
	alreadyMounted, err := prepareForMount(a.plugin.host.GetMounter(a.plugin.GetPluginName()), deviceMountPath)
	if err != nil {
		return err
	}
	if alreadyMounted {
		return nil
	}

	call := a.plugin.NewDriverCall(mountDeviceCmd)
	call.Append(deviceMountPath)
	call.Append(devicePath)
	call.AppendSpec(spec, a.plugin.host, nil)

	_, err = call.Run()
	if isCmdNotSupportedErr(err) {
		// Devicepath is empty if the plugin does not support attach calls. Ignore mountDevice calls if the
		// plugin does not implement attach interface.
		if devicePath != "" {
			return (*attacherDefaults)(a).MountDevice(spec, devicePath, deviceMountPath, a.plugin.host.GetMounter(a.plugin.GetPluginName()))
		} else {
			return nil
		}
	}
	return err
}

func (a *flexVolumeAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	volumesAttachedCheck := make(map[*volume.Spec]bool)
	for _, spec := range specs {
		volumesAttachedCheck[spec] = true

		call := a.plugin.NewDriverCall(isAttached)
		call.AppendSpec(spec, a.plugin.host, nil)
		call.Append(string(nodeName))

		status, err := call.Run()
		if isCmdNotSupportedErr(err) {
			return nil, nil
		} else if err == nil {
			if !status.Attached {
				volumesAttachedCheck[spec] = false
				glog.V(2).Infof("VolumesAreAttached: check volume (%q) is no longer attached", spec.Name())
			}
		}
	}
	return volumesAttachedCheck, nil
}
