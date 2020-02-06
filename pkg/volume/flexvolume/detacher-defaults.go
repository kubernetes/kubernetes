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

	"k8s.io/klog"
	"k8s.io/utils/mount"

	"k8s.io/apimachinery/pkg/types"
)

type detacherDefaults flexVolumeDetacher

// Detach is part of the volume.Detacher interface.
func (d *detacherDefaults) Detach(volumeName string, hostName types.NodeName) error {
	klog.Warning(logPrefix(d.plugin.flexVolumePlugin), "using default Detach for volume ", volumeName, ", host ", hostName)
	return nil
}

// WaitForDetach is part of the volume.Detacher interface.
func (d *detacherDefaults) WaitForDetach(devicePath string, timeout time.Duration) error {
	klog.Warning(logPrefix(d.plugin.flexVolumePlugin), "using default WaitForDetach for device ", devicePath)
	return nil
}

// UnmountDevice is part of the volume.Detacher interface.
func (d *detacherDefaults) UnmountDevice(deviceMountPath string) error {
	klog.Warning(logPrefix(d.plugin.flexVolumePlugin), "using default UnmountDevice for device mount path ", deviceMountPath)
	return mount.CleanupMountPoint(deviceMountPath, d.plugin.host.GetMounter(d.plugin.GetPluginName()), false)
}
