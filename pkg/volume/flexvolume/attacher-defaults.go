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
	"fmt"
	"path"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

type attacherDefaults flexVolumeAttacher

// Attach is part of the volume.Attacher interface
func (a *attacherDefaults) Attach(spec *volume.Spec, hostName types.NodeName) (string, error) {
	glog.Warning(logPrefix(a.plugin), "using default Attach for volume ", spec.Name, ", host ", hostName)
	return "", nil
}

// WaitForAttach is part of the volume.Attacher interface
func (a *attacherDefaults) WaitForAttach(spec *volume.Spec, devicePath string, timeout time.Duration) (string, error) {
	glog.Warning(logPrefix(a.plugin), "using default WaitForAttach for volume ", spec.Name, ", device ", devicePath)
	return devicePath, nil
}

// GetDeviceMountPath is part of the volume.Attacher interface
func (a *attacherDefaults) GetDeviceMountPath(spec *volume.Spec, mountsDir string) (string, error) {
	glog.Warning(logPrefix(a.plugin), "using default GetDeviceMountPath for volume ", spec.Name, ", mountsDir ", mountsDir)
	volumeName, err := a.plugin.GetVolumeName(spec)
	if err != nil {
		return "", fmt.Errorf("GetVolumeName failed from GetDeviceMountPath: %s", err)
	}

	return path.Join(mountsDir, volumeName), nil
}

// MountDevice is part of the volume.Attacher interface
func (a *attacherDefaults) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string, mounter mount.Interface) error {
	glog.Warning(logPrefix(a.plugin), "using default MountDevice for volume ", spec.Name, ", device ", devicePath, ", deviceMountPath ", deviceMountPath)
	volSource, readOnly := getVolumeSource(spec)

	options := make([]string, 0)

	if readOnly {
		options = append(options, "ro")
	} else {
		options = append(options, "rw")
	}

	diskMounter := &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()}

	return diskMounter.FormatAndMount(devicePath, deviceMountPath, volSource.FSType, options)
}
