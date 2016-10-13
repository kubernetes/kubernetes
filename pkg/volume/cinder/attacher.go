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

package cinder

import (
	"os"
	"time"
	
  "k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

type cinderDiskAttacher struct {
	*cinderVolume
}

func (attacher *cinderDiskAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	manager, err := getManager(attacher.plugin, spec, nil)
	if err != nil {
		return "", err
	}
	return manager.Attach(spec, nodeName)
}

func (attacher *cinderDiskAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	specsByManager := make(map[string][]*volume.Spec)
	managersByName := make(map[string]cinderManager)

	for _, spec := range specs {
		manager, err := getManager(attacher.plugin, spec, nil)
		if err != nil {
			return nil, err
		}

		managerName := manager.GetName()
		managersByName[managerName] = manager

		if _, ok := specsByManager[managerName]; !ok {
			specsByManager[managerName] = []*volume.Spec{spec}
		} else {
			specsByManager[managerName] = append(specsByManager[managerName], spec)
		}
	}

	results := make(map[*volume.Spec]bool)
	for managerName, specs := range specsByManager {
		managerResults, err := managersByName[managerName].VolumesAreAttached(specs, nodeName)
		if err != nil {
			return nil, err
		}
		for spec, attached := range managerResults {
			results[spec] = attached
		}
	}

	return results, nil
}

func (attacher *cinderDiskAttacher) WaitForAttach(spec *volume.Spec, devicePath string, timeout time.Duration) (string, error) {
	manager, err := getManager(attacher.plugin, spec, nil)
	if err != nil {
		return "", err
	}

	return manager.WaitForAttach(spec, devicePath, timeout)
}

func (attacher *cinderDiskAttacher) GetDeviceMountPath(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return makeGlobalPDName(attacher.plugin.host, volumeSource.VolumeID), nil
}

// FIXME: this method can be further pruned.
func (attacher *cinderDiskAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	mounter := attacher.plugin.host.GetMounter()
	notMnt, err := mounter.IsLikelyNotMountPoint(deviceMountPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err = os.MkdirAll(deviceMountPath, 0750); err != nil {
				return err
			}
			notMnt = true
		} else {
			return err
		}
	}

	volumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return err
	}

	options := []string{}
	if readOnly {
		options = append(options, "ro")
	}
	if notMnt {
		diskMounter := &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()}
		err = diskMounter.FormatAndMount(devicePath, deviceMountPath, volumeSource.FSType, options)
		if err != nil {
			os.Remove(deviceMountPath)
			return err
		}
	}
	return nil
}

type cinderDiskDetacher struct {
	*cinderVolume
}

func (detacher *cinderDiskDetacher) Detach(spec *volume.Spec, deviceMountPath string, nodeName types.NodeName) error {
	manager, err := getManager(detacher.plugin, spec, nil)
	if err != nil {
		return err
	}
	return manager.Detach(spec, deviceMountPath, nodeName)
}

func (detacher *cinderDiskDetacher) UnmountDevice(spec *volume.Spec, deviceMountPath string) error {
	manager, err := getManager(detacher.plugin, spec, nil)
	if err != nil {
		return err
	}
	return manager.UnmountDevice(spec, deviceMountPath, detacher.mounter)
}
