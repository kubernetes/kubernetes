/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

// Package attachdetach contains consts and helper methods used by various
// attach/detach components in controller and kubelet
package attachdetach

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	// ControllerManagedAnnotation is the key of the annotation on Node objects
	// that indicates attach/detach operations for the node should be managed
	// by the attach/detach controller
	ControllerManagedAnnotation string = "volumes.kubernetes.io/controller-managed-attach-detach"
)

// GetUniqueDeviceName returns a unique name representing the device with the
// spcified deviceName of the pluginName volume type.
// The returned name can be used to uniquely reference the device. For example,
// to prevent operations (attach/detach) from being triggered on the same volume
func GetUniqueDeviceName(
	pluginName, deviceName string) api.UniqueDeviceName {
	return api.UniqueDeviceName(fmt.Sprintf("%s/%s", pluginName, deviceName))
}

// GetUniqueDeviceNameFromSpec uses the given AttachableVolumePlugin to
// generate a unique name representing the device defined in the specified
// volume spec.
// This returned name can be used to uniquely reference the device. For example,
// to prevent operations (attach/detach) from being triggered on the same volume.
// If the given plugin does not support the volume spec, this returns an error.
func GetUniqueDeviceNameFromSpec(
	attachableVolumePlugin volume.AttachableVolumePlugin,
	volumeSpec *volume.Spec) (api.UniqueDeviceName, error) {
	if attachableVolumePlugin == nil {
		return "", fmt.Errorf(
			"attachablePlugin should not be nil. volumeSpec.Name=%q",
			volumeSpec.Name())
	}

	deviceName, err := attachableVolumePlugin.GetDeviceName(volumeSpec)
	if err != nil || deviceName == "" {
		return "", fmt.Errorf(
			"failed to GetDeviceName from AttachablePlugin for volumeSpec %q err=%v",
			volumeSpec.Name(),
			err)
	}

	return GetUniqueDeviceName(
			attachableVolumePlugin.Name(),
			deviceName),
		nil
}
