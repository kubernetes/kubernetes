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

// Package volumehelper contains consts and helper methods used by various
// volume components (attach/detach controller, kubelet, etc.).
package volumehelper

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

// GetUniqueVolumeName returns a unique name representing the volume/plugin.
// Caller should ensure that volumeName is a name/ID uniquely identifying the
// actual backing device, directory, path, etc. for a particular volume.
// The returned name can be used to uniquely reference the volume, for example,
// to prevent operations (attach/detach or mount/unmount) from being triggered
// on the same volume.
func GetUniqueVolumeName(
	pluginName string, volumeName string) api.UniqueVolumeName {
	return api.UniqueVolumeName(fmt.Sprintf("%s/%s", pluginName, volumeName))
}

// GetUniqueVolumeNameFromSpec uses the given VolumePlugin to generate a unique
// name representing the volume defined in the specified volume spec.
// This returned name can be used to uniquely reference the actual backing
// device, directory, path, etc. referenced by the given volumeSpec.
// If the given plugin does not support the volume spec, this returns an error.
func GetUniqueVolumeNameFromSpec(
	volumePlugin volume.VolumePlugin,
	volumeSpec *volume.Spec) (api.UniqueVolumeName, error) {
	if volumePlugin == nil {
		return "", fmt.Errorf(
			"volumePlugin should not be nil. volumeSpec.Name=%q",
			volumeSpec.Name())
	}

	volumeName, err := volumePlugin.GetVolumeName(volumeSpec)
	if err != nil || volumeName == "" {
		return "", fmt.Errorf(
			"failed to GetVolumeName from volumePlugin for volumeSpec %q err=%v",
			volumeSpec.Name(),
			err)
	}

	return GetUniqueVolumeName(
			volumePlugin.GetPluginName(),
			volumeName),
		nil
}
