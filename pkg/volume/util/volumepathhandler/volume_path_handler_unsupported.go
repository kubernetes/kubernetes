//go:build !linux

/*
Copyright 2018 The Kubernetes Authors.

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

package volumepathhandler

import (
	"fmt"

	"k8s.io/apimachinery/pkg/types"
)

// AttachFileDevice takes a path to a regular file and makes it available as an
// attached block device.
func (v VolumePathHandler) AttachFileDevice(path string) (string, error) {
	return "", fmt.Errorf("AttachFileDevice not supported for this build.")
}

// DetachFileDevice takes a path to the attached block device and
// detach it from block device.
func (v VolumePathHandler) DetachFileDevice(path string) error {
	return fmt.Errorf("DetachFileDevice not supported for this build.")
}

// GetLoopDevice returns the full path to the loop device associated with the given path.
func (v VolumePathHandler) GetLoopDevice(path string) (string, error) {
	return "", fmt.Errorf("GetLoopDevice not supported for this build.")
}

// FindGlobalMapPathUUIDFromPod finds {pod uuid} bind mount under globalMapPath
// corresponding to map path symlink, and then return global map path with pod uuid.
func (v VolumePathHandler) FindGlobalMapPathUUIDFromPod(pluginDir, mapPath string, podUID types.UID) (string, error) {
	return "", fmt.Errorf("FindGlobalMapPathUUIDFromPod not supported for this build.")
}
