/*
Copyright 2014 Google Inc. All rights reserved.

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

package volume

import (
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
)

// Examines /proc/mounts to find all other references to the device referenced
// by mountPath.
func GetMountRefs(mounter mount.Interface, mountPath string) ([]string, error) {
	mps, err := mounter.List()
	if err != nil {
		return nil, err
	}

	// Find the device name.
	deviceName := ""
	for i := range mps {
		if mps[i].Path == mountPath {
			deviceName = mps[i].Device
			break
		}
	}

	// Find all references to the device.
	var refs []string
	for i := range mps {
		if mps[i].Device == deviceName && mps[i].Path != mountPath {
			refs = append(refs, mps[i].Path)
		}
	}
	return refs, nil
}

// given a device path, find its reference count from /proc/mounts
func GetDeviceRefCount(mounter mount.Interface, deviceName string) (int, error) {
	mps, err := mounter.List()
	if err != nil {
		return -1, err
	}

	// Find the number of references to the device.
	refCount := 0
	for i := range mps {
		if strings.HasPrefix(mps[i].Device, deviceName) {
			refCount++
		}
	}
	return refCount, nil
}

// given a device path, find the mount on that device from /proc/mounts
func GetMountFromDevicePath(mounter mount.Interface, deviceName string) (string, int, error) {
	// mostly borrowed from gce-pd
	// export it so other block volume plugin can use it.
	mps, err := mounter.List()
	if err != nil {
		return "", -1, err
	}
	mnt := ""
	// Find the number of references to the device.
	refCount := 0
	for i := range mps {
		if mps[i].Device == deviceName {
			mnt = mps[i].Path
			refCount++
		}
	}
	return mnt, refCount, nil
}
