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

package gce_pd

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
)

// Examines /proc/mounts to find the source device of the PD resource and the
// number of references to that device. Returns both the full device path under
// the /dev tree and the number of references.
func getMountRefCount(mounter mount.Interface, mountPath string) (string, int, error) {
	// TODO(jonesdl) This can be split up into two procedures, finding the device path
	// and finding the number of references. The parsing could also be separated and another
	// utility could determine if a path is an active mount point.

	mps, err := mounter.List()
	if err != nil {
		return "", -1, err
	}

	// Find the device name.
	deviceName := ""
	for i := range mps {
		if mps[i].Path == mountPath {
			deviceName = mps[i].Device
			break
		}
	}

	// Find the number of references to the device.
	refCount := 0
	for i := range mps {
		if mps[i].Device == deviceName {
			refCount++
		}
	}
	return deviceName, refCount, nil
}
