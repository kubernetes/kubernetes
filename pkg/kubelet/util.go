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

package kubelet

import (
	"os"

	v1 "k8s.io/api/core/v1"
)

// dirExists returns true if the path exists and represents a directory.
func dirExists(path string) bool {
	s, err := os.Stat(path)
	if err != nil {
		return false
	}
	return s.IsDir()
}

func isHostPathDevice(volumeName string, volumes []v1.Volume) (string, bool) {
	for _, volume := range volumes {
		if volume.Name == volumeName {
			if volume.HostPath != nil {
				switch *volume.HostPath.Type {
				case v1.HostPathBlockDev, v1.HostPathCharDev:
					return volume.HostPath.Path, true
				}
			}
			break
		}
	}
	return "", false
}
