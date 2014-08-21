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

package constraint

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// PortsConflict returns true iff two containers attempt to expose
// the same host port.
func PortsConflict(manifests []api.ContainerManifest) bool {
	hostPorts := map[int]struct{}{}
	for _, manifest := range manifests {
		for _, container := range manifest.Containers {
			for _, port := range container.Ports {
				if port.HostPort == 0 {
					continue
				}
				if _, exists := hostPorts[port.HostPort]; exists {
					return true
				}
				hostPorts[port.HostPort] = struct{}{}
			}
		}
	}
	return false
}
