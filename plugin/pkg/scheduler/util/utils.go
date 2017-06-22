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

package util

import (
	"k8s.io/api/core/v1"
)

// GetUsedPorts returns the used host ports of Pods: if 'port' was used, a 'port:true' pair
// will be in the result; but it does not resolve port conflict.
func GetUsedPorts(pods ...*v1.Pod) map[int]bool {
	ports := make(map[int]bool)
	for _, pod := range pods {
		for j := range pod.Spec.Containers {
			container := &pod.Spec.Containers[j]
			for k := range container.Ports {
				podPort := &container.Ports[k]
				// "0" is explicitly ignored in PodFitsHostPorts,
				// which is the only function that uses this value.
				if podPort.HostPort != 0 {
					ports[int(podPort.HostPort)] = true
				}
			}
		}
	}
	return ports
}
