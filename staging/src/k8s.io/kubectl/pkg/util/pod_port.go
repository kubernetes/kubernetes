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

package util

import (
	"fmt"

	"k8s.io/api/core/v1"
)

// LookupContainerPortNumberByName find containerPort number by its named port name
func LookupContainerPortNumberByName(pod v1.Pod, name string) (int32, error) {
	for _, ctr := range pod.Spec.Containers {
		for _, ctrportspec := range ctr.Ports {
			if ctrportspec.Name == name {
				return ctrportspec.ContainerPort, nil
			}
		}
	}
	for _, ctr := range pod.Spec.InitContainers {
		for _, ctrportspec := range ctr.Ports {
			if ctrportspec.Name == name {
				return ctrportspec.ContainerPort, nil
			}
		}
	}
	return int32(-1), fmt.Errorf("Pod '%s' does not have a named port '%s'", pod.Name, name)
}
