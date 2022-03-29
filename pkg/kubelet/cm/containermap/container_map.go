/*
Copyright 2019 The Kubernetes Authors.

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

package containermap

import (
	"fmt"
)

// ContainerMap maps (containerID)->(*v1.Pod, *v1.Container)
type ContainerMap map[string]struct {
	podUID        string
	containerName string
}

// NewContainerMap creates a new ContainerMap struct
func NewContainerMap() ContainerMap {
	return make(ContainerMap)
}

// Add adds a mapping of (containerID)->(podUID, containerName) to the ContainerMap
func (cm ContainerMap) Add(podUID, containerName, containerID string) {
	cm[containerID] = struct {
		podUID        string
		containerName string
	}{podUID, containerName}
}

// RemoveByContainerID removes a mapping of (containerID)->(podUID, containerName) from the ContainerMap
func (cm ContainerMap) RemoveByContainerID(containerID string) {
	delete(cm, containerID)
}

// RemoveByContainerRef removes a mapping of (containerID)->(podUID, containerName) from the ContainerMap
func (cm ContainerMap) RemoveByContainerRef(podUID, containerName string) {
	containerID, err := cm.GetContainerID(podUID, containerName)
	if err == nil {
		cm.RemoveByContainerID(containerID)
	}
}

// GetContainerID retrieves a ContainerID from the ContainerMap
func (cm ContainerMap) GetContainerID(podUID, containerName string) (string, error) {
	for key, val := range cm {
		if val.podUID == podUID && val.containerName == containerName {
			return key, nil
		}
	}
	return "", fmt.Errorf("container %s not in ContainerMap for pod %s", containerName, podUID)
}

// GetContainerRef retrieves a (podUID, containerName) pair from the ContainerMap
func (cm ContainerMap) GetContainerRef(containerID string) (string, string, error) {
	if _, exists := cm[containerID]; !exists {
		return "", "", fmt.Errorf("containerID %s not in ContainerMap", containerID)
	}
	return cm[containerID].podUID, cm[containerID].containerName, nil
}

// Visit invoke visitor function to walks all of the entries in the container map
func (cm ContainerMap) Visit(visitor func(podUID, containerName, containerID string)) {
	for k, v := range cm {
		visitor(v.podUID, v.containerName, k)
	}
}
