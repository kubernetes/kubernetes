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

	"k8s.io/api/core/v1"
)

// ContainerMap maps (containerID)->(podUID, containerName)
type ContainerMap map[string]map[string]string

// NewContainerMap creates a new ContainerMap struct
func NewContainerMap() ContainerMap {
	return make(ContainerMap)
}

// Add adds a mapping of (containerID)->(podUID, containerName) to the ContainerMap
func (cm ContainerMap) Add(p *v1.Pod, c *v1.Container, containerID string) {
	podUID := string(p.UID)
	if _, exists := cm[podUID]; !exists {
		cm[podUID] = make(map[string]string)
	}
	cm[podUID][c.Name] = containerID
}

// Remove removes a mapping of (containerID)->(podUID, containerName) from the ContainerMap
func (cm ContainerMap) Remove(containerID string) {
	found := false
	for podUID := range cm {
		for containerName := range cm[podUID] {
			if containerID == cm[podUID][containerName] {
				delete(cm[podUID], containerName)
				found = true
				break
			}
		}
		if len(cm[podUID]) == 0 {
			delete(cm, podUID)
		}
		if found {
			break
		}
	}
}

// Get retrieves a ContainerID from the ContainerMap
func (cm ContainerMap) Get(p *v1.Pod, c *v1.Container) (string, error) {
	podUID := string(p.UID)
	if _, exists := cm[podUID]; !exists {
		return "", fmt.Errorf("pod %s not in ContainerMap", podUID)
	}
	if _, exists := cm[podUID][c.Name]; !exists {
		return "", fmt.Errorf("container %s not in ContainerMap for pod %s", c.Name, podUID)
	}
	return cm[podUID][c.Name], nil
}
