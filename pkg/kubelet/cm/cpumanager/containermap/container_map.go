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

// ContainerMap maps (containerID)->(*v1.Pod, *v1.Container)
type ContainerMap map[string]struct {
	pod       *v1.Pod
	container *v1.Container
}

// NewContainerMap creates a new ContainerMap struct
func NewContainerMap() ContainerMap {
	return make(ContainerMap)
}

// Add adds a mapping of (containerID)->(*v1.Pod, *v1.Container) to the ContainerMap
func (cm ContainerMap) Add(p *v1.Pod, c *v1.Container, containerID string) {
	cm[containerID] = struct {
		pod       *v1.Pod
		container *v1.Container
	}{p, c}
}

// Remove removes a mapping of (containerID)->(*v1.Pod, *.v1.Container) from the ContainerMap
func (cm ContainerMap) Remove(containerID string) {
	delete(cm, containerID)
}

// GetContainerID retrieves a ContainerID from the ContainerMap
func (cm ContainerMap) GetContainerID(p *v1.Pod, c *v1.Container) (string, error) {
	for key, val := range cm {
		if val.pod.UID == p.UID && val.container.Name == c.Name {
			return key, nil
		}
	}
	return "", fmt.Errorf("container %s not in ContainerMap for pod %s", c.Name, p.UID)
}

// GetContainerRef retrieves a (*v1.Pod, *v1.Container) pair from the ContainerMap
func (cm ContainerMap) GetContainerRef(containerID string) (*v1.Pod, *v1.Container, error) {
	if _, exists := cm[containerID]; !exists {
		return nil, nil, fmt.Errorf("containerID %s not in ContainerMap", containerID)
	}
	return cm[containerID].pod, cm[containerID].container, nil
}
