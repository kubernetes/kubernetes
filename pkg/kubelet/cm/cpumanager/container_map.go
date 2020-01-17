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

package cpumanager

import (
	"fmt"

	"k8s.io/api/core/v1"
)

// containerMap maps (podUID, containerName) -> containerID
type containerMap map[string]map[string]string

func newContainerMap() containerMap {
	return make(containerMap)
}

func (cm containerMap) Add(p *v1.Pod, c *v1.Container, containerID string) {
	podUID := string(p.UID)
	if _, exists := cm[podUID]; !exists {
		cm[podUID] = make(map[string]string)
	}
	cm[podUID][c.Name] = containerID
}

func (cm containerMap) Remove(containerID string) {
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

func (cm containerMap) Get(p *v1.Pod, c *v1.Container) (string, error) {
	podUID := string(p.UID)
	if _, exists := cm[podUID]; !exists {
		return "", fmt.Errorf("pod %s not in containerMap", podUID)
	}
	if _, exists := cm[podUID][c.Name]; !exists {
		return "", fmt.Errorf("container %s not in containerMap for pod %s", c.Name, podUID)
	}
	return cm[podUID][c.Name], nil
}
