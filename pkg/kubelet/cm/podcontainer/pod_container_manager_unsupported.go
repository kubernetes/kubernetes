// +build !linux

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

package podcontainer

import (
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

type unsupportedPodContainerManager struct {
}

var _ PodContainerManager = &unsupportedPodContainerManager{}

func (m *unsupportedPodContainerManager) Exists(_ *v1.Pod) bool {
	return true
}

func (m *unsupportedPodContainerManager) EnsureExists(_ *v1.Pod) error {
	return nil
}

func (m *unsupportedPodContainerManager) GetPodContainerName(_ *v1.Pod) (CgroupName, string) {
	return "", ""
}

func (m *unsupportedPodContainerManager) ReduceCPULimits(_ CgroupName) error {
	return nil
}

func (m *unsupportedPodContainerManager) GetAllPodsFromCgroups() (map[types.UID]CgroupName, error) {
	return nil, nil
}

func (m *unsupportedPodContainerManager) Destroy(name CgroupName) error {
	return nil
}
