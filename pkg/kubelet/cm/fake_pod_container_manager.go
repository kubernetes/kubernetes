/*
Copyright 2021 The Kubernetes Authors.

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

package cm

import (
	"reflect"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

type FakePodContainerManager struct {
	sync.Mutex
	CalledFunctions []string
	Cgroups         map[types.UID]CgroupName
}

var _ PodContainerManager = &FakePodContainerManager{}

func NewFakePodContainerManager() *FakePodContainerManager {
	return &FakePodContainerManager{
		Cgroups: make(map[types.UID]CgroupName),
	}
}

func (m *FakePodContainerManager) AddPodFromCgroups(pod *kubecontainer.Pod) {
	m.Lock()
	defer m.Unlock()
	m.Cgroups[pod.ID] = []string{pod.Name}
}

func (m *FakePodContainerManager) Exists(_ *v1.Pod) bool {
	m.Lock()
	defer m.Unlock()
	m.CalledFunctions = append(m.CalledFunctions, "Exists")
	return true
}

func (m *FakePodContainerManager) EnsureExists(_ *v1.Pod) error {
	m.Lock()
	defer m.Unlock()
	m.CalledFunctions = append(m.CalledFunctions, "EnsureExists")
	return nil
}

func (m *FakePodContainerManager) GetPodContainerName(_ *v1.Pod) (CgroupName, string) {
	m.Lock()
	defer m.Unlock()
	m.CalledFunctions = append(m.CalledFunctions, "GetPodContainerName")
	return nil, ""
}

func (m *FakePodContainerManager) Destroy(name CgroupName) error {
	m.Lock()
	defer m.Unlock()
	m.CalledFunctions = append(m.CalledFunctions, "Destroy")
	for key, cgname := range m.Cgroups {
		if reflect.DeepEqual(cgname, name) {
			delete(m.Cgroups, key)
			return nil
		}
	}
	return nil
}

func (m *FakePodContainerManager) ReduceCPULimits(_ CgroupName) error {
	m.Lock()
	defer m.Unlock()
	m.CalledFunctions = append(m.CalledFunctions, "ReduceCPULimits")
	return nil
}

func (m *FakePodContainerManager) GetAllPodsFromCgroups() (map[types.UID]CgroupName, error) {
	m.Lock()
	defer m.Unlock()
	m.CalledFunctions = append(m.CalledFunctions, "GetAllPodsFromCgroups")
	// return a copy for the race detector
	grp := make(map[types.UID]CgroupName)
	for key, value := range m.Cgroups {
		grp[key] = value
	}
	return grp, nil
}

func (m *FakePodContainerManager) IsPodCgroup(cgroupfs string) (bool, types.UID) {
	m.Lock()
	defer m.Unlock()
	m.CalledFunctions = append(m.CalledFunctions, "IsPodCgroup")
	return false, types.UID("")
}

func (cm *FakePodContainerManager) GetPodCgroupMemoryUsage(_ *v1.Pod) (uint64, error) {
	cm.Lock()
	defer cm.Unlock()
	cm.CalledFunctions = append(cm.CalledFunctions, "GetPodCgroupMemoryUsage")
	return 0, nil
}

func (cm *FakePodContainerManager) GetPodCgroupConfig(_ *v1.Pod, _ v1.ResourceName) (*ResourceConfig, error) {
	cm.Lock()
	defer cm.Unlock()
	cm.CalledFunctions = append(cm.CalledFunctions, "GetPodCgroupConfig")
	return nil, nil
}

func (cm *FakePodContainerManager) SetPodCgroupConfig(_ *v1.Pod, _ v1.ResourceName, _ *ResourceConfig) error {
	cm.Lock()
	defer cm.Unlock()
	cm.CalledFunctions = append(cm.CalledFunctions, "SetPodCgroupConfig")
	return nil
}

func (m *FakePodContainerManager) DoNotEnforceCPULimits() {
}

func (m *FakePodContainerManager) SetUnlimitedCPUQuota(*v1.Pod) {
}
