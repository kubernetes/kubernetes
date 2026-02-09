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

package cm

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
)

type podContainerManagerStub struct {
}

var _ PodContainerManager = &podContainerManagerStub{}

func (m *podContainerManagerStub) Exists(_ *v1.Pod) bool {
	return true
}

func (m *podContainerManagerStub) EnsureExists(_ klog.Logger, _ *v1.Pod) error {
	return nil
}

func (m *podContainerManagerStub) GetPodContainerName(_ *v1.Pod) (CgroupName, string) {
	return nil, ""
}

func (m *podContainerManagerStub) Destroy(_ klog.Logger, _ CgroupName) error {
	return nil
}

func (m *podContainerManagerStub) ReduceCPULimits(_ klog.Logger, _ CgroupName) error {
	return nil
}

func (m *podContainerManagerStub) GetAllPodsFromCgroups() (map[types.UID]CgroupName, error) {
	return nil, nil
}

func (m *podContainerManagerStub) IsPodCgroup(cgroupfs string) (bool, types.UID) {
	return false, types.UID("")
}

func (m *podContainerManagerStub) GetPodCgroupMemoryUsage(_ *v1.Pod) (uint64, error) {
	return 0, nil
}

func (m *podContainerManagerStub) GetPodCgroupMemoryLimit(_ *v1.Pod) (uint64, error) {
	return 0, nil
}

func (m *podContainerManagerStub) GetPodCgroupCpuLimit(_ *v1.Pod) (int64, uint64, uint64, error) {
	return 0, 0, 0, nil
}

func (m *podContainerManagerStub) SetPodCgroupMemoryLimit(_ *v1.Pod, _ int64) error {
	return nil
}

func (m *podContainerManagerStub) SetPodCgroupCPULimit(_ klog.Logger, _ *v1.Pod, _ *int64, _, _ *uint64) error {
	return nil
}
