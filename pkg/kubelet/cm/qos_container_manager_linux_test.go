//go:build linux
// +build linux

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
	"fmt"
	libcontainercgroups "github.com/opencontainers/runc/libcontainer/cgroups"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func activeTestPods() []*v1.Pod {
	return []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:       "76543210",
				Name:      "best-effort-pod",
				Namespace: "test",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "foo",
						Image: "busybox",
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:       "12345678",
				Name:      "guaranteed-pod",
				Namespace: "test",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "foo",
						Image: "busybox",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("128Mi"),
								v1.ResourceCPU:    resource.MustParse("1"),
							},
							Limits: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("128Mi"),
								v1.ResourceCPU:    resource.MustParse("1"),
							},
						},
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:       "87654321",
				Name:      "burstable-pod-1",
				Namespace: "test",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "foo",
						Image: "busybox",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("128Mi"),
								v1.ResourceCPU:    resource.MustParse("1"),
							},
							Limits: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("256Mi"),
								v1.ResourceCPU:    resource.MustParse("2"),
							},
						},
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:       "01234567",
				Name:      "burstable-pod-2",
				Namespace: "test",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "foo",
						Image: "busybox",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("256Mi"),
								v1.ResourceCPU:    resource.MustParse("2"),
							},
						},
					},
				},
			},
		},
	}
}

func getTestNodeAllocatable() v1.ResourceList {
	return v1.ResourceList{
		v1.ResourceCPU:    resource.MustParse("5"),
		v1.ResourceMemory: resource.MustParse("1Gi"),
	}
}

func createTestQOSContainerManager(nodeConfig NodeConfig, cgroupDriver string) (QOSContainerManager, error) {
	subsystems, err := GetCgroupSubsystems()
	if err != nil {
		return nil, fmt.Errorf("failed to get mounted cgroup subsystems: %v", err)
	}

	cgroupRoot := ParseCgroupfsToCgroupName("/")
	cgroupRoot = NewCgroupName(cgroupRoot, defaultNodeAllocatableCgroupName)
	cgroupManager := NewCgroupManager(subsystems, cgroupDriver)

	qosContainerManager, _ := NewQOSContainerManager(subsystems, cgroupRoot, nodeConfig, cgroupManager)
	return qosContainerManager, nil
}

func TestQoSContainerCgroup(t *testing.T) {
	nodeConfig := NodeConfig{
		CgroupsPerQOS: true,
		QOSReserved: map[v1.ResourceName]int64{
			v1.ResourceMemory: 100,
		},
	}
	m, err := createTestQOSContainerManager(nodeConfig, "cgroupfs")
	assert.Nil(t, err)
	qcm, ok := m.(*qosContainerManagerImpl)
	if !ok {
		t.Errorf("unexpected type for %v", m)
	}
	qcm.activePods = activeTestPods
	qcm.getNodeAllocatable = getTestNodeAllocatable

	qosConfigs := map[v1.PodQOSClass]*CgroupConfig{
		v1.PodQOSGuaranteed: {
			Name:               qcm.qosContainersInfo.Guaranteed,
			ResourceParameters: &ResourceConfig{},
		},
		v1.PodQOSBurstable: {
			Name:               qcm.qosContainersInfo.Burstable,
			ResourceParameters: &ResourceConfig{},
		},
		v1.PodQOSBestEffort: {
			Name:               qcm.qosContainersInfo.BestEffort,
			ResourceParameters: &ResourceConfig{},
		},
	}

	err = qcm.setHugePagesConfig(qosConfigs)
	if err != nil {
		t.Errorf("failed to set HugePages config: %v", err)
	}
	hugePageSizes := libcontainercgroups.HugePageSizes()
	assert.Equal(t, len(hugePageSizes), len(qosConfigs[v1.PodQOSGuaranteed].ResourceParameters.HugePageLimit))
	assert.Equal(t, len(hugePageSizes), len(qosConfigs[v1.PodQOSBurstable].ResourceParameters.HugePageLimit))
	assert.Equal(t, len(hugePageSizes), len(qosConfigs[v1.PodQOSBestEffort].ResourceParameters.HugePageLimit))

	err = qcm.setCPUCgroupConfig(qosConfigs)
	if err != nil {
		t.Errorf("failed to set CPU Cgroup config: %v", err)
	}
	bestEffortCPUShares := uint64(MinShares)
	burstableCPUShares := uint64((3000 * SharesPerCPU) / MilliCPUToCPU)
	assert.Equal(t, bestEffortCPUShares, *qosConfigs[v1.PodQOSBestEffort].ResourceParameters.CpuShares)
	assert.Equal(t, burstableCPUShares, *qosConfigs[v1.PodQOSBurstable].ResourceParameters.CpuShares)

	qcm.setMemoryQoS(qosConfigs)
	burstableMin := resource.MustParse("384Mi")
	guaranteedMin := resource.MustParse("128Mi")
	assert.Equal(t, strconv.FormatInt(burstableMin.Value()+guaranteedMin.Value(), 10), qosConfigs[v1.PodQOSGuaranteed].ResourceParameters.Unified["memory.min"])
	assert.Equal(t, strconv.FormatInt(burstableMin.Value(), 10), qosConfigs[v1.PodQOSBurstable].ResourceParameters.Unified["memory.min"])

	qcm.setMemoryReserve(qosConfigs, qcm.qosReserved[v1.ResourceMemory])
	qosMemoryRequests := qcm.getQoSMemoryRequests()
	allocatableResource := resource.MustParse("1Gi")
	burstableLimit := allocatableResource.Value() - qosMemoryRequests[v1.PodQOSGuaranteed]
	bestEffortLimit := burstableLimit - qosMemoryRequests[v1.PodQOSBurstable]
	assert.Equal(t, burstableLimit, *qosConfigs[v1.PodQOSBurstable].ResourceParameters.Memory)
	assert.Equal(t, bestEffortLimit, *qosConfigs[v1.PodQOSBestEffort].ResourceParameters.Memory)
}
