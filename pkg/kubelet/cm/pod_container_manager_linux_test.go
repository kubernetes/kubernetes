//go:build linux

/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package cm

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2/ktesting"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

// fakeCgroupManagerForMemoryMin is a fake cgroup manager for testing ReconcilePodMemoryMin
// It tracks calls to Set and returns configured errors.
type fakeCgroupManagerForMemoryMin struct {
	setFunc      func(name CgroupName, resourceConfig *ResourceConfig) error
	existingPods map[types.UID]CgroupName
}

func (f *fakeCgroupManagerForMemoryMin) Exists(name CgroupName) bool {
	for _, cgroup := range f.existingPods {
		if cgroup == name {
			return true
		}
	}
	return false
}

func (f *fakeCgroupManagerForMemoryMin) Create(config *CgroupConfig) error {
	return nil
}

func (f *fakeCgroupManagerForMemoryMin) Set(name CgroupName, resourceConfig *ResourceConfig) error {
	if f.setFunc != nil {
		return f.setFunc(name, resourceConfig)
	}
	return nil
}

func (f *fakeCgroupManagerForMemoryMin) GetPodCgroups() []CgroupName {
	var cgroups []CgroupName
	for _, cgroup := range f.existingPods {
		cgroups = append(cgroups, cgroup)
	}
	return cgroups
}

func TestReconcilePodMemoryMin_ClearsMemoryMinWhenMemoryQoSDisabled(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	// Setup a fake cgroup manager that tracks SetCgroupConfig calls
	type setCall struct {
		name           CgroupName
		resourceConfig *ResourceConfig
	}
	var setCalls []setCall

	fakeMgr := &fakeCgroupManagerForMemoryMin{
		setFunc: func(name CgroupName, resourceConfig *ResourceConfig) error {
			setCalls = append(setCalls, setCall{name, resourceConfig})
			return nil
		},
		existingPods: map[types.UID]CgroupName{
			"pod1": NewCgroupName(RootCgroupName, "burstable", "podpod1"),
			"pod2": NewCgroupName(RootCgroupName, "guaranteed", "podpod2"),
		},
	}

	pcm := &podContainerManagerImpl{
		cgroupManager:     fakeMgr,
		qosContainersInfo: QOSContainersInfo{
			Guaranteed: NewCgroupName(RootCgroupName, "guaranteed"),
			Burstable:  NewCgroupName(RootCgroupName, "burstable"),
			BestEffort: NewCgroupName(RootCgroupName, "besteffort"),
		},
		subsystems: &CgroupSubsystems{MountPoints: map[string]string{}},
	}

	// Simulate MemoryQoS disabled
	memoryQoSEnabled := false
	memoryQoSPolicyNone := false

	err := pcm.ReconcilePodMemoryMin(logger, memoryQoSEnabled, memoryQoSPolicyNone)
	require.NoError(t, err)
	require.Len(t, setCalls, 2)
	for _, call := range setCalls {
		require.NotNil(t, call.resourceConfig)
		require.NotNil(t, call.resourceConfig.Unified)
		assert.Equal(t, "0", call.resourceConfig.Unified["memory.min"])
	}
}

func TestReconcilePodMemoryMin_NoOpWhenMemoryQoSEnabled(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	var setCalls []struct{}
	fakeMgr := &fakeCgroupManagerForMemoryMin{
		setFunc: func(name CgroupName, resourceConfig *ResourceConfig) error {
			setCalls = append(setCalls, struct{}{})
			return nil
		},
		existingPods: map[types.UID]CgroupName{
			"pod1": NewCgroupName(RootCgroupName, "burstable", "podpod1"),
		},
	}

	pcm := &podContainerManagerImpl{
		cgroupManager:     fakeMgr,
		qosContainersInfo: QOSContainersInfo{
			Guaranteed: NewCgroupName(RootCgroupName, "guaranteed"),
			Burstable:  NewCgroupName(RootCgroupName, "burstable"),
			BestEffort: NewCgroupName(RootCgroupName, "besteffort"),
		},
		subsystems: &CgroupSubsystems{MountPoints: map[string]string{}},
	}

	// Simulate MemoryQoS enabled and policy not None
	memoryQoSEnabled := true
	memoryQoSPolicyNone := false

	err := pcm.ReconcilePodMemoryMin(logger, memoryQoSEnabled, memoryQoSPolicyNone)
	require.NoError(t, err)
	assert.Len(t, setCalls, 0)
}

func TestReconcilePodMemoryMin_ClearsMemoryMinWhenPolicyNone(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	var setCalls []struct{}
	fakeMgr := &fakeCgroupManagerForMemoryMin{
		setFunc: func(name CgroupName, resourceConfig *ResourceConfig) error {
			setCalls = append(setCalls, struct{}{})
			return nil
		},
		existingPods: map[types.UID]CgroupName{
			"pod1": NewCgroupName(RootCgroupName, "burstable", "podpod1"),
		},
	}

	pcm := &podContainerManagerImpl{
		cgroupManager:     fakeMgr,
		qosContainersInfo: QOSContainersInfo{
			Guaranteed: NewCgroupName(RootCgroupName, "guaranteed"),
			Burstable:  NewCgroupName(RootCgroupName, "burstable"),
			BestEffort: NewCgroupName(RootCgroupName, "besteffort"),
		},
		subsystems: &CgroupSubsystems{MountPoints: map[string]string{}},
	}

	// Simulate MemoryQoS enabled but policy None
	memoryQoSEnabled := true
	memoryQoSPolicyNone := true

	err := pcm.ReconcilePodMemoryMin(logger, memoryQoSEnabled, memoryQoSPolicyNone)
	require.NoError(t, err)
	assert.Len(t, setCalls, 1)
}

func TestReconcilePodMemoryMin_SetErrorAggregation(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	fakeMgr := &fakeCgroupManagerForMemoryMin{
		setFunc: func(name CgroupName, resourceConfig *ResourceConfig) error {
			return errors.New("set error")
		},
		existingPods: map[types.UID]CgroupName{
			"pod1": NewCgroupName(RootCgroupName, "burstable", "podpod1"),
			"pod2": NewCgroupName(RootCgroupName, "guaranteed", "podpod2"),
		},
	}

	pcm := &podContainerManagerImpl{
		cgroupManager:     fakeMgr,
		qosContainersInfo: QOSContainersInfo{
			Guaranteed: NewCgroupName(RootCgroupName, "guaranteed"),
			Burstable:  NewCgroupName(RootCgroupName, "burstable"),
			BestEffort: NewCgroupName(RootCgroupName, "besteffort"),
		},
		subsystems: &CgroupSubsystems{MountPoints: map[string]string{}},
	}

	// Simulate MemoryQoS disabled
	memoryQoSEnabled := false
	memoryQoSPolicyNone := false

	err := pcm.ReconcilePodMemoryMin(logger, memoryQoSEnabled, memoryQoSPolicyNone)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to clear memory.min")
}
