//go:build linux

/*
Copyright 2025 The Kubernetes Authors.

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
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2/ktesting"
)

func TestPodContainerManagerStub(t *testing.T) {
	stub := &podContainerManagerStub{}

	logger, _ := ktesting.NewTestContext(t)
	pod := &v1.Pod{}

	t.Run("Exists returns true", func(t *testing.T) {
		if !stub.Exists(pod) {
			t.Error("Expected Exists to return true")
		}
	})

	t.Run("EnsureExists returns nil", func(t *testing.T) {
		if err := stub.EnsureExists(logger, pod); err != nil {
			t.Errorf("Expected nil error, got %v", err)
		}
	})

	t.Run("GetPodContainerName returns nil", func(t *testing.T) {
		name, _ := stub.GetPodContainerName(pod)
		if name != nil {
			t.Error("Expected nil CgroupName")
		}
	})

	t.Run("Destroy returns nil", func(t *testing.T) {
		if err := stub.Destroy(logger, nil); err != nil {
			t.Errorf("Expected nil error, got %v", err)
		}
	})

	t.Run("ReduceCPULimits returns nil", func(t *testing.T) {
		if err := stub.ReduceCPULimits(logger, nil); err != nil {
			t.Errorf("Expected nil error, got %v", err)
		}
	})

	t.Run("GetAllPodsFromCgroups returns nil", func(t *testing.T) {
		result, err := stub.GetAllPodsFromCgroups()
		if err != nil {
			t.Errorf("Expected nil error, got %v", err)
		}
		if result != nil {
			t.Error("Expected nil map")
		}
	})

	t.Run("IsPodCgroup returns false", func(t *testing.T) {
		result, _ := stub.IsPodCgroup("test")
		if result != false {
			t.Error("Expected false")
		}
	})

	t.Run("GetPodCgroupMemoryUsage returns 0", func(t *testing.T) {
		result, err := stub.GetPodCgroupMemoryUsage(pod)
		if err != nil {
			t.Errorf("Expected nil error, got %v", err)
		}
		if result != 0 {
			t.Errorf("Expected 0, got %d", result)
		}
	})

	t.Run("GetPodCgroupMemoryLimit returns 0", func(t *testing.T) {
		result, err := stub.GetPodCgroupMemoryLimit(pod)
		if err != nil {
			t.Errorf("Expected nil error, got %v", err)
		}
		if result != 0 {
			t.Errorf("Expected 0, got %d", result)
		}
	})

	t.Run("GetPodCgroupCpuLimit returns 0", func(t *testing.T) {
		result, _, _, err := stub.GetPodCgroupCpuLimit(pod)
		if err != nil {
			t.Errorf("Expected nil error, got %v", err)
		}
		if result != 0 {
			t.Errorf("Expected 0, got %d", result)
		}
	})

	t.Run("SetPodCgroupMemoryLimit returns nil", func(t *testing.T) {
		if err := stub.SetPodCgroupMemoryLimit(pod, 0); err != nil {
			t.Errorf("Expected nil error, got %v", err)
		}
	})

	t.Run("SetPodCgroupCPULimit returns nil", func(t *testing.T) {
		if err := stub.SetPodCgroupCPULimit(logger, pod, nil, nil, nil); err != nil {
			t.Errorf("Expected nil error, got %v", err)
		}
	})
}

func TestPodContainerManagerStubImplementsInterface(t *testing.T) {
	var _ PodContainerManager = &podContainerManagerStub{}
}

func TestIsPodCgroupReturnsEmptyUID(t *testing.T) {
	stub := &podContainerManagerStub{}
	_, uid := stub.IsPodCgroup("test")
	if uid != types.UID("") {
		t.Errorf("Expected empty UID, got %s", uid)
	}
}
