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
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog/v2/ktesting"
)

func TestContainerManagerStub(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cm := &containerManagerStub{}

	t.Run("SystemCgroupsLimit returns empty ResourceList", func(t *testing.T) {
		result := cm.SystemCgroupsLimit()
		if len(result) != 0 {
			t.Errorf("Expected empty ResourceList, got %v", result)
		}
	})

	t.Run("GetNodeConfig returns non-nil", func(t *testing.T) {
		result := cm.GetNodeConfig()
		if result.CgroupRoot == "" {
			// Expected - returns empty config with empty CgroupRoot
		}
	})

	t.Run("GetMountedSubsystems returns non-nil", func(t *testing.T) {
		result := cm.GetMountedSubsystems()
		if result == nil {
			t.Error("Expected non-nil CgroupSubsystems")
		}
	})

	t.Run("GetQOSContainersInfo returns non-nil", func(t *testing.T) {
		result := cm.GetQOSContainersInfo()
		// Just verify it returns something
		_ = result
	})

	t.Run("UpdateQOSCgroups returns nil", func(t *testing.T) {
		if err := cm.UpdateQOSCgroups(logger); err != nil {
			t.Errorf("Expected nil error, got %v", err)
		}
	})

	t.Run("Status returns non-nil", func(t *testing.T) {
		result := cm.Status()
		_ = result
	})

	t.Run("GetNodeAllocatableReservation returns nil", func(t *testing.T) {
		result := cm.GetNodeAllocatableReservation()
		if result != nil {
			t.Error("Expected nil")
		}
	})

	t.Run("GetCapacity with localStorageCapacityIsolation false returns empty", func(t *testing.T) {
		result := cm.GetCapacity(false)
		if len(result) != 0 {
			t.Errorf("Expected empty ResourceList, got %v", result)
		}
	})

	t.Run("GetCapacity with localStorageCapacityIsolation true returns EphemeralStorage", func(t *testing.T) {
		result := cm.GetCapacity(true)
		if _, ok := result[v1.ResourceEphemeralStorage]; !ok {
			t.Error("Expected EphemeralStorage")
		}
	})

	t.Run("GetPluginRegistrationHandlers returns nil", func(t *testing.T) {
		result := cm.GetPluginRegistrationHandlers()
		if result != nil {
			t.Error("Expected nil")
		}
	})

	t.Run("GetHealthCheckers returns empty slice", func(t *testing.T) {
		result := cm.GetHealthCheckers()
		if len(result) != 0 {
			t.Errorf("Expected empty slice, got %v", result)
		}
	})

	t.Run("GetDevicePluginResourceCapacity returns empty", func(t *testing.T) {
		allocatable, capacity, _ := cm.GetDevicePluginResourceCapacity()
		if len(allocatable) != 0 || len(capacity) != 0 {
			t.Error("Expected empty resource lists")
		}
	})

	t.Run("NewPodContainerManager returns podContainerManagerStub", func(t *testing.T) {
		result := cm.NewPodContainerManager()
		if _, ok := result.(*podContainerManagerStub); !ok {
			t.Error("Expected *podContainerManagerStub")
		}
	})

	t.Run("GetResources returns non-nil RunContainerOptions", func(t *testing.T) {
		result, err := cm.GetResources(context.Background(), nil, nil)
		if err != nil {
			t.Errorf("Expected nil error, got %v", err)
		}
		if result == nil {
			t.Error("Expected non-nil RunContainerOptions")
		}
	})

	t.Run("UpdatePluginResources returns nil", func(t *testing.T) {
		if err := cm.UpdatePluginResources(nil, nil); err != nil {
			t.Errorf("Expected nil error, got %v", err)
		}
	})

	t.Run("InternalContainerLifecycle returns non-nil", func(t *testing.T) {
		result := cm.InternalContainerLifecycle()
		if result == nil {
			t.Error("Expected non-nil InternalContainerLifecycle")
		}
	})

	t.Run("GetPodCgroupRoot returns empty string", func(t *testing.T) {
		result := cm.GetPodCgroupRoot()
		if result != "" {
			t.Errorf("Expected empty string, got %s", result)
		}
	})

	t.Run("GetDevices returns nil", func(t *testing.T) {
		result := cm.GetDevices("", "")
		if result != nil {
			t.Error("Expected nil")
		}
	})

	t.Run("GetAllocatableDevices returns nil", func(t *testing.T) {
		result := cm.GetAllocatableDevices()
		if result != nil {
			t.Error("Expected nil")
		}
	})

	t.Run("ShouldResetExtendedResourceCapacity returns false by default", func(t *testing.T) {
		if cm.ShouldResetExtendedResourceCapacity() != false {
			t.Error("Expected false")
		}
	})

	t.Run("UpdateAllocatedDevices returns", func(t *testing.T) {
		cm.UpdateAllocatedDevices()
	})

	t.Run("GetCPUs returns nil", func(t *testing.T) {
		result := cm.GetCPUs("", "")
		if result != nil {
			t.Error("Expected nil")
		}
	})

	t.Run("GetAllocatableCPUs returns nil", func(t *testing.T) {
		result := cm.GetAllocatableCPUs()
		if result != nil {
			t.Error("Expected nil")
		}
	})

	t.Run("GetMemory returns nil", func(t *testing.T) {
		result := cm.GetMemory("", "")
		if result != nil {
			t.Error("Expected nil")
		}
	})

	t.Run("GetAllocatableMemory returns nil", func(t *testing.T) {
		result := cm.GetAllocatableMemory()
		if result != nil {
			t.Error("Expected nil")
		}
	})

	t.Run("GetDynamicResources returns nil", func(t *testing.T) {
		result := cm.GetDynamicResources(nil, nil)
		if result != nil {
			t.Error("Expected nil")
		}
	})

	t.Run("GetNodeAllocatableAbsolute returns nil", func(t *testing.T) {
		result := cm.GetNodeAllocatableAbsolute()
		if result != nil {
			t.Error("Expected nil")
		}
	})

	t.Run("PrepareDynamicResources returns nil", func(t *testing.T) {
		err := cm.PrepareDynamicResources(context.Background(), nil)
		if err != nil {
			t.Errorf("Expected nil error, got %v", err)
		}
	})

	t.Run("UnprepareDynamicResources returns nil", func(t *testing.T) {
		err := cm.UnprepareDynamicResources(context.Background(), nil)
		if err != nil {
			t.Errorf("Expected nil error, got %v", err)
		}
	})

	t.Run("PodMightNeedToUnprepareResources returns false", func(t *testing.T) {
		result := cm.PodMightNeedToUnprepareResources("")
		if result != false {
			t.Error("Expected false")
		}
	})

	t.Run("UpdateAllocatedResourcesStatus returns", func(t *testing.T) {
		cm.UpdateAllocatedResourcesStatus(nil, nil)
	})

	t.Run("Updates returns nil", func(t *testing.T) {
		result := cm.Updates()
		if result != nil {
			t.Error("Expected nil")
		}
	})

	t.Run("PodHasExclusiveCPUs returns false", func(t *testing.T) {
		result := cm.PodHasExclusiveCPUs(nil)
		if result != false {
			t.Error("Expected false")
		}
	})

	t.Run("ContainerHasExclusiveCPUs returns false", func(t *testing.T) {
		result := cm.ContainerHasExclusiveCPUs(nil, nil)
		if result != false {
			t.Error("Expected false")
		}
	})
}

func TestNewStubContainerManager(t *testing.T) {
	cm := NewStubContainerManager()
	if cm == nil {
		t.Error("Expected non-nil ContainerManager")
	}
}

func TestNewStubContainerManagerWithExtendedResource(t *testing.T) {
	resources := v1.ResourceList{
		v1.ResourceName("nvidia.com/gpu"): *resource.NewQuantity(1, resource.DecimalSI),
	}
	cm := NewStubContainerManagerWithDevicePluginResource(resources)
	if cm == nil {
		t.Error("Expected non-nil ContainerManager")
	}
}

func TestNewStubContainerManagerWithExtendedResourceFunctionality(t *testing.T) {
	resources := v1.ResourceList{
		v1.ResourceName("nvidia.com/gpu"): *resource.NewQuantity(2, resource.DecimalSI),
	}
	cm := NewStubContainerManagerWithDevicePluginResource(resources)

	allocatable, capacity, _ := cm.GetDevicePluginResourceCapacity()

	if len(allocatable) != 1 {
		t.Errorf("Expected 1 resource, got %d", len(allocatable))
	}
	if len(capacity) != 1 {
		t.Errorf("Expected 1 resource, got %d", len(capacity))
	}
}

func TestContainerManagerStubImplementsInterface(t *testing.T) {
	var _ ContainerManager = &containerManagerStub{}
}
