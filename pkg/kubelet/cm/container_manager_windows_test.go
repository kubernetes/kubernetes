//go:build windows

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

	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/cm/devicemanager"
)

// fakeDeviceManager is a minimal devicemanager.Manager stub. It embeds the
// interface so methods not built by the test default to nil; only
// GetAllocatableDevices is overridden.
type fakeDeviceManager struct {
	devicemanager.Manager
	allocatableDevices devicemanager.ResourceDeviceInstances
}

func (f *fakeDeviceManager) GetAllocatableDevices(_ klog.Logger) devicemanager.ResourceDeviceInstances {
	return f.allocatableDevices
}

func TestGetAllocatableDevices(t *testing.T) {
	resourceName := "example.com/res1"
	allocatable := devicemanager.NewResourceDeviceInstances()
	// Devices without topology information are still reported, mirroring
	// device plugins that do not provide NUMA topology.
	allocatable[resourceName] = devicemanager.DeviceInstances{
		"dev-1": &pluginapi.Device{ID: "dev-1"},
		"dev-2": &pluginapi.Device{ID: "dev-2"},
	}

	logger, _ := ktesting.NewTestContext(t)
	cm := &containerManagerImpl{
		deviceManager: &fakeDeviceManager{allocatableDevices: allocatable},
	}

	got := cm.GetAllocatableDevices(logger)

	// containerDevicesFromResourceDeviceInstances emits one ContainerDevices
	// entry per device, each carrying the resource name and a single device id.
	if len(got) != 2 {
		t.Fatalf("expected 2 allocatable device entries, got %d: %v", len(got), got)
	}
	expected := map[string]struct{}{
		"dev-1": {},
		"dev-2": {},
	}
	for _, cd := range got {
		if cd.ResourceName != resourceName {
			t.Errorf("expected resource %q, got %q", resourceName, cd.ResourceName)
		}
		if len(cd.DeviceIds) != 1 {
			t.Errorf("expected a single device id for %q, got %v", cd.ResourceName, cd.DeviceIds)
			continue
		}
		id := cd.DeviceIds[0]
		if _, ok := expected[id]; !ok {
			t.Errorf("unexpected device id %q", id)
		}
	}
}
