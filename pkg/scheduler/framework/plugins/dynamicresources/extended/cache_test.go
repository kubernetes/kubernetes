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

package extended

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	klog "k8s.io/klog/v2"
)

func TestExtendedResourceCache(t *testing.T) {
	logger := klog.Background()
	cache := NewExtendedResourceCache(nil, logger)

	// Test with a device class that has an explicit extended resource name
	deviceClass1 := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-class",
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: strPtr("nvidia.com/gpu"),
		},
	}

	// Test with a device class that uses the default mapping
	deviceClass2 := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "fpga-class",
		},
		Spec: resourceapi.DeviceClassSpec{
			// No explicit extended resource name
		},
	}

	// Test adding device classes
	cache.OnDeviceClassEvent(watch.Added, deviceClass1)
	cache.OnDeviceClassEvent(watch.Added, deviceClass2)

	// Verify explicit mapping
	deviceClassName, exists := cache.GetDeviceClass("nvidia.com/gpu")
	if !exists || deviceClassName != "gpu-class" {
		t.Errorf("Expected to find device class 'gpu-class' for 'nvidia.com/gpu', got %s (exists: %v)", deviceClassName, exists)
	}

	// Verify default mapping
	defaultResourceName := v1.ResourceName("deviceclass.resource.kubernetes.io/fpga-class")
	deviceClassName, exists = cache.GetDeviceClass(defaultResourceName)
	if !exists || deviceClassName != "fpga-class" {
		t.Errorf("Expected to find device class 'fpga-class' for '%s', got %s (exists: %v)", defaultResourceName, deviceClassName, exists)
	}

	// Test getting all mappings
	allMappings := cache.GetAllMappings()
	expectedMappings := 3 // nvidia.com/gpu, deviceclass.resource.kubernetes.io/gpu-class, deviceclass.resource.kubernetes.io/fpga-class
	if len(allMappings) != expectedMappings {
		t.Errorf("Expected %d mappings, got %d: %v", expectedMappings, len(allMappings), allMappings)
	}

	// Test deleting a device class
	cache.OnDeviceClassEvent(watch.Deleted, deviceClass1)
	_, exists = cache.GetDeviceClass("nvidia.com/gpu")
	if exists {
		t.Error("Expected 'nvidia.com/gpu' to be removed after deleting device class")
	}

	// Test modifying a device class
	deviceClass1Modified := deviceClass1.DeepCopy()
	deviceClass1Modified.Spec.ExtendedResourceName = strPtr("amd.com/gpu")
	cache.OnDeviceClassEvent(watch.Modified, deviceClass1Modified)

	// Should have the new mapping
	deviceClassName, exists = cache.GetDeviceClass("amd.com/gpu")
	if !exists || deviceClassName != "gpu-class" {
		t.Errorf("Expected to find device class 'gpu-class' for 'amd.com/gpu' after modification, got %s (exists: %v)", deviceClassName, exists)
	}

	// Should not have the old mapping
	_, exists = cache.GetDeviceClass("nvidia.com/gpu")
	if exists {
		t.Error("Expected 'nvidia.com/gpu' to be removed after modifying device class")
	}
}

func strPtr(s string) *string {
	return &s
}
