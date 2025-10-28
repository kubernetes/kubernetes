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

package cache

import (
	"context"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	klog "k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

func TestExtendedResourceCache(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	tCtx, tCancel := context.WithCancel(ctx)

	client := fake.NewClientset()
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	logger := klog.Background()
	cache := NewExtendedResourceCache(informerFactory.Resource().V1().DeviceClasses(), logger)
	deviceClassInformer := informerFactory.Resource().V1().DeviceClasses()
	if _, err := deviceClassInformer.Informer().AddEventHandler(cache); err != nil {
		logger.Error(err, "Failed to add event handler for device classes")
	}
	informerFactory.Start(tCtx.Done())
	t.Cleanup(func() {
		// Need to cancel before waiting for the shutdown.
		tCancel()
		// Now we can wait for all goroutines to stop.
		informerFactory.Shutdown()
	})
	informerFactory.WaitForCacheSync(tCtx.Done())

	// Test with a device class that has an explicit extended resource name
	now := time.Now()
	deviceClass1 := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-class",
			CreationTimestamp: metav1.Time{
				Time: now,
			},
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: ptr.To("example.com/gpu"),
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
	deviceClass3 := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-class-3",
			CreationTimestamp: metav1.Time{
				Time: now.Add(-24 * time.Hour),
			},
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: ptr.To("example.com/gpu"),
		},
	}
	deviceClass4 := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-class-4",
			CreationTimestamp: metav1.Time{
				Time: now.Add(time.Hour),
			},
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: ptr.To("example.com/gpu"),
		},
	}
	deviceClass0 := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-class-0",
			CreationTimestamp: metav1.Time{
				Time: now.Add(time.Hour),
			},
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: ptr.To("example.com/gpu"),
		},
	}

	// Test adding device classes
	_, err := client.ResourceV1().DeviceClasses().Create(tCtx, deviceClass1, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create device class: %v", err)
	}
	_, err = client.ResourceV1().DeviceClasses().Create(tCtx, deviceClass2, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create device class: %v", err)
	}
	time.Sleep(1 * time.Second)

	// Verify explicit mapping
	deviceClassName := cache.GetDeviceClass("example.com/gpu")
	if deviceClassName != "gpu-class" {
		t.Errorf("Expected to find device class 'gpu-class' for 'example.com/gpu', got %s", deviceClassName)
	}

	// Verify default mapping
	defaultResourceName := v1.ResourceName("deviceclass.resource.kubernetes.io/fpga-class")
	deviceClassName = cache.GetDeviceClass(defaultResourceName)
	if deviceClassName != "fpga-class" {
		t.Errorf("Expected to find device class 'fpga-class' for '%s', got %s", defaultResourceName, deviceClassName)
	}

	// Verify both device classes have default mappings
	if cache.GetDeviceClass("deviceclass.resource.kubernetes.io/gpu-class") != "gpu-class" {
		t.Error("Expected default mapping for gpu-class")
	}

	// deviceClass3 is older than deviceClass1, hence it won't replace deviceClass1
	_, err = client.ResourceV1().DeviceClasses().Create(tCtx, deviceClass3, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create device class: %v", err)
	}
	time.Sleep(1 * time.Second)

	// should keep deviceClass1, since it is newer than deviceClass3
	deviceClassName = cache.GetDeviceClass("example.com/gpu")
	if deviceClassName != "gpu-class" {
		t.Errorf("Expected to find device class 'gpu-class' for 'example.com/gpu', got %s", deviceClassName)
	}

	// deviceClass4 is newer than deviceClass1, hence it will replace deviceClass1
	_, err = client.ResourceV1().DeviceClasses().Create(tCtx, deviceClass4, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create device class: %v", err)
	}
	time.Sleep(1 * time.Second)

	// deviceClass4 replaces deviceClass1, since it is newer with the same example.com/gpu extended resource name
	deviceClassName = cache.GetDeviceClass("example.com/gpu")
	if deviceClassName != "gpu-class-4" {
		t.Errorf("Expected to find device class 'gpu-class' for 'example.com/gpu', got %s", deviceClassName)
	}

	// deviceClass0 is created at the same time as deviceClass4, but its name is alphabetically ordered earlier,
	//  hence it will replace deviceClass4
	_, err = client.ResourceV1().DeviceClasses().Create(tCtx, deviceClass0, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create device class: %v", err)
	}
	time.Sleep(1 * time.Second)

	// deviceClass0 replaces deviceClass4, it is created at the same time as deviceClass4, but its name is
	// alphabetically ordered earlier
	deviceClassName = cache.GetDeviceClass("example.com/gpu")
	if deviceClassName != "gpu-class-0" {
		t.Errorf("Expected to find device class 'gpu-class' for 'example.com/gpu', got %s", deviceClassName)
	}

	// Test modifying a device class
	deviceClass0Modified := deviceClass0.DeepCopy()
	deviceClass0Modified.Spec.ExtendedResourceName = ptr.To("test.com/gpu")
	_, err = client.ResourceV1().DeviceClasses().Update(tCtx, deviceClass0Modified, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update device class: %v", err)
	}
	time.Sleep(1 * time.Second)

	// Should have the new mapping
	deviceClassName = cache.GetDeviceClass("test.com/gpu")
	if deviceClassName != "gpu-class-0" {
		t.Errorf("Expected to find device class 'gpu-class' for 'test.com/gpu' after modification, got %s", deviceClassName)
	}

	// Test deleting a device class
	err = client.ResourceV1().DeviceClasses().Delete(tCtx, deviceClass0.Name, metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Failed to delete device class: %v", err)
	}
	time.Sleep(1 * time.Second)

	deviceClassName = cache.GetDeviceClass("test.com/gpu")
	if deviceClassName != "" {
		t.Errorf("Expected 'test.com/gpu' to be removed after deleting device class, got %s", deviceClassName)
	}

	// Should not have the old mapping
	deviceClassName = cache.GetDeviceClass("example.com/gpu")
	if deviceClassName != "" {
		t.Errorf("Expected 'test.com/gpu' to be removed after modifying device class, got %s", deviceClassName)
	}
}
