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

	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

func TestDeviceClassMapping(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	tCtx, tCancel := context.WithCancel(ctx)

	client := fake.NewClientset()
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	cache := NewDeviceClassMapping(informerFactory)
	informerFactory.Start(tCtx.Done())
	t.Cleanup(func() {
		// Need to cancel before waiting for the shutdown.
		tCancel()
		// Now we can wait for all goroutines to stop.
		informerFactory.Shutdown()
	})
	informerFactory.WaitForCacheSync(tCtx.Done())

	deviceClass1 := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-class",
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: ptr.To("example.com/gpu"),
		},
	}

	deviceClass2 := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "tpu-class",
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
	name, ok := cache.Get("gpu-class")
	if !ok || name != "example.com/gpu" {
		t.Errorf("Expected to find device class 'gpu-class', got %s (exists: %v)", name, ok)
	}
	_, ok = cache.Get("tpu-class")
	if ok {
		t.Errorf("Expected device class 'tpu-class' not found")
	}

	// Test updating device classes
	deviceClass1Modified := deviceClass1.DeepCopy()
	deviceClass1Modified.Spec.ExtendedResourceName = ptr.To("my.com/gpu")
	_, err = client.ResourceV1().DeviceClasses().Update(tCtx, deviceClass1Modified, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update device class: %v", err)
	}

	time.Sleep(1 * time.Second)
	name, ok = cache.Get("gpu-class")
	if !ok || name != "my.com/gpu" {
		t.Errorf("Expected to find device class 'gpu-class' with  'my.com/gpu' after modification, got %s (exists: %v)", name, ok)
	}

	// Test deleting device classes
	err = client.ResourceV1().DeviceClasses().Delete(tCtx, deviceClass1.Name, metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Failed to delete device class: %v", err)
	}

	time.Sleep(1 * time.Second)
	_, ok = cache.Get("gpu-class")
	if ok {
		t.Error("Expected 'gpu-class' not found after deletion")
	}
}
