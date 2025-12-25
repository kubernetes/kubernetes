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

package extendedresourcecache

import (
	"context"
	"testing"
	"testing/synctest"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2/ktesting"
	_ "k8s.io/klog/v2/ktesting/init" // Add command line flags.
	"k8s.io/utils/ptr"
)

type deviceClassResolver interface {
	GetDeviceClass(resourceName v1.ResourceName) *resourceapi.DeviceClass
}

func TestNil(t *testing.T) {
	var cache *ExtendedResourceCache
	var resolver deviceClassResolver = cache
	if class := resolver.GetDeviceClass("example.com/gpu"); class != nil {
		t.Errorf("Expected the nil class from a nil instance, got instead: %q", class.Name)
	}
}

func TestHandlers(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	var numAdd, numUpdate, numDelete int

	resourceName := v1.ResourceName("example.com/gpu")
	class := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-class",
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: (*string)(&resourceName),
		},
	}
	updatedClass := class.DeepCopy()
	updatedClass.Spec.ExtendedResourceName = nil

	firstHandler := &cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			if obj != class {
				t.Errorf("first handler expected added object %v, got %v", class, obj)
			}
			numAdd++
			if numAdd != 1 {
				t.Errorf("first handler expected Add to be called first once, actual add #%d", numAdd)
			}
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			if oldObj != class {
				t.Errorf("first handler expected old object %v, got %v", class, oldObj)
			}
			if newObj != updatedClass {
				t.Errorf("first handler expected new object %v, got %v", class, newObj)
			}
			numUpdate++
			if numUpdate != 1 {
				t.Errorf("first handler expected Update to be called first once, actual update #%d", numUpdate)
			}
		},
		DeleteFunc: func(obj interface{}) {
			if obj != updatedClass {
				t.Errorf("first handler expected deleted object %v, got %v", class, obj)
			}
			numDelete++
			if numDelete != 1 {
				t.Errorf("first handler expected Delete to be called first once, actual delete #%d", numDelete)
			}
		},
	}
	erCache := NewExtendedResourceCache(logger, firstHandler)
	secondHandler := &cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			if obj != class {
				t.Errorf("second handler expected added object %v, got %v", class, obj)
			}
			numAdd++
			if numAdd != 2 {
				t.Errorf("second handler expected Add to be called last once, actual add #%d", numAdd)
			}
			if deviceClass := erCache.GetDeviceClass(resourceName); deviceClass == nil || deviceClass.Name != class.Name {
				t.Errorf("expected %q, got %q", class.Name, deviceClass)
			}
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			if oldObj != class {
				t.Errorf("second handler expected old object %v, got %v", class, oldObj)
			}
			if newObj != updatedClass {
				t.Errorf("second handler expected new object %v, got %v", class, newObj)
			}
			numUpdate++
			if numUpdate != 2 {
				t.Errorf("second handler expected Update to be called last once, actual update #%d", numUpdate)
			}
			if class := erCache.GetDeviceClass(resourceName); class != nil {
				t.Errorf("expected %q, got %v", "", class)
			}
		},
		DeleteFunc: func(obj interface{}) {
			if obj != updatedClass {
				t.Errorf("second handler expected deleted object %v, got %v", class, obj)
			}
			numDelete++
			if numDelete != 2 {
				t.Errorf("second handler expected Delete to be called last once, actual delete #%d", numDelete)
			}
			if class := erCache.GetDeviceClass(resourceName); class != nil {
				t.Errorf("expected %q, got %q", "", class)
			}
		},
	}
	erCache.AddEventHandler(secondHandler)

	erCache.OnAdd(class, false)
	erCache.OnUpdate(class, updatedClass)
	erCache.OnDelete(updatedClass)
}

func TestExtendedResourceCache(t *testing.T) { synctest.Test(t, testExtendedResourceCache) }
func testExtendedResourceCache(t *testing.T) {
	tCtx, client, cache := setup(t)

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
	synctest.Wait()

	// Verify explicit mapping
	deviceClass := cache.GetDeviceClass("example.com/gpu")
	if deviceClass == nil || deviceClass.Name != "gpu-class" {
		t.Errorf("Expected to find device class 'gpu-class' for 'example.com/gpu', got %v", deviceClass)
	}

	// Verify default mapping
	defaultResourceName := v1.ResourceName("deviceclass.resource.kubernetes.io/fpga-class")
	deviceClass = cache.GetDeviceClass(defaultResourceName)
	if deviceClass == nil || deviceClass.Name != "fpga-class" {
		t.Errorf("Expected to find device class 'fpga-class' for '%s', got %v", defaultResourceName, deviceClass)
	}

	// Verify both device classes have default mappings
	deviceClass = cache.GetDeviceClass("deviceclass.resource.kubernetes.io/gpu-class")
	if deviceClass == nil || deviceClass.Name != "gpu-class" {
		t.Error("Expected default mapping for gpu-class")
	}

	// deviceClass3 is older than deviceClass1, hence it won't replace deviceClass1
	_, err = client.ResourceV1().DeviceClasses().Create(tCtx, deviceClass3, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create device class: %v", err)
	}
	synctest.Wait()

	// should keep deviceClass1, since it is newer than deviceClass3
	deviceClass = cache.GetDeviceClass("example.com/gpu")
	if deviceClass == nil || deviceClass.Name != "gpu-class" {
		t.Errorf("Expected to find device class 'gpu-class' for 'example.com/gpu', got %v", deviceClass)
	}

	// deviceClass4 is newer than deviceClass1, hence it will replace deviceClass1
	_, err = client.ResourceV1().DeviceClasses().Create(tCtx, deviceClass4, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create device class: %v", err)
	}
	synctest.Wait()

	// deviceClass4 replaces deviceClass1, since it is newer with the same example.com/gpu extended resource name
	deviceClass = cache.GetDeviceClass("example.com/gpu")
	if deviceClass == nil || deviceClass.Name != "gpu-class-4" {
		t.Errorf("Expected to find device class 'gpu-class' for 'example.com/gpu', got %v", deviceClass)
	}

	// deviceClass0 is created at the same time as deviceClass4, but its name is alphabetically ordered earlier,
	//  hence it will replace deviceClass4
	_, err = client.ResourceV1().DeviceClasses().Create(tCtx, deviceClass0, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create device class: %v", err)
	}
	synctest.Wait()

	// deviceClass0 replaces deviceClass4, it is created at the same time as deviceClass4, but its name is
	// alphabetically ordered earlier
	deviceClass = cache.GetDeviceClass("example.com/gpu")
	if deviceClass == nil || deviceClass.Name != "gpu-class-0" {
		t.Errorf("Expected to find device class 'gpu-class' for 'example.com/gpu', got %v", deviceClass)
	}

	// Test modifying a device class
	deviceClass0Modified := deviceClass0.DeepCopy()
	deviceClass0Modified.Spec.ExtendedResourceName = ptr.To("test.com/gpu")
	_, err = client.ResourceV1().DeviceClasses().Update(tCtx, deviceClass0Modified, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update device class: %v", err)
	}
	synctest.Wait()

	// Should have the new mapping
	deviceClass = cache.GetDeviceClass("test.com/gpu")
	if deviceClass == nil || deviceClass.Name != "gpu-class-0" {
		t.Errorf("Expected to find device class 'gpu-class-0' for 'test.com/gpu' after modification, got %v", deviceClass)
	}
	// Should not have the old mapping for example.com/gpu
	if cache.GetDeviceClass("example.com/gpu") != nil {
		t.Errorf("Expected 'example.com/gpu' to be removed after modification, got %s", cache.GetDeviceClass("example.com/gpu"))
	}

	// Test deleting a device class
	err = client.ResourceV1().DeviceClasses().Delete(tCtx, deviceClass0.Name, metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Failed to delete device class: %v", err)
	}
	synctest.Wait()

	deviceClass = cache.GetDeviceClass("test.com/gpu")
	if deviceClass != nil {
		t.Errorf("Expected 'test.com/gpu' to be removed after deleting device class, got %s", deviceClass)
	}
	// Verify the default mapping is removed
	if cache.GetDeviceClass("deviceclass.resource.kubernetes.io/gpu-class-0") != nil {
		t.Errorf("Expected 'deviceclass.resource.kubernetes.io/gpu-class-0' to be removed after deleting device class, got %s", cache.GetDeviceClass("deviceclass.resource.kubernetes.io/gpu-class"))
	}
}

func TestDeviceClassMapping(t *testing.T) { synctest.Test(t, testDeviceClassMapping) }
func testDeviceClassMapping(t *testing.T) {
	tCtx, client, cache := setup(t)

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

	// Wait for background goroutines to handle the new classes.
	synctest.Wait()
	name := cache.GetExtendedResource("gpu-class")
	if name != "example.com/gpu" {
		t.Errorf("Expected to find device class 'gpu-class', got %s", name)
	}
	name = cache.GetExtendedResource("tpu-class")
	if name != "" {
		t.Errorf("Expected device class 'tpu-class' not found")
	}

	// Test updating device classes
	deviceClass1Modified := deviceClass1.DeepCopy()
	deviceClass1Modified.Spec.ExtendedResourceName = ptr.To("my.com/gpu")
	_, err = client.ResourceV1().DeviceClasses().Update(tCtx, deviceClass1Modified, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update device class: %v", err)
	}

	synctest.Wait()
	name = cache.GetExtendedResource("gpu-class")
	if name != "my.com/gpu" {
		t.Errorf("Expected to find device class 'gpu-class' with  'my.com/gpu' after modification, got %s", name)
	}

	// Test deleting device classes
	err = client.ResourceV1().DeviceClasses().Delete(tCtx, deviceClass1.Name, metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Failed to delete device class: %v", err)
	}

	synctest.Wait()
	name = cache.GetExtendedResource("gpu-class")
	if name != "" {
		t.Error("Expected 'gpu-class' not found after deletion")
	}
}

func setup(t *testing.T) (context.Context, *fake.Clientset, *ExtendedResourceCache) {
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	t.Cleanup(cancel)

	client := fake.NewClientset()
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	ec := NewExtendedResourceCache(logger)
	handle, err := informerFactory.Resource().V1().DeviceClasses().Informer().AddEventHandler(ec)
	if err != nil {
		t.Fatalf("failed to add device class informer event handler: %v", err)
	}
	informerFactory.Start(ctx.Done())
	t.Cleanup(func() {
		// Need to cancel before waiting for the shutdown.
		cancel()
		// Now we can wait for all goroutines to stop.
		informerFactory.Shutdown()
	})
	informerFactory.WaitForCacheSync(ctx.Done())
	cache.WaitForNamedCacheSyncWithContext(ctx, handle.HasSynced)

	// fake.Clientset suffers from a race condition related to informers:
	// it does not implement resource version support in its Watch
	// implementation and instead assumes that watches are set up
	// before further changes are made.
	//
	// If a test waits for caches to be synced and then immediately
	// adds an object, that new object will never be seen by event handlers
	// if the race goes wrong and the Watch call hadn't completed yet
	// (can be triggered by adding a sleep before https://github.com/kubernetes/kubernetes/blob/b53b9fb5573323484af9a19cf3f5bfe80760abba/staging/src/k8s.io/client-go/tools/cache/reflector.go#L431).
	//
	// To work around that, we wait here for the goroutines which
	// are involved in setting up the watch *before* creating
	// DeviceClasses.
	synctest.Wait()

	return ctx, client, ec
}
