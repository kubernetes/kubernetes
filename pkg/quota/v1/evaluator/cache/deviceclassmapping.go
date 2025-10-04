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
	"sync"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/informers"
	klog "k8s.io/klog/v2"
)

// DeviceClassMapping maintains a global cache of extended resource and device class mapping backed by informer.
type DeviceClassMapping struct {
	mutex sync.RWMutex
	// mapping maps device class name to extended resource name
	mapping  map[string]string
	informer informers.SharedInformerFactory
}

// NewDeviceClassMapping creates a new DeviceClassMapping.
func NewDeviceClassMapping(i informers.SharedInformerFactory) *DeviceClassMapping {
	if i == nil {
		return nil
	}
	cache := &DeviceClassMapping{
		mapping:  make(map[string]string),
		informer: i,
	}
	if _, err := i.Resource().V1().DeviceClasses().Informer().AddEventHandler(cache); err != nil {
		klog.Error(err, "Failed to add event handler for device classes")
		return nil
	}
	klog.V(4).Info("Created device class mapping cache")
	return cache
}

func (d *DeviceClassMapping) OnAdd(obj interface{}, isInInitialList bool) {
	if deviceClass, ok := obj.(*resourceapi.DeviceClass); ok {
		d.onDeviceClassEvent(watch.Added, deviceClass)
	}
}

func (d *DeviceClassMapping) OnUpdate(oldObj, newObj interface{}) {
	if deviceClass, ok := newObj.(*resourceapi.DeviceClass); ok {
		d.onDeviceClassEvent(watch.Modified, deviceClass)
	}
}

func (d *DeviceClassMapping) OnDelete(obj interface{}) {
	if deviceClass, ok := obj.(*resourceapi.DeviceClass); ok {
		d.onDeviceClassEvent(watch.Deleted, deviceClass)
	}
}

// Get returns the extended resource name for given device class name and true if found
// Returns empty string and false otherwise.
func (d *DeviceClassMapping) Get(name string) (string, bool) {
	if d == nil {
		return "", false
	}
	d.mutex.RLock()
	defer d.mutex.RUnlock()
	extendedResourceName, exists := d.mapping[name]
	return extendedResourceName, exists
}

// OnDeviceClassEvent updates the cache when device class informer events occur.
func (d *DeviceClassMapping) onDeviceClassEvent(eventType watch.EventType, deviceClass *resourceapi.DeviceClass) {
	if d == nil {
		return
	}
	d.mutex.Lock()
	defer d.mutex.Unlock()

	switch eventType {
	case watch.Added, watch.Modified:
		if deviceClass.Spec.ExtendedResourceName == nil {
			return
		}
		d.mapping[deviceClass.Name] = *deviceClass.Spec.ExtendedResourceName
		klog.V(5).Info("Updated device class mapping", "deviceClass", deviceClass.Name, "extendedResource", *deviceClass.Spec.ExtendedResourceName)

	case watch.Deleted:
		delete(d.mapping, deviceClass.Name)
		klog.V(5).Info("Removed device class", "deviceClass", deviceClass.Name)
	}
}
