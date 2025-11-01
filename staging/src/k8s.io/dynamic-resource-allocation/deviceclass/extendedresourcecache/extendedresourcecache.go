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
	"fmt"
	"sync"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/cache"
	klog "k8s.io/klog/v2"
)

// ExtendedResourceCache maintains a global cache of extended resource to device class mappings,
// based on informer events. For that it implements the cache.ResourceEventHandler interface.
type ExtendedResourceCache struct {
	logger   klog.Logger
	handlers []cache.ResourceEventHandler

	mutex sync.RWMutex
	// mapping maps extended resource name to device class name
	mapping map[v1.ResourceName]string
	// classMapping maps device class name to extended resource name
	classMapping map[string]string
}

var _ cache.ResourceEventHandler = &ExtendedResourceCache{}

// NewExtendedResourceCache creates a new ExtendedResourceCache instance. The caller
// is responsible for registering the instance as a handler of DeviceClass events.
//
// Additional event handlers may be registered here or via AddEventHandler.
func NewExtendedResourceCache(logger klog.Logger, handlers ...cache.ResourceEventHandler) *ExtendedResourceCache {
	cache := &ExtendedResourceCache{
		logger:       logger,
		handlers:     handlers,
		mapping:      make(map[v1.ResourceName]string),
		classMapping: make(map[string]string),
	}

	return cache
}

// AddEventHandler adds an event handler which gets called after the cache
// has processed some incoming event. More than one additional event handler
// may be added. They will be called in the order in which they were registered.
// GetDeviceClass may be called from those event handlers.
//
// Not thread-safe, must be called *before* adding the cache itself to an
// informer cache.
func (c *ExtendedResourceCache) AddEventHandler(handler cache.ResourceEventHandler) {
	c.handlers = append(c.handlers, handler)
}

// GetDeviceClass returns the device class name for the given extended resource name.
// Returns empty string if the resource name is not found in the cache.
//
// This (and only this) method may be called on a nil ExtendedResourceCache. The nil
// instance always returns the empty string.
func (c *ExtendedResourceCache) GetDeviceClass(resourceName v1.ResourceName) string {
	if c == nil {
		return ""
	}
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return c.mapping[resourceName]
}

func (c *ExtendedResourceCache) GetExtendedResource(className string) string {
	if c == nil {
		return ""
	}
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return c.classMapping[className]
}

// OnAdd handles the addition of a new device class.
func (c *ExtendedResourceCache) OnAdd(obj interface{}, isInInitialList bool) {
	deviceClass, ok := obj.(*resourceapi.DeviceClass)
	if !ok {
		utilruntime.HandleErrorWithLogger(c.logger, nil, "Expected DeviceClass", "actual", fmt.Sprintf("%T", obj))
		return
	}
	c.updateMapping(deviceClass, nil)
	c.updateClassMapping(deviceClass)

	for _, handler := range c.handlers {
		handler.OnAdd(obj, isInInitialList)
	}
}

// OnUpdate handles updates to an existing device class.
func (c *ExtendedResourceCache) OnUpdate(oldObj, newObj interface{}) {
	deviceClass, ok := newObj.(*resourceapi.DeviceClass)
	if !ok {
		utilruntime.HandleErrorWithLogger(c.logger, nil, "Expected DeviceClass", "actual", fmt.Sprintf("%T", newObj))
		return
	}
	oldDeviceClass, ok := oldObj.(*resourceapi.DeviceClass)
	if !ok {
		utilruntime.HandleErrorWithLogger(c.logger, nil, "Expected DeviceClass", "actual", fmt.Sprintf("%T", oldObj))
		return
	}
	c.updateMapping(deviceClass, oldDeviceClass)
	c.updateClassMapping(deviceClass)

	for _, handler := range c.handlers {
		handler.OnUpdate(oldObj, newObj)
	}
}

// OnDelete handles deletion of a device class.
func (c *ExtendedResourceCache) OnDelete(obj interface{}) {
	if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		obj = tombstone.Obj
	}
	deviceClass, ok := obj.(*resourceapi.DeviceClass)
	if !ok {
		utilruntime.HandleErrorWithLogger(c.logger, nil, "Expected DeviceClass", "actual", fmt.Sprintf("%T", obj))
		return
	}
	c.removeMapping(deviceClass)
	c.removeClassMapping(deviceClass)

	for _, handler := range c.handlers {
		handler.OnDelete(obj)
	}
}

// updateMapping updates the cache with the device class mapping.
// It first removes any existing mappings for this device class to handle
// ExtendedResourceName changes, then adds the new mappings.
func (c *ExtendedResourceCache) updateMapping(newDeviceClass, oldDeviceClass *resourceapi.DeviceClass) {
	if newDeviceClass == nil {
		return
	}
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Remove old mappings first to handle ExtendedResourceName changes
	if oldDeviceClass != nil {
		delete(c.mapping, v1.ResourceName(resourceapi.ResourceDeviceClassPrefix+oldDeviceClass.Name))
		if oldDeviceClass.Spec.ExtendedResourceName != nil {
			delete(c.mapping, v1.ResourceName(*oldDeviceClass.Spec.ExtendedResourceName))
		}
	}

	// Add new mappings
	if newDeviceClass.Spec.ExtendedResourceName != nil {
		c.mapping[v1.ResourceName(*newDeviceClass.Spec.ExtendedResourceName)] = newDeviceClass.Name
		c.logger.V(5).Info("Updated extended resource cache for explicit mapping",
			"extendedResource", *newDeviceClass.Spec.ExtendedResourceName,
			"deviceClass", newDeviceClass.Name)
	}
	// Always add the default mapping
	defaultResourceName := v1.ResourceName(resourceapi.ResourceDeviceClassPrefix + newDeviceClass.Name)
	c.mapping[defaultResourceName] = newDeviceClass.Name
	c.logger.V(5).Info("Updated extended resource cache for default mapping",
		"extendedResource", defaultResourceName,
		"deviceClass", newDeviceClass.Name)
}

// updateClassMapping updates the cache with the device class mapping.
func (c *ExtendedResourceCache) updateClassMapping(deviceClass *resourceapi.DeviceClass) {
	if deviceClass == nil {
		return
	}
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if deviceClass.Spec.ExtendedResourceName == nil {
		delete(c.classMapping, deviceClass.Name)
		return
	}

	c.classMapping[deviceClass.Name] = *deviceClass.Spec.ExtendedResourceName
	c.logger.V(5).Info("Updated device class mapping", "deviceClass", deviceClass.Name, "extendedResource", *deviceClass.Spec.ExtendedResourceName)
}

// removeMapping removes the device class mapping from the cache.
// It searches for all mappings to the given device class name and removes them,
// because the ExtendedResourceName in the deviceClass object may be stale.
func (c *ExtendedResourceCache) removeMapping(deviceClass *resourceapi.DeviceClass) {
	if deviceClass == nil {
		return
	}
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Remove the default mapping
	delete(c.mapping, v1.ResourceName(resourceapi.ResourceDeviceClassPrefix+deviceClass.Name))
	// Remove the explicit mapping
	if deviceClass.Spec.ExtendedResourceName != nil {
		delete(c.mapping, v1.ResourceName(*deviceClass.Spec.ExtendedResourceName))
	}
	c.logger.V(5).Info("Removed extended resource from cache",
		"deviceClass", deviceClass.Name)
}

// removeClassMapping removes the device class mapping from the cache.
func (c *ExtendedResourceCache) removeClassMapping(deviceClass *resourceapi.DeviceClass) {
	if deviceClass == nil {
		return
	}
	c.mutex.Lock()
	defer c.mutex.Unlock()

	delete(c.classMapping, deviceClass.Name)
	c.logger.V(5).Info("Removed device class", "deviceClass", deviceClass.Name)
}
