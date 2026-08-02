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
	"cmp"
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
	// explicitResourceName2classes maps an explicit extended resource name to
	// the set of device classes which declare it. Several classes may declare
	// the same name, for example while migrating from one class to another;
	// the winner is the "best" class according to betterDeviceClass.
	explicitResourceName2classes map[string]map[string]*resourceapi.DeviceClass
	// resourceName2class maps extended resource name to device class. For
	// explicit names it holds the current winner of
	// explicitResourceName2classes, for implicit
	// deviceclass.resource.kubernetes.io/<class name> names it holds the
	// class itself.
	resourceName2class map[v1.ResourceName]*resourceapi.DeviceClass
	// class2ResourceName maps device class name to extended resource name
	class2ResourceName map[string]string
}

var _ cache.ResourceEventHandler = &ExtendedResourceCache{}

// NewExtendedResourceCache creates a new ExtendedResourceCache instance. The caller
// is responsible for registering the instance as a handler of DeviceClass events.
//
// Additional event handlers may be registered here or via AddEventHandler.
func NewExtendedResourceCache(logger klog.Logger, handlers ...cache.ResourceEventHandler) *ExtendedResourceCache {
	cache := &ExtendedResourceCache{
		logger:                       logger,
		handlers:                     handlers,
		explicitResourceName2classes: make(map[string]map[string]*resourceapi.DeviceClass),
		resourceName2class:           make(map[v1.ResourceName]*resourceapi.DeviceClass),
		class2ResourceName:           make(map[string]string),
	}

	return cache
}

// AddEventHandler adds an event handler which gets called after the cache
// has processed some incoming event. More than one additional event handler
// may be added. They will be called in the order in which they were registered.
// GetDeviceClass may be called from those event handlers.
//
// Not thread-safe, must be called *before* adding the cache itself to an
// informer.
func (c *ExtendedResourceCache) AddEventHandler(handler cache.ResourceEventHandler) {
	c.handlers = append(c.handlers, handler)
}

// GetDeviceClass returns the device class for the given extended resource name.
// Returns nil if the resource name is not found in the cache.
//
// This (and only this) method may be called on a nil ExtendedResourceCache. The nil
// instance always returns nil.
func (c *ExtendedResourceCache) GetDeviceClass(resourceName v1.ResourceName) *resourceapi.DeviceClass {
	if c == nil {
		return nil
	}
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return c.resourceName2class[resourceName]
}

// GetExtendedResource returns the extended resource name for the given device class name.
// Returns empty string if the class name is not found in the cache.
func (c *ExtendedResourceCache) GetExtendedResource(className string) string {
	if c == nil {
		return ""
	}
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	return c.class2ResourceName[className]
}

// OnAdd handles the addition of a new device class.
func (c *ExtendedResourceCache) OnAdd(obj interface{}, isInInitialList bool) {
	deviceClass, ok := obj.(*resourceapi.DeviceClass)
	if !ok {
		utilruntime.HandleErrorWithLogger(c.logger, nil, "Expected DeviceClass", "actual", fmt.Sprintf("%T", obj))
		return
	}
	c.logger.V(5).Info("DeviceClass added", "deviceClass", klog.KObj(deviceClass))
	c.updateResourceName2class(deviceClass, nil)
	c.updateClass2ResourceName(deviceClass)

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
	c.logger.V(5).Info("DeviceClass updated", "deviceClass", klog.KObj(deviceClass))
	c.updateResourceName2class(deviceClass, oldDeviceClass)
	c.updateClass2ResourceName(deviceClass)

	for _, handler := range c.handlers {
		handler.OnUpdate(oldObj, newObj)
	}
}

// OnDelete handles deletion of a device class.
func (c *ExtendedResourceCache) OnDelete(obj interface{}) {
	className := ""
	if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		if deviceClass, ok := tombstone.Obj.(*resourceapi.DeviceClass); ok {
			className = deviceClass.Name
		} else {
			// DeltaFIFO.Replace can emit a key-only tombstone with a nil
			// Obj when the key is no longer available from knownObjects.
			// DeviceClass is cluster-scoped and all mappings are keyed by
			// class name, so the key alone is enough to remove them all.
			className = tombstone.Key
		}
	} else if deviceClass, ok := obj.(*resourceapi.DeviceClass); ok {
		className = deviceClass.Name
	} else {
		utilruntime.HandleErrorWithLogger(c.logger, nil, "Expected DeviceClass", "actual", fmt.Sprintf("%T", obj))
		return
	}
	c.logger.V(5).Info("DeviceClass deleted", "deviceClass", className)
	c.removeResourceName2class(className)
	c.removeClass2ResourceName(className)

	for _, handler := range c.handlers {
		handler.OnDelete(obj)
	}
}

// betterDeviceClass returns true if class a should win the explicit extended
// resource name over class b: the newer class wins, with the alphabetically
// lower name as tie-breaker. This matches the arbitration documented in
// DeviceClassSpec.ExtendedResourceName.
func betterDeviceClass(a, b *resourceapi.DeviceClass) bool {
	if b == nil {
		return true
	}
	if cmp := cmp.Compare(a.CreationTimestamp.UnixNano(), b.CreationTimestamp.UnixNano()); cmp != 0 {
		return cmp > 0
	}
	return a.Name < b.Name
}

// updateResourceName2class updates the cache with the device class mapping.
// It first removes any existing mappings for this device class to handle
// ExtendedResourceName changes, then adds the new mappings.
//
// Different DeviceClasses may specify the same ExtendedResourceName, for
// example while migrating from one DeviceClass to another. All candidates are
// kept and the winner is derived from them on each event, so that a rename or
// deletion of the current winner promotes the next candidate instead of
// leaving the mapping dangling. The implicit
// deviceclass.resource.kubernetes.io/<class name> mapping is always unique
// (it cannot appear in a different class as ExtendedResourceName, prevented
// by validation), so it is registered independently of the explicit name
// arbitration.
func (c *ExtendedResourceCache) updateResourceName2class(newDeviceClass, oldDeviceClass *resourceapi.DeviceClass) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Drop this class from the candidates of all explicit names, not just
	// the one in the old object, because the old object's
	// ExtendedResourceName may be stale.
	if oldDeviceClass != nil {
		for explicitName, classes := range c.explicitResourceName2classes {
			if _, ok := classes[oldDeviceClass.Name]; !ok {
				continue
			}
			delete(classes, oldDeviceClass.Name)
			if len(classes) == 0 {
				delete(c.explicitResourceName2classes, explicitName)
			}
			c.recomputeWinner(explicitName)
		}
	}

	// Add this class to the candidates of its new explicit name, if any. The
	// freshly updated object replaces a stale cached one.
	if newDeviceClass.Spec.ExtendedResourceName != nil {
		explicitName := *newDeviceClass.Spec.ExtendedResourceName
		classes := c.explicitResourceName2classes[explicitName]
		if classes == nil {
			classes = make(map[string]*resourceapi.DeviceClass)
			c.explicitResourceName2classes[explicitName] = classes
		}
		classes[newDeviceClass.Name] = newDeviceClass
		c.recomputeWinner(explicitName)
	}

	// Always add the default mapping; it is unique to this class and
	// independent of any explicit name arbitration.
	defaultResourceName := v1.ResourceName(resourceapi.ResourceDeviceClassPrefix + newDeviceClass.Name)
	c.resourceName2class[defaultResourceName] = newDeviceClass
	c.logger.V(5).Info("Updated extended resource cache for default mapping",
		"extendedResource", defaultResourceName,
		"deviceClass", newDeviceClass.Name)
}

// recomputeWinner makes explicitName map to the best candidate device class,
// removing the mapping if no candidate remains.
func (c *ExtendedResourceCache) recomputeWinner(explicitName string) {
	classes := c.explicitResourceName2classes[explicitName]
	var winner *resourceapi.DeviceClass
	for _, class := range classes {
		if betterDeviceClass(class, winner) {
			winner = class
		}
	}
	if winner == nil {
		delete(c.resourceName2class, v1.ResourceName(explicitName))
		return
	}
	c.resourceName2class[v1.ResourceName(explicitName)] = winner
	c.logger.V(5).Info("Updated extended resource cache for explicit mapping",
		"extendedResource", explicitName,
		"deviceClass", winner.Name)
}

// updateClass2ResourceName updates the cache with the device class mapping.
func (c *ExtendedResourceCache) updateClass2ResourceName(deviceClass *resourceapi.DeviceClass) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if deviceClass.Spec.ExtendedResourceName == nil {
		delete(c.class2ResourceName, deviceClass.Name)
		return
	}

	c.class2ResourceName[deviceClass.Name] = *deviceClass.Spec.ExtendedResourceName
	c.logger.V(5).Info("Updated device class mapping", "deviceClass", deviceClass.Name, "extendedResource", *deviceClass.Spec.ExtendedResourceName)
}

// removeResourceName2class removes the device class mapping from the cache.
// The class is dropped from the candidates of all explicit names, because the
// ExtendedResourceName in the deleted object may be stale, and the next best
// candidate is promoted, if any.
func (c *ExtendedResourceCache) removeResourceName2class(className string) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// The default mapping is unique to the class and cannot be shared.
	delete(c.resourceName2class, v1.ResourceName(resourceapi.ResourceDeviceClassPrefix+className))

	// Drop the class from the candidates of all explicit names. This only
	// affects mappings owned by the deleted class; the winner of a name
	// claimed by other classes stays in place or is replaced by the next
	// best candidate.
	for explicitName, classes := range c.explicitResourceName2classes {
		if _, ok := classes[className]; !ok {
			continue
		}
		delete(classes, className)
		if len(classes) == 0 {
			delete(c.explicitResourceName2classes, explicitName)
		}
		c.recomputeWinner(explicitName)
	}
	c.logger.V(5).Info("Removed extended resource from cache",
		"deviceClass", className)
}

// removeClass2ResourceName removes the device class mapping from the cache.
func (c *ExtendedResourceCache) removeClass2ResourceName(className string) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	delete(c.class2ResourceName, className)
	c.logger.V(5).Info("Removed device class", "deviceClass", className)
}
