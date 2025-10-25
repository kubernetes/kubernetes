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
	"fmt"
	"sync"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	resourceinformers "k8s.io/client-go/informers/resource/v1"
	"k8s.io/client-go/tools/cache"
	klog "k8s.io/klog/v2"
)

// ExtendedResourceCache maintains a global cache of extended resource to device class mappings
// that can also be accessed by event handlers without requiring an extra scheduling cycle.
type ExtendedResourceCache struct {
	mutex sync.RWMutex
	// mapping maps extended resource name to device class name
	mapping map[v1.ResourceName]string
	logger  klog.Logger
}

// NewExtendedResourceCache creates a new ExtendedResourceCache instance and populates it
// with the current device classes from the informer cache.
func NewExtendedResourceCache(deviceClassInformer resourceinformers.DeviceClassInformer, logger klog.Logger) *ExtendedResourceCache {
	cache := &ExtendedResourceCache{
		mapping: make(map[v1.ResourceName]string),
		logger:  logger,
	}

	// Populate the cache with existing device classes from the informer
	if deviceClassInformer != nil {
		deviceClassLister := deviceClassInformer.Lister()
		classes, err := deviceClassLister.List(labels.Everything())
		if err != nil {
			logger.Error(err, "Failed to populate extended resource cache during initialization")
		} else {
			for _, deviceClass := range classes {
				cache.updateMapping(deviceClass)
			}
			logger.V(4).Info("Created and populated extended resource cache", "mappingCount", len(cache.mapping))
		}
	} else {
		logger.V(4).Info("Created extended resource cache without initial population (deviceClassInformer is nil)")
	}

	return cache
}

// GetDeviceClass returns the device class name for the given extended resource name.
// Returns empty string if the resource name is not found in the cache.
func (c *ExtendedResourceCache) GetDeviceClass(resourceName v1.ResourceName) string {
	if c == nil {
		return ""
	}
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return c.mapping[resourceName]
}

// OnAdd handles the addition of a new device class.
func (c *ExtendedResourceCache) OnAdd(obj interface{}, isInInitialList bool) {
	if deviceClass, ok := obj.(*resourceapi.DeviceClass); ok {
		c.updateMapping(deviceClass)
	}
}

// OnUpdate handles updates to an existing device class.
func (c *ExtendedResourceCache) OnUpdate(oldObj, newObj interface{}) {
	if deviceClass, ok := newObj.(*resourceapi.DeviceClass); ok {
		c.updateMapping(deviceClass)
	}
}

// OnDelete handles deletion of a device class.
func (c *ExtendedResourceCache) OnDelete(obj interface{}) {
	deviceClass, ok := obj.(*resourceapi.DeviceClass)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("couldn't get object from tombstone %#v", obj))
			return
		}
		deviceClass, ok = tombstone.Obj.(*resourceapi.DeviceClass)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("tombstone contained object that is not a DeviceClass %#v", obj))
			return
		}
	}
	c.removeMapping(deviceClass)
}

// updateMapping updates the cache with the device class mapping.
// It first removes any existing mappings for this device class to handle
// ExtendedResourceName changes, then adds the new mappings.
func (c *ExtendedResourceCache) updateMapping(deviceClass *resourceapi.DeviceClass) {
	if c == nil {
		return
	}
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Remove old mappings first to handle ExtendedResourceName changes
	for resourceName, className := range c.mapping {
		if className == deviceClass.Name {
			delete(c.mapping, resourceName)
		}
	}

	// Add new mappings
	if deviceClass.Spec.ExtendedResourceName != nil {
		c.mapping[v1.ResourceName(*deviceClass.Spec.ExtendedResourceName)] = deviceClass.Name
		c.logger.V(5).Info("Updated extended resource cache for explicit mapping",
			"extendedResource", *deviceClass.Spec.ExtendedResourceName,
			"deviceClass", deviceClass.Name)
	}
	// Always add the default mapping
	defaultResourceName := v1.ResourceName(resourceapi.ResourceDeviceClassPrefix + deviceClass.Name)
	c.mapping[defaultResourceName] = deviceClass.Name
	c.logger.V(5).Info("Updated extended resource cache for default mapping",
		"extendedResource", defaultResourceName,
		"deviceClass", deviceClass.Name)
}

// removeMapping removes the device class mapping from the cache.
// It searches for all mappings to the given device class name and removes them,
// because the ExtendedResourceName in the deviceClass object may be stale.
func (c *ExtendedResourceCache) removeMapping(deviceClass *resourceapi.DeviceClass) {
	if c == nil {
		return
	}
	c.mutex.Lock()
	defer c.mutex.Unlock()

	for resourceName, className := range c.mapping {
		if className == deviceClass.Name {
			delete(c.mapping, resourceName)
			c.logger.V(5).Info("Removed extended resource from cache",
				"extendedResource", resourceName,
				"deviceClass", deviceClass.Name)
		}
	}
}
