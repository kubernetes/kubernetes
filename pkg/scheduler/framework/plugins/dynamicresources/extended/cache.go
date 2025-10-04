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
	"sync"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/watch"
	klog "k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
)

// ExtendedResourceCache maintains a global cache of extended resource to device class mappings
// that can be accessed by event handlers without requiring a scheduling cycle.
type ExtendedResourceCache struct {
	mutex sync.RWMutex
	// mapping maps extended resource name to device class name
	mapping    map[v1.ResourceName]string
	draManager fwk.SharedDRAManager
	logger     klog.Logger
}

// NewExtendedResourceCache creates a new ExtendedResourceCache instance.
func NewExtendedResourceCache(draManager fwk.SharedDRAManager, logger klog.Logger) *ExtendedResourceCache {
	cache := &ExtendedResourceCache{
		mapping:    make(map[v1.ResourceName]string),
		draManager: draManager,
		logger:     logger,
	}
	logger.V(4).Info("Created extended resource cache")
	return cache
}

// GetDeviceClass returns the device class name for the given extended resource name.
// Returns the device class name and true if found, empty string and false otherwise.
func (c *ExtendedResourceCache) GetDeviceClass(resourceName v1.ResourceName) (string, bool) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	deviceClass, exists := c.mapping[resourceName]
	return deviceClass, exists
}

// GetAllMappings returns a copy of all current mappings.
func (c *ExtendedResourceCache) GetAllMappings() map[v1.ResourceName]string {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	result := make(map[v1.ResourceName]string, len(c.mapping))
	for k, v := range c.mapping {
		result[k] = v
	}
	return result
}

// OnDeviceClassEvent updates the cache when device class informer events occur.
func (c *ExtendedResourceCache) OnDeviceClassEvent(eventType watch.EventType, deviceClass *resourceapi.DeviceClass) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	switch eventType {
	case watch.Added, watch.Modified:
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

	case watch.Deleted:
		// Remove from mapping
		for resourceName, className := range c.mapping {
			if className == deviceClass.Name {
				delete(c.mapping, resourceName)
				c.logger.V(5).Info("Removed extended resource from cache",
					"extendedResource", resourceName,
					"deviceClass", deviceClass.Name)
			}
		}
	}
}

// refresh updates the cache with the current state from the DRA manager.
func (c *ExtendedResourceCache) refresh() {
	if c.draManager == nil {
		return
	}

	mapping, err := DeviceClassMapping(c.draManager)
	if err != nil {
		c.logger.Error(err, "Failed to refresh extended resource cache")
		return
	}

	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.mapping = mapping
	c.logger.V(4).Info("Refreshed extended resource cache", "mappingCount", len(mapping))
}

// Refresh manually triggers a cache refresh. This is useful for initialization
// or when the cache needs to be synchronized with the current state.
func (c *ExtendedResourceCache) Refresh() {
	c.refresh()
}
