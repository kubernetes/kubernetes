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
	"sync"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/informers"
	cache "k8s.io/client-go/tools/cache"
	klog "k8s.io/klog/v2"
)

// DeviceClassMapping maintains a global cache of extended resource and device class mapping backed by informer.
type DeviceClassMapping struct {
	informer cache.SharedIndexInformer
	logger   klog.Logger
	mutex    sync.RWMutex
	// mapping maps device class name to extended resource name
	mapping map[string]string
}

// NewDeviceClassMapping creates a new DeviceClassMapping.
func NewDeviceClassMapping(i informers.SharedInformerFactory) *DeviceClassMapping {
	if i == nil {
		return nil
	}
	d := &DeviceClassMapping{
		mapping:  make(map[string]string),
		informer: i.Resource().V1().DeviceClasses().Informer(),
		logger:   klog.FromContext(context.Background()),
	}
	items, err := i.Resource().V1().DeviceClasses().Lister().List(labels.Everything())
	if err != nil {
		d.logger.Error(err, "Failed to list device classes")
		return nil
	}

	for _, item := range items {
		if item.Spec.ExtendedResourceName != nil {
			d.mapping[item.Name] = *item.Spec.ExtendedResourceName
		}
	}
	if _, err := d.informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			d.addDeviceClass(obj)
		},
		UpdateFunc: func(old, cur interface{}) {
			d.addDeviceClass(cur)
		},
		DeleteFunc: func(obj interface{}) {
			d.deleteDeviceClass(obj)
		},
	}); err != nil {
		d.logger.Error(err, "Failed to add event handler for device classes")
		return nil
	}
	return d
}

func (d *DeviceClassMapping) act(obj interface{}, isDelete bool) {
	if d == nil {
		return
	}
	d.mutex.Lock()
	defer d.mutex.Unlock()

	if deviceClass, ok := obj.(*resourceapi.DeviceClass); ok {
		if isDelete {
			delete(d.mapping, deviceClass.Name)
			d.logger.V(5).Info("Removed device class", "deviceClass", deviceClass.Name)
			return
		}
		if deviceClass.Spec.ExtendedResourceName == nil {
			return
		}
		d.mapping[deviceClass.Name] = *deviceClass.Spec.ExtendedResourceName
		d.logger.V(5).Info("Updated device class mapping", "deviceClass", deviceClass.Name, "extendedResource", *deviceClass.Spec.ExtendedResourceName)
	}
}

func (d *DeviceClassMapping) addDeviceClass(obj interface{}) {
	d.act(obj, false)
}

func (d *DeviceClassMapping) deleteDeviceClass(obj interface{}) {
	d.act(obj, true)
}

// Get returns the extended resource name for given device class name and true if found
// Returns empty string and false otherwise.
func (d *DeviceClassMapping) Get(name string) (string, bool) {
	if d == nil {
		return "", false
	}
	d.mutex.RLock()
	defer d.mutex.RUnlock()
	if !d.informer.HasSynced() {
		return "", false
	}
	extendedResourceName, ok := d.mapping[name]
	return extendedResourceName, ok
}
