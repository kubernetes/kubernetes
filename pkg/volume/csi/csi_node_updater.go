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

package csi

import (
	"fmt"
	"sync"
	"time"

	v1 "k8s.io/api/storage/v1"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

// csiNodeUpdater is a struct that watches for changes to CSIDriver objects and manages
// the lifecycle of goroutines that periodically update CSINodeDriver.Allocatable
// information based on the NodeAllocatableUpdatePeriodSeconds setting.
type csiNodeUpdater struct {
	// Lister for CSIDriver objects
	driverLister storagelisters.CSIDriverLister

	// Informer for CSIDriver objects
	driverInformer cache.SharedIndexInformer

	// Map of driver names to stop channels for update goroutines
	driverUpdaters sync.Map

	// Ensures the updater is only started once
	once sync.Once
}

// NewCSINodeUpdater creates a new csiNodeUpdater
func NewCSINodeUpdater(
	driverLister storagelisters.CSIDriverLister, driverInformer cache.SharedIndexInformer) (*csiNodeUpdater, error) {
	if driverLister == nil {
		return nil, fmt.Errorf("driverLister must not be nil")
	}
	if driverInformer == nil {
		return nil, fmt.Errorf("driverInformer must not be nil")
	}
	return &csiNodeUpdater{
		driverLister:   driverLister,
		driverInformer: driverInformer,
		driverUpdaters: sync.Map{},
	}, nil
}

// Run starts the CSINodeUpdater
func (u *csiNodeUpdater) Run() {
	u.once.Do(func() {
		_, err := u.driverInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
			AddFunc:    u.onDriverAdd,
			UpdateFunc: u.onDriverUpdate,
			DeleteFunc: u.onDriverDelete,
		})
		if err != nil {
			klog.ErrorS(err, "Failed to add event handler for CSI driver informer")
			return
		}

		klog.InfoS("CSINodeUpdater initialized successfully")
	})
}

// onDriverAdd handles the addition of a new CSIDriver object
func (u *csiNodeUpdater) onDriverAdd(obj interface{}) {
	driver, ok := obj.(*v1.CSIDriver)
	if !ok {
		return
	}
	u.startOrReconfigureDriverUpdater(driver)
}

// onDriverUpdate handles updates to CSIDriver objects
func (u *csiNodeUpdater) onDriverUpdate(oldObj, newObj interface{}) {
	oldDriver, ok := oldObj.(*v1.CSIDriver)
	if !ok {
		return
	}
	newDriver, ok := newObj.(*v1.CSIDriver)
	if !ok {
		return
	}

	// Check if the relevant field changed
	oldPeriod := getNodeAllocatableUpdatePeriod(oldDriver)
	newPeriod := getNodeAllocatableUpdatePeriod(newDriver)

	if oldPeriod != newPeriod {
		klog.InfoS("NodeAllocatableUpdatePeriodSeconds changed", "driver", newDriver.Name, "oldPeriod", oldPeriod, "newPeriod", newPeriod)
		u.startOrReconfigureDriverUpdater(newDriver)
	}
}

// onDriverDelete handles deletion of CSIDriver objects
func (u *csiNodeUpdater) onDriverDelete(obj interface{}) {
	driver, ok := obj.(*v1.CSIDriver)
	if !ok {
		return
	}
	u.unregisterDriver(driver.Name)
}

// unregisterDriver stops updates for a driver
func (u *csiNodeUpdater) unregisterDriver(driverName string) {
	klog.V(4).InfoS("UnregisterDriver called", "driver", driverName)

	if stopCh, exists := u.driverUpdaters.Load(driverName); exists {
		close(stopCh.(chan struct{}))
		u.driverUpdaters.Delete(driverName)
		klog.V(4).InfoS("Stopped updater for driver", "driver", driverName)
	}
}

// startOrReconfigureDriverUpdater starts or reconfigures the updater for a driver
func (u *csiNodeUpdater) startOrReconfigureDriverUpdater(driver *v1.CSIDriver) {
	period := getNodeAllocatableUpdatePeriod(driver)

	// If the period is 0, disable updates for this driver
	if period == 0 {
		klog.InfoS("NodeAllocatableUpdatePeriodSeconds is 0, disabling updates", "driver", driver.Name)
		u.unregisterDriver(driver.Name)
		return
	}

	// Otherwise, stop any existing updater and start a new one
	if existingStopCh, exists := u.driverUpdaters.Load(driver.Name); exists {
		close(existingStopCh.(chan struct{}))
	}

	stopCh := make(chan struct{})
	u.driverUpdaters.Store(driver.Name, stopCh)

	go u.runPeriodicUpdate(driver.Name, period, stopCh)
	klog.V(4).InfoS("Started/reconfigured updater for driver", "driver", driver.Name, "period", period)
}

// runPeriodicUpdate runs the periodic update loop for a driver
func (u *csiNodeUpdater) runPeriodicUpdate(driverName string, period time.Duration, stopCh <-chan struct{}) {
	ticker := time.NewTicker(period)
	defer ticker.Stop()

	klog.V(4).InfoS("Starting CSINode periodic updates", "driver", driverName, "period", period)

	for {
		select {
		case <-ticker.C:
			if err := updateCSIDriver(driverName); err != nil {
				klog.ErrorS(err, "Failed to update CSIDriver", "driver", driverName)
			}
		case <-stopCh:
			klog.V(4).InfoS("Stopping periodic updates", "driver", driverName)
			return
		}
	}
}
