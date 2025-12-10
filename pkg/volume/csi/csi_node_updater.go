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
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

// csiNodeUpdater watches for changes to CSIDriver objects and manages the lifecycle
// of per-driver goroutines that periodically update CSINodeDriver.Allocatable information
// based on the NodeAllocatableUpdatePeriodSeconds setting.
type csiNodeUpdater struct {
	// Informer for CSIDriver objects
	driverInformer cache.SharedIndexInformer

	// Map of driver names to stop channels for update goroutines
	driverUpdaters sync.Map

	// Ensures the updater is only started once
	once sync.Once
}

// NewCSINodeUpdater creates a new csiNodeUpdater
func NewCSINodeUpdater(driverInformer cache.SharedIndexInformer) (*csiNodeUpdater, error) {
	if driverInformer == nil {
		return nil, fmt.Errorf("driverInformer must not be nil")
	}
	return &csiNodeUpdater{
		driverInformer: driverInformer,
		driverUpdaters: sync.Map{},
	}, nil
}

// Run starts the csiNodeUpdater by registering event handlers.
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
		klog.V(4).InfoS("csiNodeUpdater initialized successfully")
	})
}

// onDriverAdd handles the addition of a new CSIDriver object.
func (u *csiNodeUpdater) onDriverAdd(obj interface{}) {
	driver, ok := obj.(*v1.CSIDriver)
	if !ok {
		return
	}
	klog.V(7).InfoS("onDriverAdd event", "driver", driver.Name)
	u.syncDriverUpdater(driver.Name)
}

// onDriverUpdate handles updates to CSIDriver objects.
func (u *csiNodeUpdater) onDriverUpdate(oldObj, newObj interface{}) {
	oldDriver, ok := oldObj.(*v1.CSIDriver)
	if !ok {
		return
	}
	newDriver, ok := newObj.(*v1.CSIDriver)
	if !ok {
		return
	}

	// Only reconfigure if the NodeAllocatableUpdatePeriodSeconds field is updated.
	oldPeriod := getNodeAllocatableUpdatePeriod(oldDriver)
	newPeriod := getNodeAllocatableUpdatePeriod(newDriver)
	if oldPeriod != newPeriod {
		klog.V(4).InfoS("NodeAllocatableUpdatePeriodSeconds updated", "driver", newDriver.Name, "oldPeriod", oldPeriod, "newPeriod", newPeriod)
		u.syncDriverUpdater(newDriver.Name)
	}
}

// onDriverDelete handles deletion of CSIDriver objects.
func (u *csiNodeUpdater) onDriverDelete(obj interface{}) {
	driver, ok := obj.(*v1.CSIDriver)
	if !ok {
		return
	}
	klog.V(7).InfoS("onDriverDelete event", "driver", driver.Name)
	u.syncDriverUpdater(driver.Name)
}

// syncDriverUpdater re-evaluates whether the periodic updater for a given driver should run.
// It is invoked from informer events (Add/Update/Delete) and from plugin registration/deregistration.
func (u *csiNodeUpdater) syncDriverUpdater(driverName string) {
	// Check if the CSI plugin is installed on this node.
	if !isDriverInstalled(driverName) {
		klog.V(4).InfoS("Driver not installed; stopping csiNodeUpdater", "driver", driverName)
		u.unregisterDriver(driverName)
		return
	}

	// Get the CSIDriver object from the informer cache.
	obj, exists, err := u.driverInformer.GetStore().GetByKey(driverName)
	if err != nil {
		u.unregisterDriver(driverName)
		klog.ErrorS(err, "Error retrieving CSIDriver from store", "driver", driverName)
		return
	}
	if !exists {
		klog.InfoS("CSIDriver object not found; stopping csiNodeUpdater", "driver", driverName)
		u.unregisterDriver(driverName)
		return
	}
	driver, ok := obj.(*v1.CSIDriver)
	if !ok {
		klog.ErrorS(fmt.Errorf("invalid CSIDriver object type"), "failed to cast CSIDriver object", "driver", driverName)
		return
	}

	// Get the update period.
	period := getNodeAllocatableUpdatePeriod(driver)
	if period == 0 {
		klog.V(7).InfoS("NodeAllocatableUpdatePeriodSeconds is not configured; disabling updates", "driver", driverName)
		u.unregisterDriver(driverName)
		return
	}

	newStopCh := make(chan struct{})
	prevStopCh, loaded := u.driverUpdaters.Swap(driverName, newStopCh)
	// If an updater is already running, stop it so we can reconfigure.
	if loaded && prevStopCh != nil {
		if stopCh, ok := prevStopCh.(chan struct{}); ok {
			close(stopCh)
		}
	}

	// Start the periodic update goroutine.
	go u.runPeriodicUpdate(driverName, period, newStopCh)
}

// unregisterDriver stops any running periodic update goroutine for the given driver.
func (u *csiNodeUpdater) unregisterDriver(driverName string) {
	prev, loaded := u.driverUpdaters.LoadAndDelete(driverName)
	if loaded && prev != nil {
		if stopCh, ok := prev.(chan struct{}); ok {
			close(stopCh)
		}
	}
}

// runPeriodicUpdate runs the periodic update loop for a driver.
func (u *csiNodeUpdater) runPeriodicUpdate(driverName string, period time.Duration, stopCh <-chan struct{}) {
	ticker := time.NewTicker(period)
	defer ticker.Stop()
	klog.V(7).InfoS("Starting periodic updates for driver", "driver", driverName, "period", period)
	for {
		select {
		case <-ticker.C:
			if err := updateCSIDriver(driverName); err != nil {
				klog.ErrorS(err, "Failed to update CSIDriver", "driver", driverName)
			}
		case <-stopCh:
			klog.V(4).InfoS("Stopping periodic updates for driver", "driver", driverName, "period", period)
			return
		}
	}
}

// isDriverInstalled checks if the CSI driver is installed on the node by checking the global csiDrivers map
func isDriverInstalled(driverName string) bool {
	_, ok := csiDrivers.Get(driverName)
	return ok
}

// getNodeAllocatableUpdatePeriod returns the NodeAllocatableUpdatePeriodSeconds value from the CSIDriver
func getNodeAllocatableUpdatePeriod(driver *v1.CSIDriver) time.Duration {
	if driver == nil || driver.Spec.NodeAllocatableUpdatePeriodSeconds == nil {
		return 0
	}
	return time.Duration(*driver.Spec.NodeAllocatableUpdatePeriodSeconds) * time.Second
}
