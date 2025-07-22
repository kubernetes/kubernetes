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

package dra

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/cm/dra/state"
)

const (
	healthTimeout = 30 * time.Second
)

// healthInfoCache is a cache of known device health.
type healthInfoCache struct {
	sync.RWMutex
	HealthInfo *state.DevicesHealthMap
	stateFile  string
}

// newHealthInfoCache creates a new cache, loading from a checkpoint if present.
func newHealthInfoCache(stateFile string) (*healthInfoCache, error) {
	cache := &healthInfoCache{
		HealthInfo: &state.DevicesHealthMap{},
		stateFile:  stateFile,
	}
	if err := cache.loadFromCheckpoint(); err != nil {
		klog.InfoS("Failed to load health checkpoint, proceeding with empty cache", "err", err)
	}
	return cache, nil
}

// loadFromCheckpoint loads the cache from the state file.
func (cache *healthInfoCache) loadFromCheckpoint() error {
	if cache.stateFile == "" {
		return nil
	}
	data, err := os.ReadFile(cache.stateFile)
	if err != nil {
		if os.IsNotExist(err) {
			cache.HealthInfo = &state.DevicesHealthMap{}
			return nil
		}
		return err
	}
	return json.Unmarshal(data, cache.HealthInfo)
}

// withLock runs a function while holding the healthInfoCache lock.
func (cache *healthInfoCache) withLock(f func() error) error {
	cache.Lock()
	defer cache.Unlock()
	return f()
}

// withRLock runs a function while holding the healthInfoCache rlock.
func (cache *healthInfoCache) withRLock(f func() error) error {
	cache.RLock()
	defer cache.RUnlock()
	return f()
}

// saveToCheckpointInternal does the actual saving without locking.
// Assumes the caller holds the necessary lock.
func (cache *healthInfoCache) saveToCheckpointInternal() error {
	if cache.stateFile == "" {
		return nil
	}
	data, err := json.Marshal(cache.HealthInfo)
	if err != nil {
		return fmt.Errorf("failed to marshal health info: %w", err)
	}

	tempFile, err := os.CreateTemp(filepath.Dir(cache.stateFile), filepath.Base(cache.stateFile)+".tmp")
	if err != nil {
		return fmt.Errorf("failed to create temp checkpoint file: %w", err)
	}

	defer os.Remove(tempFile.Name())

	if _, err := tempFile.Write(data); err != nil {
		tempFile.Close()
		return fmt.Errorf("failed to write to temporary file: %w", err)
	}

	if err := tempFile.Close(); err != nil {
		return fmt.Errorf("failed to close temporary file: %w", err)
	}

	if err := os.Rename(tempFile.Name(), cache.stateFile); err != nil {
		return fmt.Errorf("failed to rename temporary file to state file: %w", err)
	}

	return nil
}

// getHealthInfo returns the current health info, adjusting for timeouts.
func (cache *healthInfoCache) getHealthInfo(driverName, poolName, deviceName string) state.DeviceHealthStatus {
	res := state.DeviceHealthStatusUnknown
	now := time.Now()

	_ = cache.withRLock(func() error {
		if driver, ok := (*cache.HealthInfo)[driverName]; ok {
			for _, device := range driver.Devices {
				if device.PoolName == poolName && device.DeviceName == deviceName {
					if now.Sub(device.LastUpdated) > healthTimeout {
						res = state.DeviceHealthStatusUnknown
					} else {
						res = device.Health
					}
					return nil
				}
			}
		}
		return nil
	})
	return res
}

// updateHealthInfo updates the cache with a list of devices for driver, reconciling the full state.
func (cache *healthInfoCache) updateHealthInfo(driverName string, devices []state.DeviceHealth) ([]state.DeviceHealth, error) {
	changedDevices := []state.DeviceHealth{}
	err := cache.withLock(func() error {
		now := time.Now()
		currentDriver, exists := (*cache.HealthInfo)[driverName]
		if !exists {
			currentDriver = state.DriverHealthState{Devices: []state.DeviceHealth{}}
		}

		// Build map of reported devices
		// Note: Within the driver's cache, devices can be keyed by '<pool name>/<device name>'
		// as each update iteration is within one driver's scope.
		reported := make(map[string]state.DeviceHealth)
		for _, dev := range devices {
			dev.LastUpdated = now
			key := dev.PoolName + "/" + dev.DeviceName
			reported[key] = dev
		}

		// Reconcile the cache: Update existing, add new, and mark unreported devices as "Unknown"
		newDevices := make([]state.DeviceHealth, 0, len(reported))
		for _, existing := range currentDriver.Devices {
			key := existing.PoolName + "/" + existing.DeviceName
			if updated, ok := reported[key]; ok {
				if existing.Health != updated.Health {
					changedDevices = append(changedDevices, updated)
				}
				// Add  device to new device health status list
				newDevices = append(newDevices, updated)
				// Remove from reported device list
				delete(reported, key)
				// If device surpasses health timeout set to unknown
			} else if existing.Health != state.DeviceHealthStatusUnknown && now.Sub(existing.LastUpdated) > healthTimeout {
				existing.Health = state.DeviceHealthStatusUnknown
				existing.LastUpdated = now
				changedDevices = append(changedDevices, existing)
				newDevices = append(newDevices, existing)
				// Within the health timeout consider healthy and add as new device
			} else {
				newDevices = append(newDevices, existing)
			}
		}

		// Add remaining new devices
		for _, dev := range reported {
			dev.LastUpdated = now
			newDevices = append(newDevices, dev)
			changedDevices = append(changedDevices, dev)
		}

		currentDriver.Devices = newDevices
		(*cache.HealthInfo)[driverName] = currentDriver

		if len(changedDevices) > 0 {
			if err := cache.saveToCheckpointInternal(); err != nil {
				klog.Background().Error(err, "Failed to save health checkpoint after update")
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return changedDevices, nil
}

// clearDriver clears all health data for a specific driver.
func (cache *healthInfoCache) clearDriver(driverName string) error {
	return cache.withLock(func() error {
		delete(*cache.HealthInfo, driverName)
		return cache.saveToCheckpointInternal()
	})
}
