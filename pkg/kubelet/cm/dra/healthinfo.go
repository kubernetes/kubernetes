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

// TODO(#133118): Make health timeout configurable.
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
		klog.Background().Error(err, "Failed to load health checkpoint, proceeding with empty cache")
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

	defer func() {
		if err := os.Remove(tempFile.Name()); err != nil && !os.IsNotExist(err) {
			klog.Background().Error(err, "Failed to remove temporary checkpoint file", "path", tempFile.Name())
		}
	}()

	if _, err := tempFile.Write(data); err != nil {
		_ = tempFile.Close()
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

	_ = cache.withRLock(func() error {
		now := time.Now()
		if driver, ok := (*cache.HealthInfo)[driverName]; ok {
			key := poolName + "/" + deviceName
			if device, ok := driver.Devices[key]; ok {
				if now.Sub(device.LastUpdated) > healthTimeout {
					res = state.DeviceHealthStatusUnknown
				} else {
					res = device.Health
				}
			}
		}
		return nil
	})
	return res
}

// updateHealthInfo reconciles the cache with a fresh list of device health states
// from a plugin. It identifies which devices have changed state and handles devices
// that are no longer being reported by the plugin.
func (cache *healthInfoCache) updateHealthInfo(driverName string, devices []state.DeviceHealth) ([]state.DeviceHealth, error) {
	changedDevices := []state.DeviceHealth{}
	err := cache.withLock(func() error {
		now := time.Now()
		currentDriver, exists := (*cache.HealthInfo)[driverName]
		if !exists {
			currentDriver = state.DriverHealthState{Devices: make(map[string]state.DeviceHealth)}
			(*cache.HealthInfo)[driverName] = currentDriver
		}

		reportedKeys := make(map[string]struct{})

		// Phase 1: Process the incoming report from the plugin.
		// Update existing devices, add new ones, and record all devices
		// present in this report.
		for _, reportedDevice := range devices {
			reportedDevice.LastUpdated = now
			key := reportedDevice.PoolName + "/" + reportedDevice.DeviceName
			reportedKeys[key] = struct{}{}

			existingDevice, ok := currentDriver.Devices[key]

			if !ok || existingDevice.Health != reportedDevice.Health {
				changedDevices = append(changedDevices, reportedDevice)
			}

			currentDriver.Devices[key] = reportedDevice
		}

		// Phase 2: Handle devices that are in the cache but were not in the report.
		// These devices may have been removed or the plugin may have stopped monitoring
		// them. Mark them as "Unknown" if their status has timed out.
		for key, existingDevice := range currentDriver.Devices {
			if _, wasReported := reportedKeys[key]; !wasReported {
				if existingDevice.Health != state.DeviceHealthStatusUnknown && now.Sub(existingDevice.LastUpdated) > healthTimeout {
					existingDevice.Health = state.DeviceHealthStatusUnknown
					existingDevice.LastUpdated = now
					currentDriver.Devices[key] = existingDevice

					changedDevices = append(changedDevices, existingDevice)
				}
			}
		}

		// Phase 3: Persist changes to the checkpoint file if any state changed.
		if len(changedDevices) > 0 {
			if err := cache.saveToCheckpointInternal(); err != nil {
				klog.Background().Error(err, "Failed to save health checkpoint after update. Kubelet restart may lose the device health information.")
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
