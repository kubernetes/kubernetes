/*
Copyright 2024 The Kubernetes Authors.

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
	"errors"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/kubernetes/pkg/kubelet/cm/dra/state"
)

const (
	testDriver    = "test-driver"
	testPool      = "test-pool"
	testDevice    = "test-device"
	testNamespace = "test-namespace"
	testClaim     = "test-claim"
)

var (
	testDeviceHealth = state.DeviceHealth{
		PoolName:   testPool,
		DeviceName: testDevice,
		Health:     "Healthy",
	}
)

// `TestNewHealthInfoCache tests cache creation and checkpoint loading.
func TestNewHealthInfoCache(t *testing.T) {
	tests := []struct {
		description string
		stateFile   string
		wantErr     bool
	}{
		{
			description: "successfully created cache",
			stateFile:   "/tmp/health_checkpoint",
		},
		{
			description: "empty state file",
			stateFile:   "",
		},
	}
	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			cache, err := newHealthInfoCache(test.stateFile)
			if test.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.NotNil(t, cache)
			if test.stateFile != "" {
				os.Remove(test.stateFile)
			}
		})
	}
}

// TestWithLock tests the withLock method’s behavior.
func TestWithLock(t *testing.T) {
	cache, err := newHealthInfoCache("")
	require.NoError(t, err)
	tests := []struct {
		description string
		f           func() error
		wantErr     bool
	}{
		{
			description: "lock prevents concurrent lock",
			f: func() error {
				if cache.TryLock() {
					return errors.New("Lock succeeded")
				}
				return nil
			},
		},
		{
			description: "erroring function",
			f: func() error {
				return errors.New("test error")
			},
			wantErr: true,
		},
	}
	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			err := cache.withLock(test.f)
			if test.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestWithRLock tests the withRLock method’s behavior.
func TestWithRLock(t *testing.T) {
	cache, err := newHealthInfoCache("")
	require.NoError(t, err)
	tests := []struct {
		description string
		f           func() error
		wantErr     bool
	}{
		{
			description: "rlock allows concurrent rlock",
			f: func() error {
				if !cache.TryRLock() {
					return errors.New("Concurrent RLock failed")
				}
				return nil
			},
			wantErr: false,
		},
		{
			description: "rlock prevents lock",
			f: func() error {
				if cache.TryLock() {
					return errors.New("Write Lock succeeded: Bad")
				}
				return nil
			},
			wantErr: false,
		},
	}
	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			err := cache.withRLock(test.f)
			if test.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestGetHealthInfo tests retrieving health status.
func TestGetHealthInfo(t *testing.T) {
	cache, err := newHealthInfoCache("")
	require.NoError(t, err)

	// Initial state
	assert.Equal(t, state.DeviceHealthString("Unknown"), cache.getHealthInfo(testDriver, testPool, testDevice))

	// Add a device
	cache.updateHealthInfo(testDriver, []state.DeviceHealth{testDeviceHealth})
	assert.Equal(t, state.DeviceHealthString("Healthy"), cache.getHealthInfo(testDriver, testPool, testDevice))

	// Test timeout (simulated with old LastUpdated)
	cache.withLock(func() error {
		driver := (*cache.HealthInfo)[testDriver]
		driver.Devices[0].LastUpdated = time.Now().Add((-healthTimeout) - time.Second)
		(*cache.HealthInfo)[testDriver] = driver
		return nil
	})
	assert.Equal(t, state.DeviceHealthString("Unknown"), cache.getHealthInfo(testDriver, testPool, testDevice))
}

// TestGetHealthInfoRobust tests retrieving health status logic solely & against many cases.
func TestGetHealthInfoRobust(t *testing.T) {
	tests := []struct {
		name           string
		initialState   *state.DevicesHealthMap
		driverName     string
		poolName       string
		deviceName     string
		expectedHealth state.DeviceHealthString
	}{
		{
			name:           "empty cache",
			initialState:   &state.DevicesHealthMap{},
			driverName:     testDriver,
			poolName:       testPool,
			deviceName:     testDevice,
			expectedHealth: "Unknown",
		},
		{
			name: "device exists and is healthy",
			initialState: &state.DevicesHealthMap{
				testDriver: {Devices: []state.DeviceHealth{{PoolName: testPool, DeviceName: testDevice, Health: "Healthy", LastUpdated: time.Now()}}},
			},
			driverName:     testDriver,
			poolName:       testPool,
			deviceName:     testDevice,
			expectedHealth: "Healthy",
		},
		{
			name: "device exists and is unhealthy",
			initialState: &state.DevicesHealthMap{
				testDriver: {Devices: []state.DeviceHealth{{PoolName: testPool, DeviceName: testDevice, Health: "Unhealthy", LastUpdated: time.Now()}}},
			},
			driverName:     testDriver,
			poolName:       testPool,
			deviceName:     testDevice,
			expectedHealth: "Unhealthy",
		},
		{
			name: "device exists but timed out",
			initialState: &state.DevicesHealthMap{
				testDriver: {Devices: []state.DeviceHealth{{PoolName: testPool, DeviceName: testDevice, Health: "Healthy", LastUpdated: time.Now().Add((-1 * healthTimeout) - time.Second)}}},
			},
			driverName:     testDriver,
			poolName:       testPool,
			deviceName:     testDevice,
			expectedHealth: "Unknown",
		},
		{
			name: "device exists, just within timeout",
			initialState: &state.DevicesHealthMap{
				testDriver: {Devices: []state.DeviceHealth{{PoolName: testPool, DeviceName: testDevice, Health: "Healthy", LastUpdated: time.Now().Add((-1 * healthTimeout) + time.Second)}}},
			},
			driverName:     testDriver,
			poolName:       testPool,
			deviceName:     testDevice,
			expectedHealth: "Healthy",
		},
		{
			name: "device exists, exactly at timeout(bound exclusionary)",
			initialState: &state.DevicesHealthMap{
				testDriver: {Devices: []state.DeviceHealth{{PoolName: testPool, DeviceName: testDevice, Health: "Healthy", LastUpdated: time.Now().Add(-30 * time.Second)}}},
			},
			driverName:     testDriver,
			poolName:       testPool,
			deviceName:     testDevice,
			expectedHealth: "Unknown",
		},
		{
			name: "device does not exist",
			initialState: &state.DevicesHealthMap{
				testDriver: {Devices: []state.DeviceHealth{{PoolName: testPool, DeviceName: testDevice, Health: "Healthy", LastUpdated: time.Now()}}},
			},
			driverName:     testDriver,
			poolName:       testPool,
			deviceName:     "device2",
			expectedHealth: "Unknown",
		},
		{
			name: "driver does not exist",
			initialState: &state.DevicesHealthMap{
				testDriver: {Devices: []state.DeviceHealth{{PoolName: testPool, DeviceName: testDevice, Health: "Healthy", LastUpdated: time.Now()}}},
			},
			driverName:     "driver2",
			poolName:       testPool,
			deviceName:     testDevice,
			expectedHealth: "Unknown",
		},
		{
			name: "pool does not exist",
			initialState: &state.DevicesHealthMap{
				testDriver: {Devices: []state.DeviceHealth{{PoolName: testPool, DeviceName: testDevice, Health: "Healthy", LastUpdated: time.Now()}}},
			},
			driverName:     testDriver,
			poolName:       "pool2",
			deviceName:     testDevice,
			expectedHealth: "Unknown",
		},
		{
			name: "multiple devices",
			initialState: &state.DevicesHealthMap{
				testDriver: {Devices: []state.DeviceHealth{
					{PoolName: testPool, DeviceName: testDevice, Health: "Healthy", LastUpdated: time.Now()},
					{PoolName: testPool, DeviceName: "device2", Health: "Unhealthy", LastUpdated: time.Now()},
				}},
			},
			driverName:     testDriver,
			poolName:       testPool,
			deviceName:     "device2",
			expectedHealth: "Unhealthy",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cache := &healthInfoCache{HealthInfo: tt.initialState}
			health := cache.getHealthInfo(tt.driverName, tt.poolName, tt.deviceName)
			assert.Equal(t, tt.expectedHealth, health)
		})
	}
}

// TestUpdateHealthInfo tests adding, updating, and reconciling device health.
func TestUpdateHealthInfo(t *testing.T) {
	tmpFile := "/tmp/health_checkpoint_test"
	defer os.Remove(tmpFile)
	cache, err := newHealthInfoCache(tmpFile)
	require.NoError(t, err)

	// Add new device
	changed, err := cache.updateHealthInfo(testDriver, []state.DeviceHealth{testDeviceHealth})
	assert.NoError(t, err)
	assert.True(t, changed)
	assert.Equal(t, state.DeviceHealthString("Healthy"), cache.getHealthInfo(testDriver, testPool, testDevice))

	// Update with no change
	changed, err = cache.updateHealthInfo(testDriver, []state.DeviceHealth{testDeviceHealth})
	assert.NoError(t, err)
	assert.False(t, changed)

	// Update with new health
	newHealth := testDeviceHealth
	newHealth.Health = "Unhealthy"
	changed, err = cache.updateHealthInfo(testDriver, []state.DeviceHealth{newHealth})
	assert.NoError(t, err)
	assert.True(t, changed)
	assert.Equal(t, state.DeviceHealthString("Unhealthy"), cache.getHealthInfo(testDriver, testPool, testDevice))

	// Add second device, omit first
	secondDevice := state.DeviceHealth{PoolName: testPool, DeviceName: "device2", Health: "Healthy"}
	changed, err = cache.updateHealthInfo(testDriver, []state.DeviceHealth{secondDevice})
	assert.NoError(t, err)
	assert.True(t, changed)
	assert.Equal(t, state.DeviceHealthString("Healthy"), cache.getHealthInfo(testDriver, testPool, "device2"))
	assert.Equal(t, state.DeviceHealthString("Unhealthy"), cache.getHealthInfo(testDriver, testPool, testDevice))

	// Test persistence
	cache2, err := newHealthInfoCache(tmpFile)
	assert.NoError(t, err)
	assert.Equal(t, state.DeviceHealthString("Healthy"), cache2.getHealthInfo(testDriver, testPool, "device2"))

	// Test how updateHealthInfo handles device timeouts
	timeoutDevice := state.DeviceHealth{PoolName: testPool, DeviceName: "timeoutDevice", Health: "Unhealthy"}
	changed, err = cache.updateHealthInfo(testDriver, []state.DeviceHealth{timeoutDevice})
	assert.NoError(t, err)
	assert.True(t, changed)

	// Manually manipulate the last updated time of timeoutDevice to seem like it surpassed healthtimeout.
	// This bypassed manually waiting.
	(*cache.HealthInfo)[testDriver].Devices[2].LastUpdated = time.Now().Add((-healthTimeout) - time.Second)

	changed, err = cache.updateHealthInfo(testDriver, []state.DeviceHealth{})
	assert.NoError(t, err)
	assert.True(t, changed)
	assert.Equal(t, state.DeviceHealthString("Unknown"), (*cache.HealthInfo)[testDriver].Devices[2].Health, "Health status should be Unknown after timeout in updateHealthInfo")

}

// TestClearDriver tests clearing a driver’s health data.
func TestClearDriver(t *testing.T) {
	cache, err := newHealthInfoCache("")
	require.NoError(t, err)

	cache.updateHealthInfo(testDriver, []state.DeviceHealth{testDeviceHealth})
	assert.Equal(t, state.DeviceHealthString("Healthy"), cache.getHealthInfo(testDriver, testPool, testDevice))

	err = cache.clearDriver(testDriver)
	assert.NoError(t, err)
	assert.Equal(t, state.DeviceHealthString("Unknown"), cache.getHealthInfo(testDriver, testPool, testDevice))
}
