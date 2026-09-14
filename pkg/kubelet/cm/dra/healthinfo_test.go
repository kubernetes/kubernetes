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
	"fmt"
	"os"
	"path"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/kubernetes/pkg/kubelet/cm/dra/state"
	"k8s.io/kubernetes/test/utils/ktesting"
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
		PoolName:           testPool,
		DeviceName:         testDevice,
		Health:             state.DeviceHealthStatusHealthy,
		HealthCheckTimeout: DefaultHealthTimeout,
	}
)

// `TestNewHealthInfoCache tests cache creation and checkpoint loading.
func TestNewHealthInfoCache(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	tests := []struct {
		description string
		stateFile   string
		wantErr     bool
	}{
		{
			description: "successfully created cache",
			stateFile:   path.Join(t.TempDir(), "health_checkpoint"),
		},
		{
			description: "empty state file",
			stateFile:   "",
		},
	}
	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			if test.stateFile != "" {
				f, err := os.Create(test.stateFile)
				require.NoError(t, err)
				require.NoError(t, f.Close())
			}
			cache, err := newHealthInfoCache(logger, test.stateFile)
			if test.wantErr {
				assert.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.NotNil(t, cache)
			if test.stateFile != "" {
				require.NoError(t, os.Remove(test.stateFile))
			}
		})
	}
}

// TestNewHealthInfoCacheCorruptCheckpoint verifies that a checkpoint which
// decodes to a nil map — at the top level or inside a driver entry — does not
// leave the cache with a nil map. Without this guarantee, the first
// updateHealthInfo call would panic in the health goroutine with
// "assignment to entry in nil map".
func TestNewHealthInfoCacheCorruptCheckpoint(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	tests := []struct {
		description string
		contents    string
	}{
		{description: "json null", contents: "null"},
		{description: "nil driver devices", contents: fmt.Sprintf(`{"%s":{"Devices":null}}`, testDriver)},
		{description: "torn write", contents: `{"driver": {"Devices`},
		{description: "empty file", contents: ""},
		{description: "garbage", contents: "not json"},
	}
	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			stateFile := path.Join(t.TempDir(), "health_checkpoint")
			require.NoError(t, os.WriteFile(stateFile, []byte(test.contents), 0o600))

			cache, err := newHealthInfoCache(logger, stateFile)
			require.NoError(t, err)
			require.NotNil(t, cache)
			require.NotNil(t, cache.HealthInfo)
			require.NotNil(t, *cache.HealthInfo, "top-level map must not be nil after loading a corrupt checkpoint")
			for name, driver := range *cache.HealthInfo {
				require.NotNil(t, driver.Devices, "Devices map for driver %q must not be nil after loading a corrupt checkpoint", name)
			}

			// A follow-up update must not panic on a nil map.
			_, err = cache.updateHealthInfo(logger, testDriver, []state.DeviceHealth{testDeviceHealth})
			require.NoError(t, err)
			assert.Equal(t, state.DeviceHealthStatusHealthy, cache.getHealthInfo(testDriver, testPool, testDevice).Health)
		})
	}
}

// Helper function to compare DeviceHealth slices ignoring LastUpdated time
func assertDeviceHealthElementsMatchIgnoreTime(t *testing.T, expected, actual []state.DeviceHealth) {
	require.Len(t, actual, len(expected), "Number of changed devices mismatch")

	// Create comparable versions without LastUpdated
	normalize := func(dh state.DeviceHealth) state.DeviceHealth {
		// Zero out time for comparison
		dh.LastUpdated = time.Time{}
		return dh
	}

	expectedNormalized := make([]state.DeviceHealth, len(expected))
	actualNormalized := make([]state.DeviceHealth, len(actual))

	for i := range expected {
		expectedNormalized[i] = normalize(expected[i])
	}
	for i := range actual {
		actualNormalized[i] = normalize(actual[i])
	}

	assert.ElementsMatch(t, expectedNormalized, actualNormalized, "Changed device elements mismatch (ignoring time)")
}

// TestWithLock tests the withLock method’s behavior.
func TestWithLock(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cache, err := newHealthInfoCache(logger, "")
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
					defer cache.Unlock()
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
				require.NoError(t, err)
			}
		})
	}
}

// TestWithRLock tests the withRLock method’s behavior.
func TestWithRLock(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cache, err := newHealthInfoCache(logger, "")
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
				defer cache.RUnlock()
				return nil
			},
			wantErr: false,
		},
		{
			description: "rlock prevents lock",
			f: func() error {
				if cache.TryLock() {
					defer cache.Unlock()
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
				require.NoError(t, err)
			}
		})
	}
}

// TestGetHealthInfo tests retrieving health status.
func TestGetHealthInfo(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cache, err := newHealthInfoCache(logger, "")
	require.NoError(t, err)

	// Initial state
	assert.Equal(t, state.DeviceHealthStatusUnknown, cache.getHealthInfo(testDriver, testPool, testDevice).Health)

	// Add a device
	_, err = cache.updateHealthInfo(logger, testDriver, []state.DeviceHealth{testDeviceHealth})
	require.NoError(t, err)
	assert.Equal(t, state.DeviceHealthStatusHealthy, cache.getHealthInfo(testDriver, testPool, testDevice).Health)

	// Test timeout (simulated with old LastUpdated)
	err = cache.withLock(func() error {
		driverState := (*cache.HealthInfo)[testDriver]
		deviceKey := testPool + "/" + testDevice
		device := driverState.Devices[deviceKey]
		device.LastUpdated = time.Now().Add((-DefaultHealthTimeout) - time.Second)
		driverState.Devices[deviceKey] = device
		(*cache.HealthInfo)[testDriver] = driverState
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, state.DeviceHealthStatusUnknown, cache.getHealthInfo(testDriver, testPool, testDevice).Health)
}

// TestGetHealthInfoRobust tests retrieving health status logic solely & against many cases.
func TestGetHealthInfoRobust(t *testing.T) {
	tests := []struct {
		name           string
		initialState   *state.DevicesHealthMap
		driverName     string
		poolName       string
		deviceName     string
		expectedHealth state.DeviceHealthStatus
	}{
		{
			name:           "empty cache",
			initialState:   &state.DevicesHealthMap{},
			driverName:     testDriver,
			poolName:       testPool,
			deviceName:     testDevice,
			expectedHealth: state.DeviceHealthStatusUnknown,
		},
		{
			name: "device exists and is healthy",
			initialState: &state.DevicesHealthMap{
				testDriver: {Devices: map[string]state.DeviceHealth{
					testPool + "/" + testDevice: {PoolName: testPool, DeviceName: testDevice, Health: state.DeviceHealthStatusHealthy, LastUpdated: time.Now(), HealthCheckTimeout: DefaultHealthTimeout},
				}},
			},
			driverName:     testDriver,
			poolName:       testPool,
			deviceName:     testDevice,
			expectedHealth: "Healthy",
		},
		{
			name: "device exists and is unhealthy",
			initialState: &state.DevicesHealthMap{
				testDriver: {Devices: map[string]state.DeviceHealth{
					testPool + "/" + testDevice: {PoolName: testPool, DeviceName: testDevice, Health: state.DeviceHealthStatusUnhealthy, LastUpdated: time.Now(), HealthCheckTimeout: DefaultHealthTimeout},
				}},
			},
			driverName:     testDriver,
			poolName:       testPool,
			deviceName:     testDevice,
			expectedHealth: "Unhealthy",
		},
		{
			name: "device exists but timed out",
			initialState: &state.DevicesHealthMap{
				testDriver: {Devices: map[string]state.DeviceHealth{
					testPool + "/" + testDevice: {PoolName: testPool, DeviceName: testDevice, Health: state.DeviceHealthStatusHealthy, LastUpdated: time.Now().Add((-1 * DefaultHealthTimeout) - time.Second), HealthCheckTimeout: DefaultHealthTimeout},
				}},
			},
			driverName:     testDriver,
			poolName:       testPool,
			deviceName:     testDevice,
			expectedHealth: "Unknown",
		},
		{
			name: "device exists, just within timeout",
			initialState: &state.DevicesHealthMap{
				testDriver: {Devices: map[string]state.DeviceHealth{
					testPool + "/" + testDevice: {PoolName: testPool, DeviceName: testDevice, Health: state.DeviceHealthStatusHealthy, LastUpdated: time.Now().Add((-1 * DefaultHealthTimeout) + time.Second), HealthCheckTimeout: DefaultHealthTimeout},
				}},
			},
			driverName:     testDriver,
			poolName:       testPool,
			deviceName:     testDevice,
			expectedHealth: "Healthy",
		},
		{
			name: "device does not exist, just outside of timeout",
			initialState: &state.DevicesHealthMap{
				testDriver: {Devices: map[string]state.DeviceHealth{
					testPool + "/" + testDevice: {PoolName: testPool, DeviceName: testDevice, Health: state.DeviceHealthStatusHealthy, LastUpdated: time.Now().Add((-1 * DefaultHealthTimeout) - time.Second), HealthCheckTimeout: DefaultHealthTimeout},
				}},
			},
			driverName:     testDriver,
			poolName:       testPool,
			deviceName:     "device2",
			expectedHealth: "Unknown",
		},
		{
			name: "device does not exist",
			initialState: &state.DevicesHealthMap{
				testDriver: {Devices: map[string]state.DeviceHealth{
					testPool + "/" + testDevice: {PoolName: testPool, DeviceName: testDevice, Health: state.DeviceHealthStatusHealthy, LastUpdated: time.Now(), HealthCheckTimeout: DefaultHealthTimeout},
				}},
			},
			driverName:     testDriver,
			poolName:       testPool,
			deviceName:     "device2",
			expectedHealth: "Unknown",
		},
		{
			name: "driver does not exist",
			initialState: &state.DevicesHealthMap{
				testDriver: {Devices: map[string]state.DeviceHealth{
					testPool + "/" + testDevice: {PoolName: testPool, DeviceName: testDevice, Health: state.DeviceHealthStatusHealthy, LastUpdated: time.Now(), HealthCheckTimeout: DefaultHealthTimeout},
				}},
			},
			driverName:     "driver2",
			poolName:       testPool,
			deviceName:     testDevice,
			expectedHealth: "Unknown",
		},
		{
			name: "pool does not exist",
			initialState: &state.DevicesHealthMap{
				testDriver: {Devices: map[string]state.DeviceHealth{
					testPool + "/" + testDevice: {PoolName: testPool, DeviceName: testDevice, Health: state.DeviceHealthStatusHealthy, LastUpdated: time.Now(), HealthCheckTimeout: DefaultHealthTimeout},
				}},
			},
			driverName:     testDriver,
			poolName:       "pool2",
			deviceName:     testDevice,
			expectedHealth: "Unknown",
		},
		{
			name: "multiple devices",
			initialState: &state.DevicesHealthMap{
				testDriver: {Devices: map[string]state.DeviceHealth{
					testPool + "/" + testDevice: {PoolName: testPool, DeviceName: testDevice, Health: state.DeviceHealthStatusHealthy, LastUpdated: time.Now(), HealthCheckTimeout: DefaultHealthTimeout},
					testPool + "/device2":       {PoolName: testPool, DeviceName: "device2", Health: state.DeviceHealthStatusUnhealthy, LastUpdated: time.Now(), HealthCheckTimeout: DefaultHealthTimeout},
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
			health := cache.getHealthInfo(tt.driverName, tt.poolName, tt.deviceName).Health
			assert.Equal(t, tt.expectedHealth, health)
		})
	}
}

// TestUpdateHealthInfo tests adding, updating, and reconciling device health.
func TestUpdateHealthInfo(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	tmpFile := path.Join(t.TempDir(), "health_checkpoint_test")
	cache, err := newHealthInfoCache(logger, tmpFile)
	require.NoError(t, err)

	// 1 -- Add new device
	deviceToAdd := testDeviceHealth
	expectedChanged1 := []state.DeviceHealth{deviceToAdd}
	changedDevices, err := cache.updateHealthInfo(logger, testDriver, []state.DeviceHealth{testDeviceHealth})
	require.NoError(t, err)
	assertDeviceHealthElementsMatchIgnoreTime(t, expectedChanged1, changedDevices)
	assert.Equal(t, state.DeviceHealthStatusHealthy, cache.getHealthInfo(testDriver, testPool, testDevice).Health)

	// 2 -- Update with no change
	changedDevices, err = cache.updateHealthInfo(logger, testDriver, []state.DeviceHealth{testDeviceHealth})
	require.NoError(t, err)
	assert.Empty(t, changedDevices, "Scenario 2: Changed devices list should be empty")

	// 3 -- Update with new health
	newHealth := testDeviceHealth
	newHealth.Health = state.DeviceHealthStatusUnhealthy
	expectedChanged3 := []state.DeviceHealth{newHealth}
	changedDevices, err = cache.updateHealthInfo(logger, testDriver, []state.DeviceHealth{newHealth})
	require.NoError(t, err)
	assertDeviceHealthElementsMatchIgnoreTime(t, expectedChanged3, changedDevices)
	assert.Equal(t, state.DeviceHealthStatusUnhealthy, cache.getHealthInfo(testDriver, testPool, testDevice).Health)

	// 4 -- Add second device, omit first
	secondDevice := state.DeviceHealth{PoolName: testPool, DeviceName: "device2", Health: state.DeviceHealthStatusHealthy, HealthCheckTimeout: DefaultHealthTimeout}
	// When the first device is omitted, it should be marked as "Unknown" after a timeout.
	// For this test, we simulate the timeout by not reporting it.
	firstDeviceAsUnknown := newHealth
	firstDeviceAsUnknown.Health = state.DeviceHealthStatusUnknown
	expectedChanged4 := []state.DeviceHealth{secondDevice, firstDeviceAsUnknown}
	// Manually set the time of the first device to be outside the timeout window
	err = cache.withLock(func() error {
		deviceKey := testPool + "/" + testDevice
		device := (*cache.HealthInfo)[testDriver].Devices[deviceKey]
		device.LastUpdated = time.Now().Add(-DefaultHealthTimeout * 2)
		(*cache.HealthInfo)[testDriver].Devices[deviceKey] = device
		return nil
	})
	require.NoError(t, err)

	changedDevices, err = cache.updateHealthInfo(logger, testDriver, []state.DeviceHealth{secondDevice})
	require.NoError(t, err)
	assertDeviceHealthElementsMatchIgnoreTime(t, expectedChanged4, changedDevices)
	assert.Equal(t, state.DeviceHealthStatusHealthy, cache.getHealthInfo(testDriver, testPool, "device2").Health)
	assert.Equal(t, state.DeviceHealthStatusUnknown, cache.getHealthInfo(testDriver, testPool, testDevice).Health)

	// 5 -- Test persistence
	cache2, err := newHealthInfoCache(logger, tmpFile)
	require.NoError(t, err)
	assert.Equal(t, state.DeviceHealthStatusHealthy, cache2.getHealthInfo(testDriver, testPool, "device2").Health)
	assert.Equal(t, state.DeviceHealthStatusUnknown, cache2.getHealthInfo(testDriver, testPool, testDevice).Health)

	// 6 -- Test how updateHealthInfo handles device timeouts
	timeoutDevice := state.DeviceHealth{PoolName: testPool, DeviceName: "timeoutDevice", Health: "Unhealthy", HealthCheckTimeout: DefaultHealthTimeout}
	_, err = cache.updateHealthInfo(logger, testDriver, []state.DeviceHealth{timeoutDevice})
	require.NoError(t, err)

	// Manually manipulate the last updated time of timeoutDevice to seem like it surpassed healthtimeout.
	err = cache.withLock(func() error {
		driverState := (*cache.HealthInfo)[testDriver]
		deviceKey := testPool + "/timeoutDevice"
		device := driverState.Devices[deviceKey]
		device.LastUpdated = time.Now().Add((-DefaultHealthTimeout) - time.Second)
		driverState.Devices[deviceKey] = device
		(*cache.HealthInfo)[testDriver] = driverState
		return nil
	})
	require.NoError(t, err)

	expectedTimeoutDeviceUnknown := state.DeviceHealth{PoolName: testPool, DeviceName: "timeoutDevice", Health: state.DeviceHealthStatusUnknown, HealthCheckTimeout: DefaultHealthTimeout}
	expectedChanged6 := []state.DeviceHealth{expectedTimeoutDeviceUnknown}
	changedDevices, err = cache.updateHealthInfo(logger, testDriver, []state.DeviceHealth{})
	require.NoError(t, err)
	assertDeviceHealthElementsMatchIgnoreTime(t, expectedChanged6, changedDevices)

	driverState := (*cache.HealthInfo)[testDriver]
	_, exists := driverState.Devices[testPool+"/timeoutDevice"]
	assert.False(t, exists, "stale device entry should be removed from the cache after the Unknown notification; a missing entry reads as Unknown")
	assert.Equal(t, state.DeviceHealthStatusUnknown, cache.getHealthInfo(testDriver, testPool, "timeoutDevice").Health, "removed device should still read as Unknown")
}

// TestUpdateHealthInfoStaleEntryRemoval verifies that entries which the driver
// stopped reporting are removed from the cache once stale, so that the cache
// stays bounded by what the driver keeps vouching for.
func TestUpdateHealthInfoStaleEntryRemoval(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cache, err := newHealthInfoCache(logger, "")
	require.NoError(t, err)

	healthyDevice := state.DeviceHealth{PoolName: testPool, DeviceName: "healthy-stale", Health: state.DeviceHealthStatusHealthy, HealthCheckTimeout: DefaultHealthTimeout}
	unknownDevice := state.DeviceHealth{PoolName: testPool, DeviceName: "unknown-stale", Health: state.DeviceHealthStatusUnknown, HealthCheckTimeout: DefaultHealthTimeout}
	freshDevice := state.DeviceHealth{PoolName: testPool, DeviceName: "fresh", Health: state.DeviceHealthStatusHealthy, HealthCheckTimeout: DefaultHealthTimeout}
	_, err = cache.updateHealthInfo(logger, testDriver, []state.DeviceHealth{healthyDevice, unknownDevice, freshDevice})
	require.NoError(t, err)

	// Make two of the devices stale.
	err = cache.withLock(func() error {
		for _, deviceName := range []string{"healthy-stale", "unknown-stale"} {
			key := testPool + "/" + deviceName
			device := (*cache.HealthInfo)[testDriver].Devices[key]
			device.LastUpdated = time.Now().Add(-2 * DefaultHealthTimeout)
			(*cache.HealthInfo)[testDriver].Devices[key] = device
		}
		return nil
	})
	require.NoError(t, err)

	// A report which only covers the fresh device removes the stale ones.
	changedDevices, err := cache.updateHealthInfo(logger, testDriver, []state.DeviceHealth{freshDevice})
	require.NoError(t, err)

	// Only the previously healthy device produces an Unknown transition;
	// removing an entry which already was Unknown changes nothing for readers.
	require.Len(t, changedDevices, 1)
	assert.Equal(t, "healthy-stale", changedDevices[0].DeviceName)
	assert.Equal(t, state.DeviceHealthStatusUnknown, changedDevices[0].Health)

	devices := (*cache.HealthInfo)[testDriver].Devices
	assert.NotContains(t, devices, testPool+"/healthy-stale")
	assert.NotContains(t, devices, testPool+"/unknown-stale")
	assert.Contains(t, devices, testPool+"/fresh")
}

// TestUpdateHealthInfoDeviceCap verifies that the cache does not grow beyond
// maxDevicesPerDriver entries per driver even when a plugin keeps reporting
// new device names, while updates for known devices are still accepted.
func TestUpdateHealthInfoDeviceCap(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cache, err := newHealthInfoCache(logger, "")
	require.NoError(t, err)

	// Fill the cache up to the cap, plus one device which must be ignored.
	report := make([]state.DeviceHealth, 0, maxDevicesPerDriver+1)
	for i := range maxDevicesPerDriver + 1 {
		report = append(report, state.DeviceHealth{
			PoolName:           testPool,
			DeviceName:         fmt.Sprintf("device-%d", i),
			Health:             state.DeviceHealthStatusHealthy,
			HealthCheckTimeout: DefaultHealthTimeout,
		})
	}
	changedDevices, err := cache.updateHealthInfo(logger, testDriver, report)
	require.NoError(t, err)
	assert.Len(t, changedDevices, maxDevicesPerDriver)
	assert.Len(t, (*cache.HealthInfo)[testDriver].Devices, maxDevicesPerDriver)

	// New device names are ignored while the cache is full.
	newDevice := state.DeviceHealth{PoolName: testPool, DeviceName: "one-too-many", Health: state.DeviceHealthStatusHealthy, HealthCheckTimeout: DefaultHealthTimeout}
	changedDevices, err = cache.updateHealthInfo(logger, testDriver, []state.DeviceHealth{newDevice})
	require.NoError(t, err)
	assert.Empty(t, changedDevices)
	assert.NotContains(t, (*cache.HealthInfo)[testDriver].Devices, testPool+"/one-too-many")

	// Updates for known devices are still accepted while the cache is full.
	updatedDevice := state.DeviceHealth{PoolName: testPool, DeviceName: "device-0", Health: state.DeviceHealthStatusUnhealthy, HealthCheckTimeout: DefaultHealthTimeout}
	changedDevices, err = cache.updateHealthInfo(logger, testDriver, []state.DeviceHealth{updatedDevice})
	require.NoError(t, err)
	require.Len(t, changedDevices, 1)
	assert.Equal(t, state.DeviceHealthStatusUnhealthy, changedDevices[0].Health)
	assert.Equal(t, state.DeviceHealthStatusUnhealthy, cache.getHealthInfo(testDriver, testPool, "device-0").Health)
	assert.Len(t, (*cache.HealthInfo)[testDriver].Devices, maxDevicesPerDriver)
}

// TestClearDriver tests clearing a driver’s health data.
func TestClearDriver(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cache, err := newHealthInfoCache(logger, "")
	require.NoError(t, err)

	_, err = cache.updateHealthInfo(logger, testDriver, []state.DeviceHealth{testDeviceHealth})
	require.NoError(t, err)
	assert.Equal(t, state.DeviceHealthStatusHealthy, cache.getHealthInfo(testDriver, testPool, testDevice).Health)

	err = cache.clearDriver(logger, testDriver)
	require.NoError(t, err)
	assert.Equal(t, state.DeviceHealthStatusUnknown, cache.getHealthInfo(testDriver, testPool, testDevice).Health)
}
