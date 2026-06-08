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
	"reflect"
	"testing"

	v1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/client-go/informers"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
)

const testDriverName = "test-driver"

type testFixture struct {
	client         *fakeclient.Clientset
	factory        informers.SharedInformerFactory
	driverInformer cache.SharedIndexInformer
	updater        *csiNodeUpdater
	stopCh         chan struct{}
}

func setupTest(t *testing.T, driver *v1.CSIDriver) *testFixture {
	var clientObjects []runtime.Object
	if driver != nil {
		clientObjects = append(clientObjects, driver)
	}

	client := fakeclient.NewSimpleClientset(clientObjects...)
	factory := informers.NewSharedInformerFactory(client, 0)
	driverInformer := factory.Storage().V1().CSIDrivers().Informer()

	stopCh := make(chan struct{})
	factory.Start(stopCh)

	if !cache.WaitForCacheSync(stopCh, driverInformer.HasSynced) {
		close(stopCh)
		t.Fatalf("Timed out waiting for caches to sync")
	}

	updater, err := NewCSINodeUpdater(driverInformer)
	if err != nil {
		close(stopCh)
		t.Fatalf("Failed to create CSINodeUpdater: %v", err)
	}

	return &testFixture{
		client:         client,
		factory:        factory,
		driverInformer: driverInformer,
		updater:        updater,
		stopCh:         stopCh,
	}
}

func TestNewCSINodeUpdater(t *testing.T) {
	t.Run("valid informer", func(t *testing.T) {
		f := setupTest(t, nil)
		defer f.cleanup()

		updater, err := NewCSINodeUpdater(f.driverInformer)

		if err != nil {
			t.Errorf("Expected no error but got: %v", err)
		}
		if updater == nil {
			t.Errorf("Expected non-nil updater")
		}
	})

	t.Run("nil informer", func(t *testing.T) {
		updater, err := NewCSINodeUpdater(nil)

		if err == nil {
			t.Errorf("Expected error but got none")
		}
		if updater != nil {
			t.Errorf("Expected nil updater when error occurs")
		}
	})
}

func TestSyncDriverUpdater(t *testing.T) {
	t.Run("driver not installed, should stop updater", func(t *testing.T) {
		f := setupTest(t, nil)
		defer f.cleanup()

		updaterStopCh := make(chan struct{})
		f.updater.driverUpdaters.Store(testDriverName, updaterStopCh)

		// Verify the initial state - updater should exist for the driver
		verifyUpdaterState(t, f.updater, testDriverName, true)

		// Detect the driver is not installed
		f.updater.syncDriverUpdater(testDriverName)

		// Verify the driver was unregistered from the updater map
		verifyUpdaterState(t, f.updater, testDriverName, false)

		if !isChannelClosed(updaterStopCh) {
			t.Errorf("Stop channel was not closed")
		}
	})

	t.Run("driver not found in informer, should stop updater", func(t *testing.T) {
		f := setupTest(t, nil)
		defer f.cleanup()

		// Register driver in global map but not in informer
		registerTestDriver(testDriverName)
		defer unregisterTestDriver(testDriverName)

		updaterStopCh := make(chan struct{})
		f.updater.driverUpdaters.Store(testDriverName, updaterStopCh)

		verifyUpdaterState(t, f.updater, testDriverName, true)

		// Detect the driver exists in global map but not in informer
		f.updater.syncDriverUpdater(testDriverName)
		verifyUpdaterState(t, f.updater, testDriverName, false)

		if !isChannelClosed(updaterStopCh) {
			t.Errorf("Stop channel was not closed")
		}
	})

	t.Run("driver with unset updatePeriodSeconds, should stop updater", func(t *testing.T) {
		driver := createTestDriver(testDriverName, nil)
		f := setupTest(t, driver)
		defer f.cleanup()

		registerTestDriver(testDriverName)
		defer unregisterTestDriver(testDriverName)

		updaterStopCh := make(chan struct{})
		f.updater.driverUpdaters.Store(testDriverName, updaterStopCh)

		verifyUpdaterState(t, f.updater, testDriverName, true)

		// Detect the driver has unset updatePeriodSeconds
		f.updater.syncDriverUpdater(testDriverName)
		verifyUpdaterState(t, f.updater, testDriverName, false)

		if !isChannelClosed(updaterStopCh) {
			t.Errorf("Stop channel was not closed")
		}
	})

	t.Run("replace existing updater", func(t *testing.T) {
		updatePeriod := int64(60)
		driver := createTestDriver(testDriverName, &updatePeriod)
		f := setupTest(t, driver)
		defer f.cleanup()

		registerTestDriver(testDriverName)
		defer unregisterTestDriver(testDriverName)

		oldStopCh := make(chan struct{})
		f.updater.driverUpdaters.Store(testDriverName, oldStopCh)
		verifyUpdaterState(t, f.updater, testDriverName, true)

		// Replace the old updater with a new one
		f.updater.syncDriverUpdater(testDriverName)

		if !isChannelClosed(oldStopCh) {
			t.Errorf("Previous stop channel was not closed during updater replacement")
		}

		// Verify a new entry was added and is different from old one
		value, exists := f.updater.driverUpdaters.Load(testDriverName)
		if !exists {
			t.Errorf("No updater entry exists after replacement")
		} else {
			newStopCh, ok := value.(chan struct{})
			if !ok {
				t.Errorf("Updated entry is not a channel")
			} else if reflect.ValueOf(newStopCh).Pointer() == reflect.ValueOf(oldStopCh).Pointer() {
				t.Errorf("New stop channel is the same as the old one")
			}

			close(newStopCh)
		}
	})
}

func (f *testFixture) cleanup() {
	close(f.stopCh)
}

func createTestDriver(name string, updatePeriodSeconds *int64) *v1.CSIDriver {
	return &v1.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.CSIDriverSpec{
			NodeAllocatableUpdatePeriodSeconds: updatePeriodSeconds,
		},
	}
}

func registerTestDriver(name string) {
	csiDrivers.Set(name, Driver{
		endpoint:                "test-endpoint",
		highestSupportedVersion: &utilversion.Version{},
	})
}

func unregisterTestDriver(name string) {
	csiDrivers.Delete(name)
}

func isChannelClosed(ch chan struct{}) bool {
	select {
	case <-ch:
		return true
	default:
		return false
	}
}

func verifyUpdaterState(t *testing.T, updater *csiNodeUpdater, driverName string, shouldExist bool) {
	_, exists := updater.driverUpdaters.Load(driverName)
	if shouldExist && !exists {
		t.Errorf("Expected updater for driver %s to exist, but it doesn't", driverName)
	} else if !shouldExist && exists {
		t.Errorf("Expected updater for driver %s to not exist, but it does", driverName)
	}
}
