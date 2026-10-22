/*
Copyright The Kubernetes Authors.

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
	"testing"

	"k8s.io/client-go/tools/cache"
	"k8s.io/utils/ptr"
)

// TestDriverDeleteTombstone verifies that onDriverDelete unwraps a
// cache.DeletedFinalStateUnknown tombstone and stops the periodic updater,
// so a CSIDriver whose delete was delivered as a tombstone does not leak
// its updater goroutine.
func TestDriverDeleteTombstone(t *testing.T) {
	driver := createTestDriver(testDriverName, ptr.To[int64](10))
	f := setupTest(t, driver)
	defer f.cleanup()
	registerTestDriver(testDriverName)
	defer unregisterTestDriver(testDriverName)

	stopCh := make(chan struct{})
	f.updater.driverUpdaters.Store(testDriverName, stopCh)

	// The informer has already dropped the object, as it does on relist.
	if err := f.driverInformer.GetStore().Delete(driver); err != nil {
		t.Fatal(err)
	}
	f.updater.onDriverDelete(cache.DeletedFinalStateUnknown{Key: testDriverName, Obj: driver})

	if _, running := f.updater.driverUpdaters.Load(testDriverName); running {
		t.Errorf("periodic updater for %q still running after its CSIDriver was deleted", testDriverName)
	}
}
