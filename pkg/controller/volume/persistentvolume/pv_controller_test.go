/*
Copyright 2016 The Kubernetes Authors.

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

package persistentvolume

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-helpers/storage/volume"
	csitrans "k8s.io/csi-translation-lib"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
	pvtesting "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume/csimigration"
	"k8s.io/kubernetes/pkg/volume/util"
)

// Test the real controller methods (add/update/delete claim/volume) with
// a fake API server.
// There is no controller API to 'initiate syncAll now', therefore these tests
// can't reliably simulate periodic sync of volumes/claims - it would be
// either very timing-sensitive or slow to wait for real periodic sync.
func TestControllerSync(t *testing.T) {
	// Default enable the HonorPVReclaimPolicy feature gate.
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HonorPVReclaimPolicy, true)()
	tests := []controllerTest{
		// [Unit test set 5] - controller tests.
		// We test the controller as if
		// it was connected to real API server, i.e. we call add/update/delete
		// Claim/Volume methods. Also, all changes to volumes and claims are
		// sent to add/update/delete Claim/Volume as real controller would do.
		{
			// addClaim gets a new claim. Check it's bound to a volume.
			"5-2 - complete bind",
			newVolumeArray("volume5-2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume5-2", "1Gi", "uid5-2", "claim5-2", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			noclaims, /* added in testAddClaim5_2 */
			newClaimArray("claim5-2", "uid5-2", "1Gi", "volume5-2", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			noevents, noerrors,
			// Custom test function that generates an add event
			func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error {
				claim := newClaim("claim5-2", "uid5-2", "1Gi", "", v1.ClaimPending, nil)
				reactor.AddClaimEvent(claim)
				return nil
			},
		},
		{
			"5-2-2 - complete bind when PV and PVC both exist",
			newVolumeArray("volume5-2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume5-2", "1Gi", "uid5-2", "claim5-2", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			newClaimArray("claim5-2", "uid5-2", "1Gi", "", v1.ClaimPending, nil),
			newClaimArray("claim5-2", "uid5-2", "1Gi", "volume5-2", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			noevents, noerrors,
			func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error {
				return nil
			},
		},
		{
			"5-2-3 - complete bind when PV and PVC both exist and PV has AnnPreResizeCapacity annotation",
			volumesWithAnnotation(util.AnnPreResizeCapacity, "1Gi", newVolumeArray("volume5-2", "2Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController)),
			volumesWithAnnotation(util.AnnPreResizeCapacity, "1Gi", newVolumeArray("volume5-2", "2Gi", "uid5-2", "claim5-2", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController)),
			withExpectedCapacity("2Gi", newClaimArray("claim5-2", "uid5-2", "2Gi", "", v1.ClaimPending, nil)),
			withExpectedCapacity("1Gi", newClaimArray("claim5-2", "uid5-2", "2Gi", "volume5-2", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted)),
			noevents, noerrors,
			func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error {
				return nil
			},
		},
		{
			// deleteClaim with a bound claim makes bound volume released.
			"5-3 - delete claim",
			newVolumeArray("volume5-3", "10Gi", "uid5-3", "claim5-3", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			newVolumeArray("volume5-3", "10Gi", "uid5-3", "claim5-3", v1.VolumeReleased, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			newClaimArray("claim5-3", "uid5-3", "1Gi", "volume5-3", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			noclaims,
			noevents, noerrors,
			// Custom test function that generates a delete event
			func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error {
				obj := ctrl.claims.List()[0]
				claim := obj.(*v1.PersistentVolumeClaim)
				reactor.DeleteClaimEvent(claim)
				return nil
			},
		},
		{
			// deleteVolume with a bound volume. Check the claim is Lost.
			"5-4 - delete volume",
			newVolumeArray("volume5-4", "1Gi", "uid5-4", "claim5-4", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			novolumes,
			newClaimArray("claim5-4", "uid5-4", "1Gi", "volume5-4", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			newClaimArray("claim5-4", "uid5-4", "1Gi", "volume5-4", v1.ClaimLost, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			[]string{"Warning ClaimLost"}, noerrors,
			// Custom test function that generates a delete event
			func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error {
				obj := ctrl.volumes.store.List()[0]
				volume := obj.(*v1.PersistentVolume)
				reactor.DeleteVolumeEvent(volume)
				return nil
			},
		},
		{
			// deleteClaim with a bound claim makes bound volume released with external deleter.
			// delete the corresponding volume from apiserver, and report latency metric
			"5-5 - delete claim and delete volume report metric",
			volumesWithAnnotation(volume.AnnDynamicallyProvisioned, "gcr.io/vendor-csi",
				newVolumeArray("volume5-5", "10Gi", "uid5-5", "claim5-5", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classExternal, volume.AnnBoundByController)),
			novolumes,
			claimWithAnnotation(volume.AnnStorageProvisioner, "gcr.io/vendor-csi",
				newClaimArray("claim5-5", "uid5-5", "1Gi", "volume5-5", v1.ClaimBound, &classExternal, volume.AnnBoundByController, volume.AnnBindCompleted)),
			noclaims,
			noevents, noerrors,
			// Custom test function that generates a delete claim event which should have been caught by
			// "deleteClaim" to remove the claim from controller's cache, after that, a volume deleted
			// event will be generated to trigger "deleteVolume" call for metric reporting
			func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error {
				test.initialVolumes[0].Annotations[volume.AnnDynamicallyProvisioned] = "gcr.io/vendor-csi"
				obj := ctrl.claims.List()[0]
				claim := obj.(*v1.PersistentVolumeClaim)
				reactor.DeleteClaimEvent(claim)
				err := wait.Poll(10*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
					return len(ctrl.claims.ListKeys()) == 0, nil
				})
				if err != nil {
					return err
				}
				// claim has been removed from controller's cache, generate a volume deleted event
				volume := ctrl.volumes.store.List()[0].(*v1.PersistentVolume)
				reactor.DeleteVolumeEvent(volume)
				return nil
			},
		},
		{
			// deleteClaim with a bound claim makes bound volume released with external deleter pending
			// there should be an entry in operation timestamps cache in controller
			"5-6 - delete claim and waiting for external volume deletion",
			volumesWithAnnotation(volume.AnnDynamicallyProvisioned, "gcr.io/vendor-csi", []*v1.PersistentVolume{newExternalProvisionedVolume("volume5-6", "10Gi", "uid5-6", "claim5-6", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classExternal, "fake.driver.csi", nil, volume.AnnBoundByController)}),
			volumesWithAnnotation(volume.AnnDynamicallyProvisioned, "gcr.io/vendor-csi", []*v1.PersistentVolume{newExternalProvisionedVolume("volume5-6", "10Gi", "uid5-6", "claim5-6", v1.VolumeReleased, v1.PersistentVolumeReclaimDelete, classExternal, "fake.driver.csi", nil, volume.AnnBoundByController)}),
			claimWithAnnotation(volume.AnnStorageProvisioner, "gcr.io/vendor-csi",
				newClaimArray("claim5-6", "uid5-6", "1Gi", "volume5-6", v1.ClaimBound, &classExternal, volume.AnnBoundByController, volume.AnnBindCompleted)),
			noclaims,
			noevents, noerrors,
			// Custom test function that generates a delete claim event which should have been caught by
			// "deleteClaim" to remove the claim from controller's cache and mark bound volume to be released
			func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error {
				// should have been provisioned by external provisioner
				obj := ctrl.claims.List()[0]
				claim := obj.(*v1.PersistentVolumeClaim)
				reactor.DeleteClaimEvent(claim)
				// wait until claim is cleared from cache, i.e., deleteClaim is called
				err := wait.Poll(10*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
					return len(ctrl.claims.ListKeys()) == 0, nil
				})
				if err != nil {
					return err
				}
				// wait for volume delete operation to appear once volumeWorker() runs
				return wait.PollImmediate(10*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
					// make sure the operation timestamp cache is NOT empty
					if ctrl.operationTimestamps.Has("volume5-6") {
						return true, nil
					}
					t.Logf("missing volume5-6 from timestamp cache, will retry")
					return false, nil
				})
			},
		},
		{
			// deleteVolume event issued before deleteClaim, no metric should have been reported
			// and no delete operation start timestamp should be inserted into controller.operationTimestamps cache
			"5-7 - delete volume event makes claim lost, delete claim event will not report metric",
			newVolumeArray("volume5-7", "10Gi", "uid5-7", "claim5-7", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classExternal, volume.AnnBoundByController, volume.AnnDynamicallyProvisioned),
			novolumes,
			claimWithAnnotation(volume.AnnStorageProvisioner, "gcr.io/vendor-csi",
				newClaimArray("claim5-7", "uid5-7", "1Gi", "volume5-7", v1.ClaimBound, &classExternal, volume.AnnBoundByController, volume.AnnBindCompleted)),
			noclaims,
			[]string{"Warning ClaimLost"},
			noerrors,
			// Custom test function that generates a delete claim event which should have been caught by
			// "deleteClaim" to remove the claim from controller's cache and mark bound volume to be released
			func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error {
				volume := ctrl.volumes.store.List()[0].(*v1.PersistentVolume)
				reactor.DeleteVolumeEvent(volume)
				err := wait.Poll(10*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
					return len(ctrl.volumes.store.ListKeys()) == 0, nil
				})
				if err != nil {
					return err
				}

				// Wait for the PVC to get fully processed. This avoids races between PV controller and DeleteClaimEvent
				// below.
				err = wait.Poll(10*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
					obj := ctrl.claims.List()[0]
					claim := obj.(*v1.PersistentVolumeClaim)
					return claim.Status.Phase == v1.ClaimLost, nil
				})
				if err != nil {
					return err
				}

				// trying to remove the claim as well
				obj := ctrl.claims.List()[0]
				claim := obj.(*v1.PersistentVolumeClaim)
				reactor.DeleteClaimEvent(claim)
				// wait until claim is cleared from cache, i.e., deleteClaim is called
				err = wait.Poll(10*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
					return len(ctrl.claims.ListKeys()) == 0, nil
				})
				if err != nil {
					return err
				}
				// make sure operation timestamp cache is empty
				if ctrl.operationTimestamps.Has("volume5-7") {
					return errors.New("failed checking timestamp cache")
				}
				return nil
			},
		},
		{
			// delete a claim waiting for being bound cleans up provision(volume ref == "") entry from timestamp cache
			"5-8 - delete claim cleans up operation timestamp cache for provision",
			novolumes,
			novolumes,
			claimWithAnnotation(volume.AnnStorageProvisioner, "gcr.io/vendor-csi",
				newClaimArray("claim5-8", "uid5-8", "1Gi", "", v1.ClaimPending, &classExternal)),
			noclaims,
			[]string{"Normal ExternalProvisioning"},
			noerrors,
			// Custom test function that generates a delete claim event which should have been caught by
			// "deleteClaim" to remove the claim from controller's cache and mark bound volume to be released
			func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error {
				// wait until the provision timestamp has been inserted
				err := wait.Poll(10*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
					return ctrl.operationTimestamps.Has("default/claim5-8"), nil
				})
				if err != nil {
					return err
				}
				// delete the claim
				obj := ctrl.claims.List()[0]
				claim := obj.(*v1.PersistentVolumeClaim)
				reactor.DeleteClaimEvent(claim)
				// wait until claim is cleared from cache, i.e., deleteClaim is called
				err = wait.Poll(10*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
					return len(ctrl.claims.ListKeys()) == 0, nil
				})
				if err != nil {
					return err
				}
				// make sure operation timestamp cache is empty
				if ctrl.operationTimestamps.Has("default/claim5-8") {
					return errors.New("failed checking timestamp cache")
				}
				return nil
			},
		},
		{
			// Test that the finalizer gets removed if CSI migration is disabled. The in-tree finalizer is added
			// back on the PV since migration is disabled.
			"5-9 - volume has its external PV deletion protection finalizer removed as CSI migration is disabled",
			volumesWithFinalizers(
				volumesWithAnnotation(volume.AnnMigratedTo, "pd.csi.storage.gke.io",
					newVolumeArray("volume-5-9", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimDelete, classEmpty, volume.AnnDynamicallyProvisioned)),
				[]string{volume.PVDeletionProtectionFinalizer},
			),
			volumesWithFinalizers(newVolumeArray("volume-5-9", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimDelete, classEmpty, volume.AnnDynamicallyProvisioned), []string{volume.PVDeletionInTreeProtectionFinalizer}),
			noclaims,
			noclaims,
			noevents,
			noerrors,
			func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error {
				return nil
			},
		},
	}

	doit := func(test controllerTest) {
		// Initialize the controller
		client := &fake.Clientset{}

		fakeVolumeWatch := watch.NewFake()
		client.PrependWatchReactor("persistentvolumes", core.DefaultWatchReactor(fakeVolumeWatch, nil))
		fakeClaimWatch := watch.NewFake()
		client.PrependWatchReactor("persistentvolumeclaims", core.DefaultWatchReactor(fakeClaimWatch, nil))
		client.PrependWatchReactor("storageclasses", core.DefaultWatchReactor(watch.NewFake(), nil))
		client.PrependWatchReactor("nodes", core.DefaultWatchReactor(watch.NewFake(), nil))
		client.PrependWatchReactor("pods", core.DefaultWatchReactor(watch.NewFake(), nil))

		informers := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
		ctrl, err := newTestController(client, informers, true)
		if err != nil {
			t.Fatalf("Test %q construct persistent volume failed: %v", test.name, err)
		}

		// Inject storage classes into controller via a custom lister for test [5-5]
		storageClasses := []*storagev1.StorageClass{
			makeStorageClass(classExternal, &modeImmediate),
		}

		storageClasses[0].Provisioner = "gcr.io/vendor-csi"
		indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
		for _, class := range storageClasses {
			indexer.Add(class)
		}
		ctrl.classLister = storagelisters.NewStorageClassLister(indexer)

		reactor := newVolumeReactor(client, ctrl, fakeVolumeWatch, fakeClaimWatch, test.errors)
		for _, claim := range test.initialClaims {
			claim = claim.DeepCopy()
			reactor.AddClaim(claim)
			go func(claim *v1.PersistentVolumeClaim) {
				fakeClaimWatch.Add(claim)
			}(claim)
		}
		for _, volume := range test.initialVolumes {
			volume = volume.DeepCopy()
			reactor.AddVolume(volume)
			go func(volume *v1.PersistentVolume) {
				fakeVolumeWatch.Add(volume)
			}(volume)
		}

		// Start the controller
		ctx, cancel := context.WithCancel(context.TODO())
		informers.Start(ctx.Done())
		informers.WaitForCacheSync(ctx.Done())
		go ctrl.Run(ctx)

		// Wait for the controller to pass initial sync and fill its caches.
		err = wait.Poll(10*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
			return len(ctrl.claims.ListKeys()) >= len(test.initialClaims) &&
				len(ctrl.volumes.store.ListKeys()) >= len(test.initialVolumes), nil
		})
		if err != nil {
			t.Errorf("Test %q controller sync failed: %v", test.name, err)
		}
		klog.V(4).Infof("controller synced, starting test")

		// Call the tested function
		err = test.test(ctrl, reactor.VolumeReactor, test)
		if err != nil {
			t.Errorf("Test %q initial test call failed: %v", test.name, err)
		}
		// Simulate a periodic resync, just in case some events arrived in a
		// wrong order.
		ctrl.resync()

		err = reactor.waitTest(test)
		if err != nil {
			t.Errorf("Failed to run test %s: %v", test.name, err)
		}
		cancel()

		evaluateTestResults(ctrl, reactor.VolumeReactor, test, t)
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			doit(test)
		})
	}
}

func storeVersion(t *testing.T, prefix string, c cache.Store, version string, expectedReturn bool) {
	pv := newVolume("pvName", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimDelete, classEmpty)
	pv.ResourceVersion = version
	ret, err := storeObjectUpdate(c, pv, "volume")
	if err != nil {
		t.Errorf("%s: expected storeObjectUpdate to succeed, got: %v", prefix, err)
	}
	if expectedReturn != ret {
		t.Errorf("%s: expected storeObjectUpdate to return %v, got: %v", prefix, expectedReturn, ret)
	}

	// find the stored version

	pvObj, found, err := c.GetByKey("pvName")
	if err != nil {
		t.Errorf("expected volume 'pvName' in the cache, got error instead: %v", err)
	}
	if !found {
		t.Errorf("expected volume 'pvName' in the cache but it was not found")
	}
	pv, ok := pvObj.(*v1.PersistentVolume)
	if !ok {
		t.Errorf("expected volume in the cache, got different object instead: %#v", pvObj)
	}

	if ret {
		if pv.ResourceVersion != version {
			t.Errorf("expected volume with version %s in the cache, got %s instead", version, pv.ResourceVersion)
		}
	} else {
		if pv.ResourceVersion == version {
			t.Errorf("expected volume with version other than %s in the cache, got %s instead", version, pv.ResourceVersion)
		}
	}
}

// TestControllerCache tests func storeObjectUpdate()
func TestControllerCache(t *testing.T) {
	// Cache under test
	c := cache.NewStore(cache.DeletionHandlingMetaNamespaceKeyFunc)

	// Store new PV
	storeVersion(t, "Step1", c, "1", true)
	// Store the same PV
	storeVersion(t, "Step2", c, "1", true)
	// Store newer PV
	storeVersion(t, "Step3", c, "2", true)
	// Store older PV - simulating old "PV updated" event or periodic sync with
	// old data
	storeVersion(t, "Step4", c, "1", false)
	// Store newer PV - test integer parsing ("2" > "10" as string,
	// while 2 < 10 as integers)
	storeVersion(t, "Step5", c, "10", true)
}

func TestControllerCacheParsingError(t *testing.T) {
	c := cache.NewStore(cache.DeletionHandlingMetaNamespaceKeyFunc)
	// There must be something in the cache to compare with
	storeVersion(t, "Step1", c, "1", true)

	pv := newVolume("pvName", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimDelete, classEmpty)
	pv.ResourceVersion = "xxx"
	_, err := storeObjectUpdate(c, pv, "volume")
	if err == nil {
		t.Errorf("Expected parsing error, got nil instead")
	}
}

func makeStorageClass(scName string, mode *storagev1.VolumeBindingMode) *storagev1.StorageClass {
	return &storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: scName,
		},
		VolumeBindingMode: mode,
	}
}

func TestAnnealMigrationAnnotations(t *testing.T) {
	const testPlugin = "non-migrated-plugin"
	const gcePlugin = "kubernetes.io/gce-pd"
	const gceDriver = "pd.csi.storage.gke.io"
	tests := []struct {
		name                 string
		volumeAnnotations    map[string]string
		expVolumeAnnotations map[string]string
		claimAnnotations     map[string]string
		expClaimAnnotations  map[string]string
		migratedDriverGates  []featuregate.Feature
		disabledDriverGates  []featuregate.Feature
	}{
		{
			name:                 "migration on for GCE",
			volumeAnnotations:    map[string]string{volume.AnnDynamicallyProvisioned: gcePlugin},
			expVolumeAnnotations: map[string]string{volume.AnnDynamicallyProvisioned: gcePlugin, volume.AnnMigratedTo: gceDriver},
			claimAnnotations:     map[string]string{volume.AnnStorageProvisioner: gcePlugin},
			expClaimAnnotations:  map[string]string{volume.AnnStorageProvisioner: gcePlugin, volume.AnnMigratedTo: gceDriver},
			migratedDriverGates:  []featuregate.Feature{features.CSIMigrationGCE},
			disabledDriverGates:  []featuregate.Feature{},
		},
		{
			name:                 "migration on for GCE with Beta storage provisioner annontation",
			volumeAnnotations:    map[string]string{volume.AnnDynamicallyProvisioned: gcePlugin},
			expVolumeAnnotations: map[string]string{volume.AnnDynamicallyProvisioned: gcePlugin, volume.AnnMigratedTo: gceDriver},
			claimAnnotations:     map[string]string{volume.AnnBetaStorageProvisioner: gcePlugin},
			expClaimAnnotations:  map[string]string{volume.AnnBetaStorageProvisioner: gcePlugin, volume.AnnMigratedTo: gceDriver},
			migratedDriverGates:  []featuregate.Feature{features.CSIMigrationGCE},
			disabledDriverGates:  []featuregate.Feature{},
		},
		{
			name:                 "migration off for GCE",
			volumeAnnotations:    map[string]string{volume.AnnDynamicallyProvisioned: gcePlugin},
			expVolumeAnnotations: map[string]string{volume.AnnDynamicallyProvisioned: gcePlugin},
			claimAnnotations:     map[string]string{volume.AnnStorageProvisioner: gcePlugin},
			expClaimAnnotations:  map[string]string{volume.AnnStorageProvisioner: gcePlugin},
			migratedDriverGates:  []featuregate.Feature{},
			disabledDriverGates:  []featuregate.Feature{features.CSIMigrationGCE},
		},
		{
			name:                 "migration off for GCE removes migrated to (rollback)",
			volumeAnnotations:    map[string]string{volume.AnnDynamicallyProvisioned: gcePlugin, volume.AnnMigratedTo: gceDriver},
			expVolumeAnnotations: map[string]string{volume.AnnDynamicallyProvisioned: gcePlugin},
			claimAnnotations:     map[string]string{volume.AnnStorageProvisioner: gcePlugin, volume.AnnMigratedTo: gceDriver},
			expClaimAnnotations:  map[string]string{volume.AnnStorageProvisioner: gcePlugin},
			migratedDriverGates:  []featuregate.Feature{},
			disabledDriverGates:  []featuregate.Feature{features.CSIMigrationGCE},
		},
		{
			name:                 "migration off for GCE removes migrated to (rollback) with Beta storage provisioner annontation",
			volumeAnnotations:    map[string]string{volume.AnnDynamicallyProvisioned: gcePlugin, volume.AnnMigratedTo: gceDriver},
			expVolumeAnnotations: map[string]string{volume.AnnDynamicallyProvisioned: gcePlugin},
			claimAnnotations:     map[string]string{volume.AnnBetaStorageProvisioner: gcePlugin, volume.AnnMigratedTo: gceDriver},
			expClaimAnnotations:  map[string]string{volume.AnnBetaStorageProvisioner: gcePlugin},
			migratedDriverGates:  []featuregate.Feature{},
			disabledDriverGates:  []featuregate.Feature{features.CSIMigrationGCE},
		},
		{
			name:                 "migration on for GCE other plugin not affected",
			volumeAnnotations:    map[string]string{volume.AnnDynamicallyProvisioned: testPlugin},
			expVolumeAnnotations: map[string]string{volume.AnnDynamicallyProvisioned: testPlugin},
			claimAnnotations:     map[string]string{volume.AnnStorageProvisioner: testPlugin},
			expClaimAnnotations:  map[string]string{volume.AnnStorageProvisioner: testPlugin},
			migratedDriverGates:  []featuregate.Feature{features.CSIMigrationGCE},
			disabledDriverGates:  []featuregate.Feature{},
		},
		{
			name:                 "not dynamically provisioned migration off for GCE",
			volumeAnnotations:    map[string]string{},
			expVolumeAnnotations: map[string]string{},
			claimAnnotations:     map[string]string{},
			expClaimAnnotations:  map[string]string{},
			migratedDriverGates:  []featuregate.Feature{},
			disabledDriverGates:  []featuregate.Feature{features.CSIMigrationGCE},
		},
		{
			name:                 "not dynamically provisioned migration on for GCE",
			volumeAnnotations:    map[string]string{},
			expVolumeAnnotations: map[string]string{},
			claimAnnotations:     map[string]string{},
			expClaimAnnotations:  map[string]string{},
			migratedDriverGates:  []featuregate.Feature{features.CSIMigrationGCE},
			disabledDriverGates:  []featuregate.Feature{},
		},
		{
			name:                 "nil annotations migration off for GCE",
			volumeAnnotations:    nil,
			expVolumeAnnotations: nil,
			claimAnnotations:     nil,
			expClaimAnnotations:  nil,
			migratedDriverGates:  []featuregate.Feature{},
			disabledDriverGates:  []featuregate.Feature{features.CSIMigrationGCE},
		},
		{
			name:                 "nil annotations migration on for GCE",
			volumeAnnotations:    nil,
			expVolumeAnnotations: nil,
			claimAnnotations:     nil,
			expClaimAnnotations:  nil,
			migratedDriverGates:  []featuregate.Feature{features.CSIMigrationGCE},
			disabledDriverGates:  []featuregate.Feature{},
		},
	}

	translator := csitrans.New()
	cmpm := csimigration.NewPluginManager(translator, utilfeature.DefaultFeatureGate)

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			for _, f := range tc.migratedDriverGates {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, f, true)()
			}
			for _, f := range tc.disabledDriverGates {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, f, false)()
			}
			if tc.volumeAnnotations != nil {
				ann := tc.volumeAnnotations
				updateMigrationAnnotations(cmpm, translator, ann, false)
				if !reflect.DeepEqual(tc.expVolumeAnnotations, ann) {
					t.Errorf("got volume annoations: %v, but expected: %v", ann, tc.expVolumeAnnotations)
				}
			}
			if tc.claimAnnotations != nil {
				ann := tc.claimAnnotations
				updateMigrationAnnotations(cmpm, translator, ann, true)
				if !reflect.DeepEqual(tc.expClaimAnnotations, ann) {
					t.Errorf("got volume annoations: %v, but expected: %v", ann, tc.expVolumeAnnotations)
				}
			}

		})
	}
}

func TestModifyDeletionFinalizers(t *testing.T) {
	// This set of tests ensures that protection finalizer is removed when CSI migration is disabled
	// and PV controller needs to remove finalizers added by the external-provisioner.
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigrationGCE, false)()
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HonorPVReclaimPolicy, true)()
	const gcePlugin = "kubernetes.io/gce-pd"
	const gceDriver = "pd.csi.storage.gke.io"
	const customFinalizer = "test.volume.kubernetes.io/finalizer"
	tests := []struct {
		name                string
		initialVolume       *v1.PersistentVolume
		volumeAnnotations   map[string]string
		expVolumeFinalizers []string
		expModified         bool
		migratedDriverGates []featuregate.Feature
	}{
		{
			// Represents a CSI volume provisioned through external-provisioner, no CSI migration enabled.
			name:                "13-1 migration was never enabled, volume has the finalizer",
			initialVolume:       newExternalProvisionedVolume("volume-13-1", "1Gi", "uid11-23", "claim11-23", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classCopper, gceDriver, []string{volume.PVDeletionProtectionFinalizer}, volume.AnnDynamicallyProvisioned, volume.AnnBoundByController),
			expVolumeFinalizers: []string{volume.PVDeletionProtectionFinalizer},
			expModified:         false,
			migratedDriverGates: []featuregate.Feature{},
		},
		{
			// Represents a volume provisioned through external-provisioner but the external-provisioner has
			// yet to sync the volume to add the new finalizer
			name:                "13-2 migration was never enabled, volume does not have the finalizer",
			initialVolume:       newExternalProvisionedVolume("volume-13-2", "1Gi", "uid11-23", "claim11-23", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classCopper, gceDriver, nil, volume.AnnDynamicallyProvisioned, volume.AnnBoundByController),
			expVolumeFinalizers: nil,
			expModified:         false,
			migratedDriverGates: []featuregate.Feature{},
		},
		{
			// Represents an in-tree volume that has the migrated-to annotation but the external-provisioner is
			// yet to sync the volume and add the pv deletion protection finalizer. The custom finalizer is some
			// pre-existing finalizer, for example the pv-protection finalizer. When csi-migration is disabled,
			// the migrated-to annotation will be removed shortly when updateVolumeMigrationAnnotationsAndFinalizers
			// is called followed by adding back the in-tree pv protection finalizer.
			name:                "13-3 migration was disabled, volume has existing custom finalizer, does not have in-tree pv deletion protection finalizer",
			initialVolume:       newVolumeWithFinalizers("volume-13-3", "1Gi", "uid11-23", "claim11-23", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classCopper, []string{customFinalizer}, volume.AnnDynamicallyProvisioned, volume.AnnBoundByController),
			expVolumeFinalizers: []string{customFinalizer, volume.PVDeletionInTreeProtectionFinalizer},
			expModified:         true,
			migratedDriverGates: []featuregate.Feature{},
		},
		{
			name:                "13-4 migration was disabled, volume has no finalizers",
			initialVolume:       newVolumeWithFinalizers("volume-13-4", "1Gi", "uid11-23", "claim11-23", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classCopper, nil, volume.AnnDynamicallyProvisioned, volume.AnnBoundByController),
			expVolumeFinalizers: []string{volume.PVDeletionInTreeProtectionFinalizer},
			expModified:         true,
			migratedDriverGates: []featuregate.Feature{},
		},
		{
			// Represents roll back scenario where the external-provisioner has added the pv deletion protection
			// finalizer and later the csi migration was disabled. The pv deletion protection finalizer added through
			// external-provisioner will be removed and the in-tree pv deletion protection finalizer will be added.
			name:                "13-5 migration was disabled, volume has external PV deletion finalizer",
			initialVolume:       newVolumeWithFinalizers("volume-13-5", "1Gi", "uid11-23", "claim11-23", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classCopper, []string{volume.PVDeletionProtectionFinalizer}, volume.AnnDynamicallyProvisioned, volume.AnnBoundByController),
			expVolumeFinalizers: []string{volume.PVDeletionInTreeProtectionFinalizer},
			expModified:         true,
			migratedDriverGates: []featuregate.Feature{},
		},
		{
			// Represents roll-back of csi-migration as 13-5, here there are multiple finalizers, only the pv deletion
			// protection finalizer added by external-provisioner will be removed and the in-tree pv deletion protection
			// finalizer will be added.
			name:                "13-6 migration was disabled, volume has multiple finalizers",
			initialVolume:       newVolumeWithFinalizers("volume-13-6", "1Gi", "uid11-23", "claim11-23", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classCopper, []string{volume.PVDeletionProtectionFinalizer, customFinalizer}, volume.AnnDynamicallyProvisioned, volume.AnnBoundByController),
			expVolumeFinalizers: []string{customFinalizer, volume.PVDeletionInTreeProtectionFinalizer},
			expModified:         true,
			migratedDriverGates: []featuregate.Feature{},
		},
		{
			// csi migration is enabled, the pv controller should not delete the finalizer added by the
			// external-provisioner and the in-tree finalizer should be deleted.
			name:                "13-7 migration is enabled, volume has both the in-tree and external PV deletion protection finalizer",
			initialVolume:       newVolumeWithFinalizers("volume-13-7", "1Gi", "uid11-23", "claim11-23", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classCopper, []string{volume.PVDeletionProtectionFinalizer, volume.PVDeletionInTreeProtectionFinalizer}, volume.AnnDynamicallyProvisioned, volume.AnnBoundByController),
			volumeAnnotations:   map[string]string{volume.AnnDynamicallyProvisioned: gcePlugin, volume.AnnMigratedTo: gceDriver},
			expVolumeFinalizers: []string{volume.PVDeletionProtectionFinalizer},
			expModified:         true,
			migratedDriverGates: []featuregate.Feature{features.CSIMigrationGCE},
		},
		{
			// csi-migration is not completely enabled as the specific plugin feature is not present. This is equivalent
			// of disabled csi-migration.
			name:                "13-8 migration is enabled but plugin migration feature is disabled, volume has the external PV deletion protection finalizer",
			initialVolume:       newVolumeWithFinalizers("volume-13-8", "1Gi", "uid11-23", "claim11-23", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classCopper, []string{volume.PVDeletionProtectionFinalizer}, volume.AnnDynamicallyProvisioned, volume.AnnBoundByController),
			expVolumeFinalizers: []string{volume.PVDeletionInTreeProtectionFinalizer},
			expModified:         true,
			migratedDriverGates: []featuregate.Feature{},
		},
		{
			// same as 13-8 but multiple finalizers exists, only the pv deletion protection finalizer needs to be
			// removed and the in-tree pv deletion protection finalizer needs to be added.
			name:                "13-9 migration is enabled but plugin migration feature is disabled, volume has multiple finalizers including external PV deletion protection finalizer",
			initialVolume:       newVolumeWithFinalizers("volume-13-9", "1Gi", "uid11-23", "claim11-23", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classCopper, []string{volume.PVDeletionProtectionFinalizer, customFinalizer}, volume.AnnDynamicallyProvisioned, volume.AnnBoundByController),
			expVolumeFinalizers: []string{customFinalizer, volume.PVDeletionInTreeProtectionFinalizer},
			expModified:         true,
			migratedDriverGates: []featuregate.Feature{},
		},
		{
			// corner error case.
			name:                "13-10 missing annotations but finalizers exist",
			initialVolume:       newVolumeWithFinalizers("volume-13-10", "1Gi", "uid11-23", "claim11-23", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classCopper, []string{volume.PVDeletionProtectionFinalizer}),
			expVolumeFinalizers: []string{volume.PVDeletionProtectionFinalizer},
			expModified:         false,
			migratedDriverGates: []featuregate.Feature{},
		},
		{
			name:                "13-11 missing annotations and finalizers",
			initialVolume:       newVolumeWithFinalizers("volume-13-11", "1Gi", "uid11-23", "claim11-23", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classCopper, nil),
			expVolumeFinalizers: nil,
			expModified:         false,
			migratedDriverGates: []featuregate.Feature{},
		},
		{
			// When ReclaimPolicy is Retain ensure that in-tree pv deletion protection finalizer is not added.
			name:                "13-12 migration is disabled, volume has no finalizers, reclaimPolicy is Retain",
			initialVolume:       newVolumeWithFinalizers("volume-13-12", "1Gi", "uid11-23", "claim11-23", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classCopper, nil, volume.AnnDynamicallyProvisioned, volume.AnnBoundByController),
			expVolumeFinalizers: nil,
			expModified:         false,
			migratedDriverGates: []featuregate.Feature{},
		},
		{
			// When ReclaimPolicy is Recycle ensure that in-tree pv deletion protection finalizer is not added.
			name:                "13-13 migration is disabled, volume has no finalizers, reclaimPolicy is Recycle",
			initialVolume:       newVolumeWithFinalizers("volume-13-13", "1Gi", "uid11-23", "claim11-23", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle, classCopper, nil, volume.AnnDynamicallyProvisioned, volume.AnnBoundByController),
			expVolumeFinalizers: nil,
			expModified:         false,
			migratedDriverGates: []featuregate.Feature{},
		},
		{
			// When ReclaimPolicy is Retain ensure that in-tree pv deletion protection finalizer present is removed.
			name:                "13-14 migration is disabled, volume has in-tree pv deletion finalizers, reclaimPolicy is Retain",
			initialVolume:       newVolumeWithFinalizers("volume-13-14", "1Gi", "uid11-23", "claim11-23", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classCopper, []string{volume.PVDeletionInTreeProtectionFinalizer}, volume.AnnDynamicallyProvisioned, volume.AnnBoundByController),
			expVolumeFinalizers: nil,
			expModified:         true,
			migratedDriverGates: []featuregate.Feature{},
		},
		{
			// Statically provisioned volumes should not have the in-tree pv deletion protection finalizer
			name:                "13-15 migration is disabled, statically provisioned PV",
			initialVolume:       newVolumeWithFinalizers("volume-13-14", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimDelete, classCopper, nil),
			expVolumeFinalizers: nil,
			expModified:         false,
			migratedDriverGates: []featuregate.Feature{},
		},
	}

	translator := csitrans.New()
	cmpm := csimigration.NewPluginManager(translator, utilfeature.DefaultFeatureGate)

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			for _, f := range tc.migratedDriverGates {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, f, true)()
			}
			if tc.volumeAnnotations != nil {
				tc.initialVolume.SetAnnotations(tc.volumeAnnotations)
			}
			modifiedFinalizers, modified := modifyDeletionFinalizers(cmpm, tc.initialVolume)
			if modified != tc.expModified {
				t.Errorf("got modified: %v, but expected: %v", modified, tc.expModified)
			}
			if !reflect.DeepEqual(tc.expVolumeFinalizers, modifiedFinalizers) {
				t.Errorf("got volume finaliers: %v, but expected: %v", modifiedFinalizers, tc.expVolumeFinalizers)
			}

		})
	}
}
