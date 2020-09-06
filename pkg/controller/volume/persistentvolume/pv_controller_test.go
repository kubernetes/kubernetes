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
	csitrans "k8s.io/csi-translation-lib"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
	pvtesting "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/testing"
	pvutil "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/util"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume/csimigration"
)

var (
	classNotHere       = "not-here"
	classNoMode        = "no-mode"
	classImmediateMode = "immediate-mode"
	classWaitMode      = "wait-mode"
)

// Test the real controller methods (add/update/delete claim/volume) with
// a fake API server.
// There is no controller API to 'initiate syncAll now', therefore these tests
// can't reliably simulate periodic sync of volumes/claims - it would be
// either very timing-sensitive or slow to wait for real periodic sync.
func TestControllerSync(t *testing.T) {
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
			newVolumeArray("volume5-2", "1Gi", "uid5-2", "claim5-2", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			noclaims, /* added in testAddClaim5_2 */
			newClaimArray("claim5-2", "uid5-2", "1Gi", "volume5-2", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			noevents, noerrors,
			// Custom test function that generates an add event
			func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error {
				claim := newClaim("claim5-2", "uid5-2", "1Gi", "", v1.ClaimPending, nil)
				reactor.AddClaimEvent(claim)
				return nil
			},
		},
		{
			// deleteClaim with a bound claim makes bound volume released.
			"5-3 - delete claim",
			newVolumeArray("volume5-3", "10Gi", "uid5-3", "claim5-3", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			newVolumeArray("volume5-3", "10Gi", "uid5-3", "claim5-3", v1.VolumeReleased, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			newClaimArray("claim5-3", "uid5-3", "1Gi", "volume5-3", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
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
			newClaimArray("claim5-4", "uid5-4", "1Gi", "volume5-4", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			newClaimArray("claim5-4", "uid5-4", "1Gi", "volume5-4", v1.ClaimLost, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
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
			volumesWithAnnotation(pvutil.AnnDynamicallyProvisioned, "gcr.io/vendor-csi",
				newVolumeArray("volume5-6", "10Gi", "uid5-6", "claim5-6", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classExternal, pvutil.AnnBoundByController)),
			novolumes,
			claimWithAnnotation(pvutil.AnnStorageProvisioner, "gcr.io/vendor-csi",
				newClaimArray("claim5-5", "uid5-5", "1Gi", "volume5-5", v1.ClaimBound, &classExternal, pvutil.AnnBoundByController, pvutil.AnnBindCompleted)),
			noclaims,
			noevents, noerrors,
			// Custom test function that generates a delete claim event which should have been caught by
			// "deleteClaim" to remove the claim from controller's cache, after that, a volume deleted
			// event will be generated to trigger "deleteVolume" call for metric reporting
			func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error {
				test.initialVolumes[0].Annotations[pvutil.AnnDynamicallyProvisioned] = "gcr.io/vendor-csi"
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
			volumesWithAnnotation(pvutil.AnnDynamicallyProvisioned, "gcr.io/vendor-csi",
				newVolumeArray("volume5-6", "10Gi", "uid5-6", "claim5-6", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classExternal, pvutil.AnnBoundByController)),
			volumesWithAnnotation(pvutil.AnnDynamicallyProvisioned, "gcr.io/vendor-csi",
				newVolumeArray("volume5-6", "10Gi", "uid5-6", "claim5-6", v1.VolumeReleased, v1.PersistentVolumeReclaimDelete, classExternal, pvutil.AnnBoundByController)),
			claimWithAnnotation(pvutil.AnnStorageProvisioner, "gcr.io/vendor-csi",
				newClaimArray("claim5-6", "uid5-6", "1Gi", "volume5-6", v1.ClaimBound, &classExternal, pvutil.AnnBoundByController, pvutil.AnnBindCompleted)),
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
			newVolumeArray("volume5-7", "10Gi", "uid5-7", "claim5-7", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classExternal, pvutil.AnnBoundByController, pvutil.AnnDynamicallyProvisioned),
			novolumes,
			claimWithAnnotation(pvutil.AnnStorageProvisioner, "gcr.io/vendor-csi",
				newClaimArray("claim5-7", "uid5-7", "1Gi", "volume5-7", v1.ClaimBound, &classExternal, pvutil.AnnBoundByController, pvutil.AnnBindCompleted)),
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
			claimWithAnnotation(pvutil.AnnStorageProvisioner, "gcr.io/vendor-csi",
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
			// delete success(?) - volume has deletion timestamp before doDelete() starts
			"8-13 - volume is has deletion timestamp and is not processed",
			withVolumeDeletionTimestamp(newVolumeArray("volume8-13", "1Gi", "uid8-13", "claim8-13", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classEmpty)),
			withVolumeDeletionTimestamp(newVolumeArray("volume8-13", "1Gi", "uid8-13", "claim8-13", v1.VolumeReleased, v1.PersistentVolumeReclaimDelete, classEmpty)),
			noclaims,
			noclaims,
			noevents, noerrors,
			// We don't need to do anything in test function because deletion will be noticed automatically and synced.
			// Attempting to use testSyncVolume here will cause an error because of race condition between manually
			// calling testSyncVolume and volume loop running.
			func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error {
				return nil
			},
		},
	}

	for _, test := range tests {
		klog.V(4).Infof("starting test %q", test.name)

		// Initialize the controller
		client := &fake.Clientset{}

		fakeVolumeWatch := watch.NewFake()
		client.PrependWatchReactor("persistentvolumes", core.DefaultWatchReactor(fakeVolumeWatch, nil))
		fakeClaimWatch := watch.NewFake()
		client.PrependWatchReactor("persistentvolumeclaims", core.DefaultWatchReactor(fakeClaimWatch, nil))
		client.PrependWatchReactor("storageclasses", core.DefaultWatchReactor(watch.NewFake(), nil))

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
		stopCh := make(chan struct{})
		informers.Start(stopCh)
		go ctrl.Run(stopCh)

		// Wait for the controller to pass initial sync and fill its caches.
		err = wait.Poll(10*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
			return ctrl.volumeListerSynced() &&
				ctrl.claimListerSynced() &&
				len(ctrl.claims.ListKeys()) >= len(test.initialClaims) &&
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
		close(stopCh)

		evaluateTestResults(ctrl, reactor.VolumeReactor, test, t)
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

func makePVCClass(scName *string) *v1.PersistentVolumeClaim {
	claim := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{},
		},
		Spec: v1.PersistentVolumeClaimSpec{
			StorageClassName: scName,
		},
	}

	return claim
}

func makeStorageClass(scName string, mode *storagev1.VolumeBindingMode) *storagev1.StorageClass {
	return &storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: scName,
		},
		VolumeBindingMode: mode,
	}
}

func TestDelayBindingMode(t *testing.T) {
	tests := map[string]struct {
		pvc         *v1.PersistentVolumeClaim
		shouldDelay bool
		shouldFail  bool
	}{
		"nil-class": {
			pvc:         makePVCClass(nil),
			shouldDelay: false,
		},
		"class-not-found": {
			pvc:         makePVCClass(&classNotHere),
			shouldDelay: false,
		},
		"no-mode-class": {
			pvc:         makePVCClass(&classNoMode),
			shouldDelay: false,
			shouldFail:  true,
		},
		"immediate-mode-class": {
			pvc:         makePVCClass(&classImmediateMode),
			shouldDelay: false,
		},
		"wait-mode-class": {
			pvc:         makePVCClass(&classWaitMode),
			shouldDelay: true,
		},
	}

	classes := []*storagev1.StorageClass{
		makeStorageClass(classNoMode, nil),
		makeStorageClass(classImmediateMode, &modeImmediate),
		makeStorageClass(classWaitMode, &modeWait),
	}

	client := &fake.Clientset{}
	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	classInformer := informerFactory.Storage().V1().StorageClasses()
	ctrl := &PersistentVolumeController{
		classLister: classInformer.Lister(),
		translator:  csitrans.New(),
	}

	for _, class := range classes {
		if err := classInformer.Informer().GetIndexer().Add(class); err != nil {
			t.Fatalf("Failed to add storage class %q: %v", class.Name, err)
		}
	}

	for name, test := range tests {
		shouldDelay, err := pvutil.IsDelayBindingMode(test.pvc, ctrl.classLister)
		if err != nil && !test.shouldFail {
			t.Errorf("Test %q returned error: %v", name, err)
		}
		if err == nil && test.shouldFail {
			t.Errorf("Test %q returned success, expected error", name)
		}
		if shouldDelay != test.shouldDelay {
			t.Errorf("Test %q returned unexpected %v", name, test.shouldDelay)
		}
	}
}

func TestAnnealMigrationAnnotations(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigration, true)()

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
	}{
		{
			name:                 "migration on for GCE",
			volumeAnnotations:    map[string]string{pvutil.AnnDynamicallyProvisioned: gcePlugin},
			expVolumeAnnotations: map[string]string{pvutil.AnnDynamicallyProvisioned: gcePlugin, pvutil.AnnMigratedTo: gceDriver},
			claimAnnotations:     map[string]string{pvutil.AnnStorageProvisioner: gcePlugin},
			expClaimAnnotations:  map[string]string{pvutil.AnnStorageProvisioner: gcePlugin, pvutil.AnnMigratedTo: gceDriver},
			migratedDriverGates:  []featuregate.Feature{features.CSIMigrationGCE},
		},
		{
			name:                 "migration off for GCE",
			volumeAnnotations:    map[string]string{pvutil.AnnDynamicallyProvisioned: gcePlugin},
			expVolumeAnnotations: map[string]string{pvutil.AnnDynamicallyProvisioned: gcePlugin},
			claimAnnotations:     map[string]string{pvutil.AnnStorageProvisioner: gcePlugin},
			expClaimAnnotations:  map[string]string{pvutil.AnnStorageProvisioner: gcePlugin},
			migratedDriverGates:  []featuregate.Feature{},
		},
		{
			name:                 "migration off for GCE removes migrated to (rollback)",
			volumeAnnotations:    map[string]string{pvutil.AnnDynamicallyProvisioned: gcePlugin, pvutil.AnnMigratedTo: gceDriver},
			expVolumeAnnotations: map[string]string{pvutil.AnnDynamicallyProvisioned: gcePlugin},
			claimAnnotations:     map[string]string{pvutil.AnnStorageProvisioner: gcePlugin, pvutil.AnnMigratedTo: gceDriver},
			expClaimAnnotations:  map[string]string{pvutil.AnnStorageProvisioner: gcePlugin},
			migratedDriverGates:  []featuregate.Feature{},
		},
		{
			name:                 "migration on for GCE other plugin not affected",
			volumeAnnotations:    map[string]string{pvutil.AnnDynamicallyProvisioned: testPlugin},
			expVolumeAnnotations: map[string]string{pvutil.AnnDynamicallyProvisioned: testPlugin},
			claimAnnotations:     map[string]string{pvutil.AnnStorageProvisioner: testPlugin},
			expClaimAnnotations:  map[string]string{pvutil.AnnStorageProvisioner: testPlugin},
			migratedDriverGates:  []featuregate.Feature{features.CSIMigrationGCE},
		},
		{
			name:                 "not dynamically provisioned migration off for GCE",
			volumeAnnotations:    map[string]string{},
			expVolumeAnnotations: map[string]string{},
			claimAnnotations:     map[string]string{},
			expClaimAnnotations:  map[string]string{},
			migratedDriverGates:  []featuregate.Feature{},
		},
		{
			name:                 "not dynamically provisioned migration on for GCE",
			volumeAnnotations:    map[string]string{},
			expVolumeAnnotations: map[string]string{},
			claimAnnotations:     map[string]string{},
			expClaimAnnotations:  map[string]string{},
			migratedDriverGates:  []featuregate.Feature{features.CSIMigrationGCE},
		},
		{
			name:                 "nil annotations migration off for GCE",
			volumeAnnotations:    nil,
			expVolumeAnnotations: nil,
			claimAnnotations:     nil,
			expClaimAnnotations:  nil,
			migratedDriverGates:  []featuregate.Feature{},
		},
		{
			name:                 "nil annotations migration on for GCE",
			volumeAnnotations:    nil,
			expVolumeAnnotations: nil,
			claimAnnotations:     nil,
			expClaimAnnotations:  nil,
			migratedDriverGates:  []featuregate.Feature{features.CSIMigrationGCE},
		},
	}

	translator := csitrans.New()
	cmpm := csimigration.NewPluginManager(translator)

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			for _, f := range tc.migratedDriverGates {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, f, true)()
			}
			if tc.volumeAnnotations != nil {
				ann := tc.volumeAnnotations
				updateMigrationAnnotations(cmpm, translator, ann, pvutil.AnnDynamicallyProvisioned)
				if !reflect.DeepEqual(tc.expVolumeAnnotations, ann) {
					t.Errorf("got volume annoations: %v, but expected: %v", ann, tc.expVolumeAnnotations)
				}
			}
			if tc.claimAnnotations != nil {
				ann := tc.claimAnnotations
				updateMigrationAnnotations(cmpm, translator, ann, pvutil.AnnStorageProvisioner)
				if !reflect.DeepEqual(tc.expClaimAnnotations, ann) {
					t.Errorf("got volume annoations: %v, but expected: %v", ann, tc.expVolumeAnnotations)
				}
			}

		})
	}
}
