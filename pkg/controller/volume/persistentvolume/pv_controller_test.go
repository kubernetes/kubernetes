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
	"testing"
	"time"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/features"
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
			newVolumeArray("volume5-2", "1Gi", "uid5-2", "claim5-2", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, annBoundByController),
			noclaims, /* added in testAddClaim5_2 */
			newClaimArray("claim5-2", "uid5-2", "1Gi", "volume5-2", v1.ClaimBound, nil, annBoundByController, annBindCompleted),
			noevents, noerrors,
			// Custom test function that generates an add event
			func(ctrl *PersistentVolumeController, reactor *volumeReactor, test controllerTest) error {
				claim := newClaim("claim5-2", "uid5-2", "1Gi", "", v1.ClaimPending, nil)
				reactor.addClaimEvent(claim)
				return nil
			},
		},
		{
			// deleteClaim with a bound claim makes bound volume released.
			"5-3 - delete claim",
			newVolumeArray("volume5-3", "10Gi", "uid5-3", "claim5-3", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, annBoundByController),
			newVolumeArray("volume5-3", "10Gi", "uid5-3", "claim5-3", v1.VolumeReleased, v1.PersistentVolumeReclaimRetain, classEmpty, annBoundByController),
			newClaimArray("claim5-3", "uid5-3", "1Gi", "volume5-3", v1.ClaimBound, nil, annBoundByController, annBindCompleted),
			noclaims,
			noevents, noerrors,
			// Custom test function that generates a delete event
			func(ctrl *PersistentVolumeController, reactor *volumeReactor, test controllerTest) error {
				obj := ctrl.claims.List()[0]
				claim := obj.(*v1.PersistentVolumeClaim)
				reactor.deleteClaimEvent(claim)
				return nil
			},
		},
		{
			// deleteVolume with a bound volume. Check the claim is Lost.
			"5-4 - delete volume",
			newVolumeArray("volume5-4", "1Gi", "uid5-4", "claim5-4", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			novolumes,
			newClaimArray("claim5-4", "uid5-4", "1Gi", "volume5-4", v1.ClaimBound, nil, annBoundByController, annBindCompleted),
			newClaimArray("claim5-4", "uid5-4", "1Gi", "volume5-4", v1.ClaimLost, nil, annBoundByController, annBindCompleted),
			[]string{"Warning ClaimLost"}, noerrors,
			// Custom test function that generates a delete event
			func(ctrl *PersistentVolumeController, reactor *volumeReactor, test controllerTest) error {
				obj := ctrl.volumes.store.List()[0]
				volume := obj.(*v1.PersistentVolume)
				reactor.deleteVolumeEvent(volume)
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

		reactor := newVolumeReactor(client, ctrl, fakeVolumeWatch, fakeClaimWatch, test.errors)
		for _, claim := range test.initialClaims {
			reactor.claims[claim.Name] = claim
			go func(claim *v1.PersistentVolumeClaim) {
				fakeClaimWatch.Add(claim)
			}(claim)
		}
		for _, volume := range test.initialVolumes {
			reactor.volumes[volume.Name] = volume
			go func(volume *v1.PersistentVolume) {
				fakeVolumeWatch.Add(volume)
			}(volume)
		}

		// Start the controller
		stopCh := make(chan struct{})
		informers.Start(stopCh)
		go ctrl.Run(stopCh)

		// Wait for the controller to pass initial sync and fill its caches.
		for !ctrl.volumeListerSynced() ||
			!ctrl.claimListerSynced() ||
			len(ctrl.claims.ListKeys()) < len(test.initialClaims) ||
			len(ctrl.volumes.store.ListKeys()) < len(test.initialVolumes) {

			time.Sleep(10 * time.Millisecond)
		}
		klog.V(4).Infof("controller synced, starting test")

		// Call the tested function
		err = test.test(ctrl, reactor, test)
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

		evaluateTestResults(ctrl, reactor, test, t)
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

func addVolumeAnnotation(volume *v1.PersistentVolume, annName, annValue string) *v1.PersistentVolume {
	if volume.Annotations == nil {
		volume.Annotations = make(map[string]string)
	}
	volume.Annotations[annName] = annValue
	return volume
}

func makePVCClass(scName *string, hasSelectNodeAnno bool) *v1.PersistentVolumeClaim {
	claim := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{},
		},
		Spec: v1.PersistentVolumeClaimSpec{
			StorageClassName: scName,
		},
	}

	if hasSelectNodeAnno {
		claim.Annotations[annSelectedNode] = "node-name"
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

func TestDelayBinding(t *testing.T) {
	tests := map[string]struct {
		pvc         *v1.PersistentVolumeClaim
		shouldDelay bool
		shouldFail  bool
	}{
		"nil-class": {
			pvc:         makePVCClass(nil, false),
			shouldDelay: false,
		},
		"class-not-found": {
			pvc:         makePVCClass(&classNotHere, false),
			shouldDelay: false,
		},
		"no-mode-class": {
			pvc:         makePVCClass(&classNoMode, false),
			shouldDelay: false,
			shouldFail:  true,
		},
		"immediate-mode-class": {
			pvc:         makePVCClass(&classImmediateMode, false),
			shouldDelay: false,
		},
		"wait-mode-class": {
			pvc:         makePVCClass(&classWaitMode, false),
			shouldDelay: true,
		},
		"wait-mode-class-with-selectedNode": {
			pvc:         makePVCClass(&classWaitMode, true),
			shouldDelay: false,
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
	}

	for _, class := range classes {
		if err := classInformer.Informer().GetIndexer().Add(class); err != nil {
			t.Fatalf("Failed to add storage class %q: %v", class.Name, err)
		}
	}

	for name, test := range tests {
		shouldDelay, err := ctrl.shouldDelayBinding(test.pvc)
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

func TestDelayBindingDisabled(t *testing.T) {
	// When volumeScheduling feature gate is disabled, should always be immediate
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeScheduling, false)()

	client := &fake.Clientset{}
	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	classInformer := informerFactory.Storage().V1().StorageClasses()
	ctrl := &PersistentVolumeController{
		classLister: classInformer.Lister(),
	}

	name := "volumeScheduling-feature-disabled"
	shouldDelay, err := ctrl.shouldDelayBinding(makePVCClass(&classWaitMode, false))
	if err != nil {
		t.Errorf("Test %q returned error: %v", name, err)
	}
	if shouldDelay {
		t.Errorf("Test %q returned true, expected false", name)
	}
}
