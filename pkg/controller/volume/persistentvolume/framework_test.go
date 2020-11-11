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
	"fmt"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	corelisters "k8s.io/client-go/listers/core/v1"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/controller"
	pvtesting "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/testing"
	pvutil "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/util"
	vol "k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/recyclerclient"
)

func init() {
	klog.InitFlags(nil)
}

// This is a unit test framework for persistent volume controller.
// It fills the controller with test claims/volumes and can simulate these
// scenarios:
// 1) Call syncClaim/syncVolume once.
// 2) Call syncClaim/syncVolume several times (both simulating "claim/volume
//    modified" events and periodic sync), until the controller settles down and
//    does not modify anything.
// 3) Simulate almost real API server/etcd and call add/update/delete
//    volume/claim.
// In all these scenarios, when the test finishes, the framework can compare
// resulting claims/volumes with list of expected claims/volumes and report
// differences.

// controllerTest contains a single controller test input.
// Each test has initial set of volumes and claims that are filled into the
// controller before the test starts. The test then contains a reference to
// function to call as the actual test. Available functions are:
//   - testSyncClaim - calls syncClaim on the first claim in initialClaims.
//   - testSyncClaimError - calls syncClaim on the first claim in initialClaims
//                          and expects an error to be returned.
//   - testSyncVolume - calls syncVolume on the first volume in initialVolumes.
//   - any custom function for specialized tests.
// The test then contains list of volumes/claims that are expected at the end
// of the test and list of generated events.
type controllerTest struct {
	// Name of the test, for logging
	name string
	// Initial content of controller volume cache.
	initialVolumes []*v1.PersistentVolume
	// Expected content of controller volume cache at the end of the test.
	expectedVolumes []*v1.PersistentVolume
	// Initial content of controller claim cache.
	initialClaims []*v1.PersistentVolumeClaim
	// Expected content of controller claim cache at the end of the test.
	expectedClaims []*v1.PersistentVolumeClaim
	// Expected events - any event with prefix will pass, we don't check full
	// event message.
	expectedEvents []string
	// Errors to produce on matching action
	errors []pvtesting.ReactorError
	// Function to call as the test.
	test testCall
}

// annSkipLocalStore can be used to mark initial PVs or PVCs that are meant to be added only
// to the fake apiserver (i.e. available via Get) but not to the local store (i.e. the controller
// won't have them in its cache).
const annSkipLocalStore = "pv-testing-skip-local-store"

type testCall func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error

const testNamespace = "default"
const mockPluginName = "kubernetes.io/mock-volume"

var novolumes []*v1.PersistentVolume
var noclaims []*v1.PersistentVolumeClaim
var noevents = []string{}
var noerrors = []pvtesting.ReactorError{}

type volumeReactor struct {
	*pvtesting.VolumeReactor
	ctrl *PersistentVolumeController
}

func newVolumeReactor(client *fake.Clientset, ctrl *PersistentVolumeController, fakeVolumeWatch, fakeClaimWatch *watch.FakeWatcher, errors []pvtesting.ReactorError) *volumeReactor {
	return &volumeReactor{
		pvtesting.NewVolumeReactor(client, fakeVolumeWatch, fakeClaimWatch, errors),
		ctrl,
	}
}

// waitForIdle waits until all tests, controllers and other goroutines do their
// job and no new actions are registered for 10 milliseconds.
func (r *volumeReactor) waitForIdle() {
	r.ctrl.runningOperations.WaitForCompletion()
	// Check every 10ms if the controller does something and stop if it's
	// idle.
	oldChanges := -1
	for {
		time.Sleep(10 * time.Millisecond)
		changes := r.GetChangeCount()
		if changes == oldChanges {
			// No changes for last 10ms -> controller must be idle.
			break
		}
		oldChanges = changes
	}
}

// waitTest waits until all tests, controllers and other goroutines do their
// job and list of current volumes/claims is equal to list of expected
// volumes/claims (with ~10 second timeout).
func (r *volumeReactor) waitTest(test controllerTest) error {
	// start with 10 ms, multiply by 2 each step, 10 steps = 10.23 seconds
	backoff := wait.Backoff{
		Duration: 10 * time.Millisecond,
		Jitter:   0,
		Factor:   2,
		Steps:    10,
	}
	err := wait.ExponentialBackoff(backoff, func() (done bool, err error) {
		// Finish all operations that are in progress
		r.ctrl.runningOperations.WaitForCompletion()

		// Return 'true' if the reactor reached the expected state
		err1 := r.CheckClaims(test.expectedClaims)
		err2 := r.CheckVolumes(test.expectedVolumes)
		if err1 == nil && err2 == nil {
			return true, nil
		}
		return false, nil
	})
	return err
}

// checkEvents compares all expectedEvents with events generated during the test
// and reports differences.
func checkEvents(t *testing.T, expectedEvents []string, ctrl *PersistentVolumeController) error {
	var err error

	// Read recorded events - wait up to 1 minute to get all the expected ones
	// (just in case some goroutines are slower with writing)
	timer := time.NewTimer(time.Minute)
	defer timer.Stop()

	fakeRecorder := ctrl.eventRecorder.(*record.FakeRecorder)
	gotEvents := []string{}
	finished := false
	for len(gotEvents) < len(expectedEvents) && !finished {
		select {
		case event, ok := <-fakeRecorder.Events:
			if ok {
				klog.V(5).Infof("event recorder got event %s", event)
				gotEvents = append(gotEvents, event)
			} else {
				klog.V(5).Infof("event recorder finished")
				finished = true
			}
		case _, _ = <-timer.C:
			klog.V(5).Infof("event recorder timeout")
			finished = true
		}
	}

	// Evaluate the events
	for i, expected := range expectedEvents {
		if len(gotEvents) <= i {
			t.Errorf("Event %q not emitted", expected)
			err = fmt.Errorf("Events do not match")
			continue
		}
		received := gotEvents[i]
		if !strings.HasPrefix(received, expected) {
			t.Errorf("Unexpected event received, expected %q, got %q", expected, received)
			err = fmt.Errorf("Events do not match")
		}
	}
	for i := len(expectedEvents); i < len(gotEvents); i++ {
		t.Errorf("Unexpected event received: %q", gotEvents[i])
		err = fmt.Errorf("Events do not match")
	}
	return err
}

func alwaysReady() bool { return true }

func newTestController(kubeClient clientset.Interface, informerFactory informers.SharedInformerFactory, enableDynamicProvisioning bool) (*PersistentVolumeController, error) {
	if informerFactory == nil {
		informerFactory = informers.NewSharedInformerFactory(kubeClient, controller.NoResyncPeriodFunc())
	}
	params := ControllerParameters{
		KubeClient:                kubeClient,
		SyncPeriod:                5 * time.Second,
		VolumePlugins:             []vol.VolumePlugin{},
		VolumeInformer:            informerFactory.Core().V1().PersistentVolumes(),
		ClaimInformer:             informerFactory.Core().V1().PersistentVolumeClaims(),
		ClassInformer:             informerFactory.Storage().V1().StorageClasses(),
		PodInformer:               informerFactory.Core().V1().Pods(),
		NodeInformer:              informerFactory.Core().V1().Nodes(),
		EventRecorder:             record.NewFakeRecorder(1000),
		EnableDynamicProvisioning: enableDynamicProvisioning,
	}
	ctrl, err := NewController(params)
	if err != nil {
		return nil, fmt.Errorf("failed to construct persistentvolume controller: %v", err)
	}
	ctrl.volumeListerSynced = alwaysReady
	ctrl.claimListerSynced = alwaysReady
	ctrl.classListerSynced = alwaysReady
	// Speed up the test
	ctrl.createProvisionedPVInterval = 5 * time.Millisecond
	return ctrl, nil
}

// newVolume returns a new volume with given attributes
func newVolume(name, capacity, boundToClaimUID, boundToClaimName string, phase v1.PersistentVolumePhase, reclaimPolicy v1.PersistentVolumeReclaimPolicy, class string, annotations ...string) *v1.PersistentVolume {
	fs := v1.PersistentVolumeFilesystem
	volume := v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			ResourceVersion: "1",
		},
		Spec: v1.PersistentVolumeSpec{
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse(capacity),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{},
			},
			AccessModes:                   []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce, v1.ReadOnlyMany},
			PersistentVolumeReclaimPolicy: reclaimPolicy,
			StorageClassName:              class,
			VolumeMode:                    &fs,
		},
		Status: v1.PersistentVolumeStatus{
			Phase: phase,
		},
	}

	if boundToClaimName != "" {
		volume.Spec.ClaimRef = &v1.ObjectReference{
			Kind:       "PersistentVolumeClaim",
			APIVersion: "v1",
			UID:        types.UID(boundToClaimUID),
			Namespace:  testNamespace,
			Name:       boundToClaimName,
		}
	}

	if len(annotations) > 0 {
		volume.Annotations = make(map[string]string)
		for _, a := range annotations {
			switch a {
			case pvutil.AnnDynamicallyProvisioned:
				volume.Annotations[a] = mockPluginName
			default:
				volume.Annotations[a] = "yes"
			}
		}
	}

	return &volume
}

// withLabels applies the given labels to the first volume in the array and
// returns the array.  Meant to be used to compose volumes specified inline in
// a test.
func withLabels(labels map[string]string, volumes []*v1.PersistentVolume) []*v1.PersistentVolume {
	volumes[0].Labels = labels
	return volumes
}

// withLabelSelector sets the label selector of the first claim in the array
// to be MatchLabels of the given label set and returns the array.  Meant
// to be used to compose claims specified inline in a test.
func withLabelSelector(labels map[string]string, claims []*v1.PersistentVolumeClaim) []*v1.PersistentVolumeClaim {
	claims[0].Spec.Selector = &metav1.LabelSelector{
		MatchLabels: labels,
	}

	return claims
}

// withVolumeVolumeMode applies the given VolumeMode to the first volume in the array and
// returns the array.  Meant to be used to compose volumes specified inline in
// a test.
func withVolumeVolumeMode(mode *v1.PersistentVolumeMode, volumes []*v1.PersistentVolume) []*v1.PersistentVolume {
	volumes[0].Spec.VolumeMode = mode
	return volumes
}

// withClaimVolumeMode applies the given VolumeMode to the first claim in the array and
// returns the array.  Meant to be used to compose volumes specified inline in
// a test.
func withClaimVolumeMode(mode *v1.PersistentVolumeMode, claims []*v1.PersistentVolumeClaim) []*v1.PersistentVolumeClaim {
	claims[0].Spec.VolumeMode = mode
	return claims
}

// withExpectedCapacity sets the claim.Spec.Capacity of the first claim in the
// array to given value and returns the array.  Meant to be used to compose
// claims specified inline in a test.
func withExpectedCapacity(capacity string, claims []*v1.PersistentVolumeClaim) []*v1.PersistentVolumeClaim {
	claims[0].Status.Capacity = v1.ResourceList{
		v1.ResourceName(v1.ResourceStorage): resource.MustParse(capacity),
	}

	return claims
}

// withMessage saves given message into volume.Status.Message of the first
// volume in the array and returns the array.  Meant to be used to compose
// volumes specified inline in a test.
func withMessage(message string, volumes []*v1.PersistentVolume) []*v1.PersistentVolume {
	volumes[0].Status.Message = message
	return volumes
}

// newVolumeArray returns array with a single volume that would be returned by
// newVolume() with the same parameters.
func newVolumeArray(name, capacity, boundToClaimUID, boundToClaimName string, phase v1.PersistentVolumePhase, reclaimPolicy v1.PersistentVolumeReclaimPolicy, class string, annotations ...string) []*v1.PersistentVolume {
	return []*v1.PersistentVolume{
		newVolume(name, capacity, boundToClaimUID, boundToClaimName, phase, reclaimPolicy, class, annotations...),
	}
}

func withVolumeDeletionTimestamp(pvs []*v1.PersistentVolume) []*v1.PersistentVolume {
	result := []*v1.PersistentVolume{}
	for _, pv := range pvs {
		// Using time.Now() here will cause mismatching deletion timestamps in tests
		deleteTime := metav1.Date(2020, time.February, 18, 10, 30, 30, 10, time.UTC)
		pv.SetDeletionTimestamp(&deleteTime)
		result = append(result, pv)
	}
	return result
}

// newClaim returns a new claim with given attributes
func newClaim(name, claimUID, capacity, boundToVolume string, phase v1.PersistentVolumeClaimPhase, class *string, annotations ...string) *v1.PersistentVolumeClaim {
	fs := v1.PersistentVolumeFilesystem
	claim := v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			Namespace:       testNamespace,
			UID:             types.UID(claimUID),
			ResourceVersion: "1",
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce, v1.ReadOnlyMany},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse(capacity),
				},
			},
			VolumeName:       boundToVolume,
			StorageClassName: class,
			VolumeMode:       &fs,
		},
		Status: v1.PersistentVolumeClaimStatus{
			Phase: phase,
		},
	}
	// Make sure ref.GetReference(claim) works
	claim.ObjectMeta.SelfLink = "/api/v1/namespaces/" + testNamespace + "/persistentvolumeclaims/" + name

	if len(annotations) > 0 {
		claim.Annotations = make(map[string]string)
		for _, a := range annotations {
			switch a {
			case pvutil.AnnStorageProvisioner:
				claim.Annotations[a] = mockPluginName
			default:
				claim.Annotations[a] = "yes"
			}
		}
	}

	// Bound claims must have proper Status.
	if phase == v1.ClaimBound {
		claim.Status.AccessModes = claim.Spec.AccessModes
		// For most of the tests it's enough to copy claim's requested capacity,
		// individual tests can adjust it using withExpectedCapacity()
		claim.Status.Capacity = claim.Spec.Resources.Requests
	}

	return &claim
}

// newClaimArray returns array with a single claim that would be returned by
// newClaim() with the same parameters.
func newClaimArray(name, claimUID, capacity, boundToVolume string, phase v1.PersistentVolumeClaimPhase, class *string, annotations ...string) []*v1.PersistentVolumeClaim {
	return []*v1.PersistentVolumeClaim{
		newClaim(name, claimUID, capacity, boundToVolume, phase, class, annotations...),
	}
}

// claimWithAnnotation saves given annotation into given claims. Meant to be
// used to compose claims specified inline in a test.
// TODO(refactor): This helper function (and other helpers related to claim
// arrays) could use some cleaning up (most assume an array size of one)-
// replace with annotateClaim at all callsites. The tests require claimArrays
// but mostly operate on single claims
func claimWithAnnotation(name, value string, claims []*v1.PersistentVolumeClaim) []*v1.PersistentVolumeClaim {
	if claims[0].Annotations == nil {
		claims[0].Annotations = map[string]string{name: value}
	} else {
		claims[0].Annotations[name] = value
	}
	return claims
}

func annotateClaim(claim *v1.PersistentVolumeClaim, ann map[string]string) *v1.PersistentVolumeClaim {
	if claim.Annotations == nil {
		claim.Annotations = map[string]string{}
	}
	for key, val := range ann {
		claim.Annotations[key] = val
	}
	return claim
}

// volumeWithAnnotation saves given annotation into given volume.
// Meant to be used to compose volume specified inline in a test.
func volumeWithAnnotation(name, value string, volume *v1.PersistentVolume) *v1.PersistentVolume {
	if volume.Annotations == nil {
		volume.Annotations = map[string]string{name: value}
	} else {
		volume.Annotations[name] = value
	}
	return volume
}

// volumesWithAnnotation saves given annotation into given volumes.
// Meant to be used to compose volumes specified inline in a test.
func volumesWithAnnotation(name, value string, volumes []*v1.PersistentVolume) []*v1.PersistentVolume {
	for _, volume := range volumes {
		volumeWithAnnotation(name, value, volume)
	}
	return volumes
}

// claimWithAccessMode saves given access into given claims.
// Meant to be used to compose claims specified inline in a test.
func claimWithAccessMode(modes []v1.PersistentVolumeAccessMode, claims []*v1.PersistentVolumeClaim) []*v1.PersistentVolumeClaim {
	claims[0].Spec.AccessModes = modes
	return claims
}

func testSyncClaim(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error {
	return ctrl.syncClaim(test.initialClaims[0])
}

func testSyncClaimError(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error {
	err := ctrl.syncClaim(test.initialClaims[0])

	if err != nil {
		return nil
	}
	return fmt.Errorf("syncClaim succeeded when failure was expected")
}

func testSyncVolume(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error {
	return ctrl.syncVolume(test.initialVolumes[0])
}

type operationType string

const operationDelete = "Delete"
const operationRecycle = "Recycle"

var (
	classGold                    string = "gold"
	classSilver                  string = "silver"
	classCopper                  string = "copper"
	classEmpty                   string = ""
	classNonExisting             string = "non-existing"
	classExternal                string = "external"
	classExternalWait            string = "external-wait"
	classUnknownInternal         string = "unknown-internal"
	classUnsupportedMountOptions string = "unsupported-mountoptions"
	classLarge                   string = "large"
	classWait                    string = "wait"

	modeWait = storage.VolumeBindingWaitForFirstConsumer
)

// wrapTestWithPluginCalls returns a testCall that:
// - configures controller with a volume plugin that implements recycler,
//   deleter and provisioner. The plugin returns provided errors when a volume
//   is deleted, recycled or provisioned.
// - calls given testCall
func wrapTestWithPluginCalls(expectedRecycleCalls, expectedDeleteCalls []error, expectedProvisionCalls []provisionCall, toWrap testCall) testCall {
	return func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error {
		plugin := &mockVolumePlugin{
			recycleCalls:   expectedRecycleCalls,
			deleteCalls:    expectedDeleteCalls,
			provisionCalls: expectedProvisionCalls,
		}
		ctrl.volumePluginMgr.InitPlugins([]vol.VolumePlugin{plugin}, nil /* prober */, ctrl)
		return toWrap(ctrl, reactor, test)
	}
}

// wrapTestWithReclaimCalls returns a testCall that:
// - configures controller with recycler or deleter which will return provided
//   errors when a volume is deleted or recycled
// - calls given testCall
func wrapTestWithReclaimCalls(operation operationType, expectedOperationCalls []error, toWrap testCall) testCall {
	if operation == operationDelete {
		return wrapTestWithPluginCalls(nil, expectedOperationCalls, nil, toWrap)
	} else {
		return wrapTestWithPluginCalls(expectedOperationCalls, nil, nil, toWrap)
	}
}

// wrapTestWithProvisionCalls returns a testCall that:
// - configures controller with a provisioner which will return provided errors
//   when a claim is provisioned
// - calls given testCall
func wrapTestWithProvisionCalls(expectedProvisionCalls []provisionCall, toWrap testCall) testCall {
	return wrapTestWithPluginCalls(nil, nil, expectedProvisionCalls, toWrap)
}

type fakeCSINameTranslator struct{}

func (t fakeCSINameTranslator) GetCSINameFromInTreeName(pluginName string) (string, error) {
	return "vendor.com/MockCSIDriver", nil
}

type fakeCSIMigratedPluginManager struct{}

func (t fakeCSIMigratedPluginManager) IsMigrationEnabledForPlugin(pluginName string) bool {
	return true
}

// wrapTestWithCSIMigrationProvisionCalls returns a testCall that:
// - configures controller with a volume plugin that emulates CSI migration
// - calls given testCall
func wrapTestWithCSIMigrationProvisionCalls(toWrap testCall) testCall {
	plugin := &mockVolumePlugin{}
	return func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error {
		ctrl.volumePluginMgr.InitPlugins([]vol.VolumePlugin{plugin}, nil /* prober */, ctrl)
		ctrl.translator = fakeCSINameTranslator{}
		ctrl.csiMigratedPluginManager = fakeCSIMigratedPluginManager{}
		return toWrap(ctrl, reactor, test)
	}
}

// wrapTestWithInjectedOperation returns a testCall that:
// - starts the controller and lets it run original testCall until
//   scheduleOperation() call. It blocks the controller there and calls the
//   injected function to simulate that something is happening when the
//   controller waits for the operation lock. Controller is then resumed and we
//   check how it behaves.
func wrapTestWithInjectedOperation(toWrap testCall, injectBeforeOperation func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor)) testCall {

	return func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest) error {
		// Inject a hook before async operation starts
		ctrl.preOperationHook = func(operationName string) {
			// Inside the hook, run the function to inject
			klog.V(4).Infof("reactor: scheduleOperation reached, injecting call")
			injectBeforeOperation(ctrl, reactor)
		}

		// Run the tested function (typically syncClaim/syncVolume) in a
		// separate goroutine.
		var testError error
		var testFinished int32

		go func() {
			testError = toWrap(ctrl, reactor, test)
			// Let the "main" test function know that syncVolume has finished.
			atomic.StoreInt32(&testFinished, 1)
		}()

		// Wait for the controller to finish the test function.
		for atomic.LoadInt32(&testFinished) == 0 {
			time.Sleep(time.Millisecond * 10)
		}

		return testError
	}
}

func evaluateTestResults(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor, test controllerTest, t *testing.T) {
	// Evaluate results
	if err := reactor.CheckClaims(test.expectedClaims); err != nil {
		t.Errorf("Test %q: %v", test.name, err)

	}
	if err := reactor.CheckVolumes(test.expectedVolumes); err != nil {
		t.Errorf("Test %q: %v", test.name, err)
	}

	if err := checkEvents(t, test.expectedEvents, ctrl); err != nil {
		t.Errorf("Test %q: %v", test.name, err)
	}
}

// Test single call to syncClaim and syncVolume methods.
// For all tests:
// 1. Fill in the controller with initial data
// 2. Call the tested function (syncClaim/syncVolume) via
//    controllerTest.testCall *once*.
// 3. Compare resulting volumes and claims with expected volumes and claims.
func runSyncTests(t *testing.T, tests []controllerTest, storageClasses []*storage.StorageClass, pods []*v1.Pod) {
	doit := func(t *testing.T, test controllerTest) {
		// Initialize the controller
		client := &fake.Clientset{}
		ctrl, err := newTestController(client, nil, true)
		if err != nil {
			t.Fatalf("Test %q construct persistent volume failed: %v", test.name, err)
		}
		reactor := newVolumeReactor(client, ctrl, nil, nil, test.errors)
		for _, claim := range test.initialClaims {
			if metav1.HasAnnotation(claim.ObjectMeta, annSkipLocalStore) {
				continue
			}
			ctrl.claims.Add(claim)
		}
		for _, volume := range test.initialVolumes {
			if metav1.HasAnnotation(volume.ObjectMeta, annSkipLocalStore) {
				continue
			}
			ctrl.volumes.store.Add(volume)
		}
		reactor.AddClaims(test.initialClaims)
		reactor.AddVolumes(test.initialVolumes)

		// Inject classes into controller via a custom lister.
		indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
		for _, class := range storageClasses {
			indexer.Add(class)
		}
		ctrl.classLister = storagelisters.NewStorageClassLister(indexer)

		podIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
		for _, pod := range pods {
			podIndexer.Add(pod)
			ctrl.podIndexer.Add(pod)
		}
		ctrl.podLister = corelisters.NewPodLister(podIndexer)

		// Run the tested functions
		err = test.test(ctrl, reactor.VolumeReactor, test)
		if err != nil {
			t.Errorf("Test %q failed: %v", test.name, err)
		}

		// Wait for the target state
		err = reactor.waitTest(test)
		if err != nil {
			t.Errorf("Test %q failed: %v", test.name, err)
		}

		evaluateTestResults(ctrl, reactor.VolumeReactor, test, t)
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			doit(t, test)
		})
	}
}

// Test multiple calls to syncClaim/syncVolume and periodic sync of all
// volume/claims. For all tests, the test follows this pattern:
// 0. Load the controller with initial data.
// 1. Call controllerTest.testCall() once as in TestSync()
// 2. For all volumes/claims changed by previous syncVolume/syncClaim calls,
//    call appropriate syncVolume/syncClaim (simulating "volume/claim changed"
//    events). Go to 2. if these calls change anything.
// 3. When all changes are processed and no new changes were made, call
//    syncVolume/syncClaim on all volumes/claims (simulating "periodic sync").
// 4. If some changes were done by step 3., go to 2. (simulation of
//    "volume/claim updated" events, eventually performing step 3. again)
// 5. When 3. does not do any changes, finish the tests and compare final set
//    of volumes/claims with expected claims/volumes and report differences.
// Some limit of calls in enforced to prevent endless loops.
func runMultisyncTests(t *testing.T, tests []controllerTest, storageClasses []*storage.StorageClass, defaultStorageClass string) {
	for _, test := range tests {
		klog.V(4).Infof("starting multisync test %q", test.name)

		// Initialize the controller
		client := &fake.Clientset{}
		ctrl, err := newTestController(client, nil, true)
		if err != nil {
			t.Fatalf("Test %q construct persistent volume failed: %v", test.name, err)
		}

		// Inject classes into controller via a custom lister.
		indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
		for _, class := range storageClasses {
			indexer.Add(class)
		}
		ctrl.classLister = storagelisters.NewStorageClassLister(indexer)

		reactor := newVolumeReactor(client, ctrl, nil, nil, test.errors)
		for _, claim := range test.initialClaims {
			ctrl.claims.Add(claim)
		}
		for _, volume := range test.initialVolumes {
			ctrl.volumes.store.Add(volume)
		}
		reactor.AddClaims(test.initialClaims)
		reactor.AddVolumes(test.initialVolumes)

		// Run the tested function
		err = test.test(ctrl, reactor.VolumeReactor, test)
		if err != nil {
			t.Errorf("Test %q failed: %v", test.name, err)
		}

		// Simulate any "changed" events and "periodical sync" until we reach a
		// stable state.
		firstSync := true
		counter := 0
		for {
			counter++
			klog.V(4).Infof("test %q: iteration %d", test.name, counter)

			if counter > 100 {
				t.Errorf("Test %q failed: too many iterations", test.name)
				break
			}

			// Wait for all goroutines to finish
			reactor.waitForIdle()

			obj := reactor.PopChange()
			if obj == nil {
				// Nothing was changed, should we exit?
				if firstSync || reactor.GetChangeCount() > 0 {
					// There were some changes after the last "periodic sync".
					// Simulate "periodic sync" of everything (until it produces
					// no changes).
					firstSync = false
					klog.V(4).Infof("test %q: simulating periodical sync of all claims and volumes", test.name)
					reactor.SyncAll()
				} else {
					// Last sync did not produce any updates, the test reached
					// stable state -> finish.
					break
				}
			}
			// waiting here cools down exponential backoff
			time.Sleep(600 * time.Millisecond)

			// There were some changes, process them
			switch obj.(type) {
			case *v1.PersistentVolumeClaim:
				claim := obj.(*v1.PersistentVolumeClaim)
				// Simulate "claim updated" event
				ctrl.claims.Update(claim)
				err = ctrl.syncClaim(claim)
				if err != nil {
					if err == pvtesting.ErrVersionConflict {
						// Ignore version errors
						klog.V(4).Infof("test intentionally ignores version error.")
					} else {
						t.Errorf("Error calling syncClaim: %v", err)
						// Finish the loop on the first error
						break
					}
				}
				// Process generated changes
				continue
			case *v1.PersistentVolume:
				volume := obj.(*v1.PersistentVolume)
				// Simulate "volume updated" event
				ctrl.volumes.store.Update(volume)
				err = ctrl.syncVolume(volume)
				if err != nil {
					if err == pvtesting.ErrVersionConflict {
						// Ignore version errors
						klog.V(4).Infof("test intentionally ignores version error.")
					} else {
						t.Errorf("Error calling syncVolume: %v", err)
						// Finish the loop on the first error
						break
					}
				}
				// Process generated changes
				continue
			}
		}
		evaluateTestResults(ctrl, reactor.VolumeReactor, test, t)
		klog.V(4).Infof("test %q finished after %d iterations", test.name, counter)
	}
}

// Dummy volume plugin for provisioning, deletion and recycling. It contains
// lists of expected return values to simulate errors.
type mockVolumePlugin struct {
	provisionCalls       []provisionCall
	provisionCallCounter int
	deleteCalls          []error
	deleteCallCounter    int
	recycleCalls         []error
	recycleCallCounter   int
	provisionOptions     vol.VolumeOptions
}

type provisionCall struct {
	expectedParameters map[string]string
	ret                error
}

var _ vol.VolumePlugin = &mockVolumePlugin{}
var _ vol.RecyclableVolumePlugin = &mockVolumePlugin{}
var _ vol.DeletableVolumePlugin = &mockVolumePlugin{}
var _ vol.ProvisionableVolumePlugin = &mockVolumePlugin{}

func (plugin *mockVolumePlugin) Init(host vol.VolumeHost) error {
	return nil
}

func (plugin *mockVolumePlugin) GetPluginName() string {
	return mockPluginName
}

func (plugin *mockVolumePlugin) GetVolumeName(spec *vol.Spec) (string, error) {
	return spec.Name(), nil
}

func (plugin *mockVolumePlugin) CanSupport(spec *vol.Spec) bool {
	return true
}

func (plugin *mockVolumePlugin) RequiresRemount() bool {
	return false
}

func (plugin *mockVolumePlugin) SupportsMountOption() bool {
	return false
}

func (plugin *mockVolumePlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *mockVolumePlugin) ConstructVolumeSpec(volumeName, mountPath string) (*vol.Spec, error) {
	return nil, nil
}

func (plugin *mockVolumePlugin) NewMounter(spec *vol.Spec, podRef *v1.Pod, opts vol.VolumeOptions) (vol.Mounter, error) {
	return nil, fmt.Errorf("Mounter is not supported by this plugin")
}

func (plugin *mockVolumePlugin) NewUnmounter(name string, podUID types.UID) (vol.Unmounter, error) {
	return nil, fmt.Errorf("Unmounter is not supported by this plugin")
}

// Provisioner interfaces

func (plugin *mockVolumePlugin) NewProvisioner(options vol.VolumeOptions) (vol.Provisioner, error) {
	if len(plugin.provisionCalls) > 0 {
		// mockVolumePlugin directly implements Provisioner interface
		klog.V(4).Infof("mock plugin NewProvisioner called, returning mock provisioner")
		plugin.provisionOptions = options
		return plugin, nil
	} else {
		return nil, fmt.Errorf("Mock plugin error: no provisionCalls configured")
	}
}

func (plugin *mockVolumePlugin) Provision(selectedNode *v1.Node, allowedTopologies []v1.TopologySelectorTerm) (*v1.PersistentVolume, error) {
	if len(plugin.provisionCalls) <= plugin.provisionCallCounter {
		return nil, fmt.Errorf("Mock plugin error: unexpected provisioner call %d", plugin.provisionCallCounter)
	}

	var pv *v1.PersistentVolume
	call := plugin.provisionCalls[plugin.provisionCallCounter]
	if !reflect.DeepEqual(call.expectedParameters, plugin.provisionOptions.Parameters) {
		klog.Errorf("invalid provisioner call, expected options: %+v, got: %+v", call.expectedParameters, plugin.provisionOptions.Parameters)
		return nil, fmt.Errorf("Mock plugin error: invalid provisioner call")
	}
	if call.ret == nil {
		// Create a fake PV with known GCE volume (to match expected volume)
		capacity := plugin.provisionOptions.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
		accessModes := plugin.provisionOptions.PVC.Spec.AccessModes
		pv = &v1.PersistentVolume{
			ObjectMeta: metav1.ObjectMeta{
				Name: plugin.provisionOptions.PVName,
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): capacity,
				},
				AccessModes:                   accessModes,
				PersistentVolumeReclaimPolicy: plugin.provisionOptions.PersistentVolumeReclaimPolicy,
				PersistentVolumeSource: v1.PersistentVolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{},
				},
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeAvailable,
			},
		}
		pv.Spec.VolumeMode = plugin.provisionOptions.PVC.Spec.VolumeMode
	}

	plugin.provisionCallCounter++
	klog.V(4).Infof("mock plugin Provision call nr. %d, returning %v: %v", plugin.provisionCallCounter, pv, call.ret)
	return pv, call.ret
}

// Deleter interfaces

func (plugin *mockVolumePlugin) NewDeleter(spec *vol.Spec) (vol.Deleter, error) {
	if len(plugin.deleteCalls) > 0 {
		// mockVolumePlugin directly implements Deleter interface
		klog.V(4).Infof("mock plugin NewDeleter called, returning mock deleter")
		return plugin, nil
	} else {
		return nil, fmt.Errorf("Mock plugin error: no deleteCalls configured")
	}
}

func (plugin *mockVolumePlugin) Delete() error {
	if len(plugin.deleteCalls) <= plugin.deleteCallCounter {
		return fmt.Errorf("Mock plugin error: unexpected deleter call %d", plugin.deleteCallCounter)
	}
	ret := plugin.deleteCalls[plugin.deleteCallCounter]
	plugin.deleteCallCounter++
	klog.V(4).Infof("mock plugin Delete call nr. %d, returning %v", plugin.deleteCallCounter, ret)
	return ret
}

// Volume interfaces

func (plugin *mockVolumePlugin) GetPath() string {
	return ""
}

func (plugin *mockVolumePlugin) GetMetrics() (*vol.Metrics, error) {
	return nil, nil
}

// Recycler interfaces

func (plugin *mockVolumePlugin) Recycle(pvName string, spec *vol.Spec, eventRecorder recyclerclient.RecycleEventRecorder) error {
	if len(plugin.recycleCalls) == 0 {
		return fmt.Errorf("Mock plugin error: no recycleCalls configured")
	}

	if len(plugin.recycleCalls) <= plugin.recycleCallCounter {
		return fmt.Errorf("Mock plugin error: unexpected recycle call %d", plugin.recycleCallCounter)
	}
	ret := plugin.recycleCalls[plugin.recycleCallCounter]
	plugin.recycleCallCounter++
	klog.V(4).Infof("mock plugin Recycle call nr. %d, returning %v", plugin.recycleCallCounter, ret)
	return ret
}
