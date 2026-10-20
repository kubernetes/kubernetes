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

package dra

import (
	"context"
	"reflect"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	draapi "k8s.io/dynamic-resource-allocation/api"
	resourceslicetracker "k8s.io/dynamic-resource-allocation/resourceslice/tracker"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

func TestSharingAffinityStateLockLifecycle(t *testing.T) {
	fixture := newSharingAffinityStateFixture(t, true)
	first := fixture.allocateClaim(t, "first", "1Gi", sharingAffinityStateParameters)
	second := fixture.allocateClaim(t, "second", "2Gi", sharingAffinityStateParameters)

	state := fixture.waitForCapacity(t, "3Gi")
	fixture.assertAffinity(t, state, structured.DeviceAffinityState{
		Status:         structured.AffinityStatusLocked,
		LockedAffinity: map[string]string{"subnet": "subnet-a"},
	})

	fixture.deleteClaim(t, first)
	state = fixture.waitForCapacity(t, "2Gi")
	fixture.assertAffinity(t, state, structured.DeviceAffinityState{
		Status:         structured.AffinityStatusLocked,
		LockedAffinity: map[string]string{"subnet": "subnet-a"},
	})

	fixture.deleteClaim(t, second)
	state = fixture.waitForCapacity(t, "")
	if affinity, found := state.DeviceSharingAffinity.DeviceStates[fixture.device]; found {
		t.Fatalf("last claim deleted: affinity state remains: %#v", affinity)
	}
}

func TestSharingAffinityStateUnsatisfiedUnreconstructable(t *testing.T) {
	fixture := newSharingAffinityStateFixture(t, true)
	claim := fixture.allocateClaim(t, "unsatisfied", "1Gi",
		`{"apiVersion":"networking.example.com/v1","kind":"NICConfig"}`)

	// Capacity is published while the owner's record is still pending; only
	// a settled (non-stale) snapshot may be judged for Unreconstructable.
	state := fixture.waitForSettledCapacity(t, "1Gi")
	fixture.assertAffinity(t, state, structured.DeviceAffinityState{
		Status: structured.AffinityStatusUnreconstructable,
	})

	fixture.deleteClaim(t, claim)
	state = fixture.waitForSettledCapacity(t, "")
	if affinity, found := state.DeviceSharingAffinity.DeviceStates[fixture.device]; found {
		t.Fatalf("unreconstructable claim drained: affinity state remains: %#v", affinity)
	}
}

func TestSharingAffinityStateGateDisabled(t *testing.T) {
	fixture := newSharingAffinityStateFixture(t, false)
	fixture.allocateClaim(t, "first", "1Gi", sharingAffinityStateParameters)
	fixture.allocateClaim(t, "second", "2Gi", sharingAffinityStateParameters)

	state := fixture.waitForCapacity(t, "3Gi")
	if state.DeviceSharingAffinity.DeviceStates != nil {
		t.Fatalf("sharing affinity disabled: want nil affinity states, got %#v", state.DeviceSharingAffinity.DeviceStates)
	}
}

const sharingAffinityStateParameters = `{"apiVersion":"networking.example.com/v1","kind":"NICConfig","subnetID":"subnet-a"}`

type sharingAffinityStateFixture struct {
	ctx       context.Context
	client    kubernetes.Interface
	manager   *dynamicresources.DefaultDRAManager
	namespace string
	device    structured.DeviceID
}

func newSharingAffinityStateFixture(t *testing.T, enabled bool) *sharingAffinityStateFixture {
	t.Helper()
	// The server helper scopes process-global gates to this serial test.
	server := startSharingAffinityServer(t, enabled, framework.SharedEtcd())
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	t.Cleanup(cancel)
	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("create sharing affinity state client: %v", err)
	}
	namespace, err := client.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{GenerateName: "sharing-affinity-state-"},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("create sharing affinity state namespace: %v", err)
	}

	informerFactory := informers.NewSharedInformerFactory(client, 0)
	options := resourceslicetracker.Options{
		EnableDeviceTaintRules:   utilfeature.DefaultFeatureGate.Enabled(features.DRADeviceTaintRules),
		EnableConsumableCapacity: true,
		SliceInformer:            informerFactory.Resource().V1().ResourceSlices(),
		KubeClient:               client,
	}
	if options.EnableDeviceTaintRules {
		options.TaintInformer = informerFactory.Resource().V1().DeviceTaintRules()
	}
	tracker, err := resourceslicetracker.StartTracker(ctx, options)
	if err != nil {
		t.Fatalf("start sharing affinity ResourceSlice tracker: %v", err)
	}
	t.Cleanup(func() {
		cancel()
		tracker.Stop()
		informerFactory.Shutdown()
	})
	claimsCache := assumecache.NewAssumeCache(logger, informerFactory.Resource().V1().ResourceClaims().Informer(), "resource claim", "", nil)
	manager := dynamicresources.NewDRAManager(ctx, claimsCache, tracker, informerFactory)
	// Informer synchronization alone does not wait for assume-cache handlers.
	claimsSynced := claimsCache.AddEventHandler(cache.ResourceEventHandlerFuncs{}).HasSyncedChecker()
	informerFactory.Start(ctx.Done())
	syncCtx, syncCancel := context.WithTimeout(ctx, 30*time.Second)
	defer syncCancel()
	for informerType, synced := range informerFactory.WaitForCacheSync(syncCtx.Done()) {
		if !synced {
			t.Fatalf("sharing affinity informer %v did not synchronize", informerType)
		}
	}
	if !cache.WaitFor(syncCtx, "sharing affinity cache handlers", tracker.HasSyncedChecker(), claimsSynced) {
		t.Fatal("sharing affinity tracker and assume-cache handlers did not synchronize")
	}

	metadata := sharingAffinitySlice()
	metadata.Spec.Pool.ResourceSliceCount = 2
	devices := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{GenerateName: "sharing-affinity-devices-"},
		Spec: resourceapi.ResourceSliceSpec{
			Driver:   metadata.Spec.Driver,
			Pool:     metadata.Spec.Pool,
			AllNodes: ptr.To(true),
			Devices: []resourceapi.Device{{
				Name:                     "nic",
				AllowMultipleAllocations: ptr.To(true),
				Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
					"memory": {Value: resource.MustParse("8Gi")},
				},
			}},
		},
	}
	published := make(map[string]*resourceapi.ResourceSlice)
	for _, slice := range []*resourceapi.ResourceSlice{metadata, devices} {
		created, err := client.ResourceV1().ResourceSlices().Create(ctx, slice, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("publish sharing affinity pool slice: %v", err)
		}
		published[created.Name] = created
	}
	// Claim events must not race ahead of complete pool metadata observations.
	if err := wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, 30*time.Second, true, func(context.Context) (bool, error) {
		slices, err := tracker.ListPatchedResourceSlices()
		if err != nil {
			return false, err
		}
		observed := 0
		for _, slice := range slices {
			if slice.Spec.Driver != metadata.Spec.Driver || slice.Spec.Pool.Name != metadata.Spec.Pool.Name {
				continue
			}
			want, found := published[slice.Name]
			if !found || slice.UID != want.UID || !reflect.DeepEqual(slice.Spec, want.Spec) {
				return false, nil
			}
			observed++
		}
		return observed == 2, nil
	}); err != nil {
		t.Fatalf("wait for complete sharing affinity metadata/device pool in tracker: %v", err)
	}
	return &sharingAffinityStateFixture{
		ctx:       ctx,
		client:    client,
		manager:   manager,
		namespace: namespace.Name,
		device:    structured.MakeDeviceID(metadata.Spec.Driver, metadata.Spec.Pool.Name, "nic"),
	}
}

func (f *sharingAffinityStateFixture) allocateClaim(t *testing.T, name, capacity, parameters string) *resourceapi.ResourceClaim {
	t.Helper()
	claims := f.client.ResourceV1().ResourceClaims(f.namespace)
	claim, err := claims.Create(f.ctx, &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: resourceapi.ResourceClaimSpec{
			Devices: resourceapi.DeviceClaim{
				Requests: []resourceapi.DeviceRequest{{
					Name: "nic",
					Exactly: &resourceapi.ExactDeviceRequest{
						DeviceClassName: "sharing-affinity",
						AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
						Count:           1,
					},
				}},
				Config: []resourceapi.DeviceClaimConfiguration{{
					Requests: []string{"nic"},
					DeviceConfiguration: resourceapi.DeviceConfiguration{
						Opaque: &resourceapi.OpaqueDeviceConfiguration{
							Driver:     f.device.Driver.String(),
							Parameters: runtime.RawExtension{Raw: []byte(parameters)},
						},
					},
				}},
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("create sharing affinity claim %q: %v", name, err)
	}
	claim.Status.Allocation = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{{
				Request: "nic",
				Driver:  f.device.Driver.String(),
				Pool:    f.device.Pool.String(),
				Device:  f.device.Device.String(),
				ShareID: ptr.To(uuid.NewUUID()),
				ConsumedCapacity: map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse(capacity),
				},
			}},
		},
	}
	claim, err = claims.UpdateStatus(f.ctx, claim, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("allocate sharing affinity claim %q through status update: %v", name, err)
	}
	return claim
}

func (f *sharingAffinityStateFixture) deleteClaim(t *testing.T, claim *resourceapi.ResourceClaim) {
	t.Helper()
	if err := f.client.ResourceV1().ResourceClaims(f.namespace).Delete(f.ctx, claim.Name, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("delete sharing affinity claim %q: %v", claim.Name, err)
	}
}

func (f *sharingAffinityStateFixture) waitForCapacity(t *testing.T, capacity string) *structured.AllocatedState {
	t.Helper()
	var want resource.Quantity
	if capacity != "" {
		want = resource.MustParse(capacity)
	}
	key := draapi.MakeFullyQualifiedName("memory", f.device.Driver.String())
	var state *structured.AllocatedState
	var gatherErr error
	err := wait.PollUntilContextTimeout(f.ctx, 10*time.Millisecond, 30*time.Second, true, func(context.Context) (bool, error) {
		state, gatherErr = f.manager.ResourceClaims().GatherAllocatedState()
		if gatherErr != nil {
			// GatherAllocatedState asks callers to retry concurrent revisions.
			return false, nil
		}
		if state == nil {
			return false, nil
		}
		if capacity == "" {
			_, found := state.AggregatedCapacity[f.device]
			return !found && !state.AllocatedSharedDeviceIDs.Has(f.device), nil
		}
		got := state.AggregatedCapacity[f.device][key]
		return got != nil && got.Cmp(want) == 0, nil
	})
	if err != nil {
		t.Fatalf("wait for device %s capacity %q: %v; last gather error: %v; last state: %#v", f.device, capacity, err, gatherErr, state)
	}
	if len(state.AllocatedDevices) != 0 {
		t.Fatalf("shared claims unexpectedly own dedicated devices: %v", state.AllocatedDevices)
	}
	if capacity != "" && (len(state.AllocatedSharedDeviceIDs) != 1 || !state.AllocatedSharedDeviceIDs.Has(f.device)) {
		t.Fatalf("want one shared device %s, got %v", f.device, state.AllocatedSharedDeviceIDs)
	}
	return state
}

// waitForSettledCapacity waits for exact capacity on a pool that has left
// StalePools. Every pending snapshot seen on the way must fail closed: the pool
// excluded and no ready (Clean/Locked) state exposed for the device.
func (f *sharingAffinityStateFixture) waitForSettledCapacity(t *testing.T, capacity string) *structured.AllocatedState {
	t.Helper()
	pool := schedulerapi.PoolID{Driver: f.device.Driver.String(), Pool: f.device.Pool.String()}
	var state *structured.AllocatedState
	err := wait.PollUntilContextTimeout(f.ctx, 10*time.Millisecond, 30*time.Second, true, func(context.Context) (bool, error) {
		state = f.waitForCapacity(t, capacity)
		if !state.DeviceSharingAffinity.StalePools.Has(pool) {
			return true, nil
		}
		if affinity, found := state.DeviceSharingAffinity.DeviceStates[f.device]; found && affinity.Status != structured.AffinityStatusUnreconstructable {
			t.Fatalf("pending pool %v exposes non-fenced affinity %#v", pool, affinity)
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("wait for pool %v to settle with capacity %q: %v; last state: %#v", pool, capacity, err, state)
	}
	return state
}

func (f *sharingAffinityStateFixture) assertAffinity(t *testing.T, state *structured.AllocatedState, want structured.DeviceAffinityState) {
	t.Helper()
	got, found := state.DeviceSharingAffinity.DeviceStates[f.device]
	if !found || len(state.DeviceSharingAffinity.DeviceStates) != 1 || !reflect.DeepEqual(got, want) {
		t.Fatalf("device %s: want affinity %#v, got %#v (present: %t); all states: %#v", f.device, want, got, found, state.DeviceSharingAffinity.DeviceStates)
	}
}
