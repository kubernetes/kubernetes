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

package scheduler

import (
	"context"
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/events"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestFrameworkComponentsWaitForHandlersSync(t *testing.T) {
	deviceClass := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{Name: "gpu-class"},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: new("example.com/gpu"),
		},
	}
	claim := st.MakeResourceClaim().
		Name("test-claim").
		Namespace("default").
		UID("claim-uid-1").
		Obj()
	claim.ResourceVersion = "1"
	slice := st.MakeResourceSlice("node-1", "driver.example.com").
		Devices("dev-1").
		Obj()

	tests := []struct {
		name            string
		emulatedVersion string
		overrides       featuregatetesting.FeatureOverrides
		wantDeviceClass *resourceapi.DeviceClass
	}{
		{
			name:            "DRAExtendedResource enabled",
			wantDeviceClass: deviceClass,
		},
		{
			name:            "DRAExtendedResource disabled",
			emulatedVersion: "1.36",
			overrides: featuregatetesting.FeatureOverrides{
				features.DRAExtendedResource: false,
			},
			wantDeviceClass: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.emulatedVersion != "" {
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse(tt.emulatedVersion))
			}
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, tt.overrides)

			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			client := fake.NewClientset(deviceClass, claim, slice)
			informerFactory := informers.NewSharedInformerFactory(client, 0)

			comps, err := NewFrameworkComponents(ctx, client, informerFactory)
			if err != nil {
				t.Fatalf("NewFrameworkComponents() error = %v", err)
			}

			// Before informers start, handlers are not synced yet, so WaitForHandlersSync
			// must fail with context.Canceled rather than returning nil.
			canceledCtx, cancelNow := context.WithCancel(ctx)
			cancelNow()
			if err := comps.WaitForHandlersSync(canceledCtx); !errors.Is(err, context.Canceled) {
				t.Fatalf("WaitForHandlersSync() before Start error = %v, want context.Canceled", err)
			}

			informerFactory.StartWithContext(ctx)
			if err := informerFactory.WaitForCacheSyncWithContext(ctx).AsError(); err != nil {
				t.Fatalf("WaitForCacheSyncWithContext() error = %v", err)
			}
			if err := comps.WaitForHandlersSync(ctx); err != nil {
				t.Fatalf("WaitForHandlersSync() after Start error = %v", err)
			}

			// After WaitForHandlersSync completes, extendedResourceCache, resourceClaimCache,
			// and resourceSliceTracker must all reflect the initial objects.
			gotClass := comps.draManager.DeviceClassResolver().GetDeviceClass(v1.ResourceName("example.com/gpu"))
			if diff := cmp.Diff(tt.wantDeviceClass, gotClass); diff != "" {
				t.Errorf("DeviceClassResolver().GetDeviceClass() mismatch (-want +got):\n%s", diff)
			}
			gotClaim, err := comps.draManager.ResourceClaims().Get("default", "test-claim")
			if err != nil {
				t.Fatalf("ResourceClaims().Get() error = %v", err)
			}
			if diff := cmp.Diff(claim, gotClaim); diff != "" {
				t.Errorf("ResourceClaims().Get() mismatch (-want +got):\n%s", diff)
			}
			gotSlices, err := comps.draManager.ResourceSlices().ListWithDeviceTaintRules()
			if err != nil {
				t.Fatalf("ResourceSlices().ListWithDeviceTaintRules() error = %v", err)
			}
			if len(gotSlices) != 1 || gotSlices[0].Name != slice.Name {
				t.Errorf("ResourceSlices().ListWithDeviceTaintRules() = %v, want single slice %q", gotSlices, slice.Name)
			}
		})
	}
}

func TestNewFrameworkMapDRAIsolation(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	claimUID := types.UID("claim-uid-1")
	claim := st.MakeResourceClaim().
		Name("shared-claim").
		Namespace("default").
		UID(string(claimUID)).
		Obj()
	claim.ResourceVersion = "1"

	client := fake.NewClientset(claim)
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	const secondSchedulerName = "second-scheduler"
	comps, err := NewFrameworkComponents(
		ctx,
		client,
		informerFactory,
		WithProfiles(
			schedulerapi.KubeSchedulerProfile{
				SchedulerName: v1.DefaultSchedulerName,
				Plugins: &schedulerapi.Plugins{
					QueueSort: schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "PrioritySort"}}},
					Bind:      schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "DefaultBinder"}}},
				},
			},
			schedulerapi.KubeSchedulerProfile{
				SchedulerName: secondSchedulerName,
				Plugins: &schedulerapi.Plugins{
					QueueSort: schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "PrioritySort"}}},
					Bind:      schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "DefaultBinder"}}},
				},
			},
		),
	)
	if err != nil {
		t.Fatalf("NewFrameworkComponents() error = %v", err)
	}

	recorderFactory := func(string) events.EventRecorderLogger {
		return events.NewFakeRecorder(10)
	}
	profilesA, err := NewFrameworkMap(ctx, comps, recorderFactory, internalcache.NewEmptySnapshot())
	if err != nil {
		t.Fatalf("NewFrameworkMap() A error = %v", err)
	}
	profilesB, err := NewFrameworkMap(ctx, comps, recorderFactory, internalcache.NewEmptySnapshot())
	if err != nil {
		t.Fatalf("NewFrameworkMap() B error = %v", err)
	}

	informerFactory.StartWithContext(ctx)
	if err := informerFactory.WaitForCacheSyncWithContext(ctx).AsError(); err != nil {
		t.Fatalf("WaitForCacheSyncWithContext() error = %v", err)
	}
	if err := comps.WaitForHandlersSync(ctx); err != nil {
		t.Fatalf("WaitForHandlersSync() error = %v", err)
	}

	draA1 := profilesA[v1.DefaultSchedulerName].SharedDRAManager()
	draA2 := profilesA[secondSchedulerName].SharedDRAManager()
	draB := profilesB[v1.DefaultSchedulerName].SharedDRAManager()

	allocatedClaim := st.FromResourceClaim(claim).Obj()
	allocatedClaim.Status.Allocation = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{
				{
					Request: "req-1",
					Driver:  "driver.example.com",
					Pool:    "pool-1",
					Device:  "dev-1",
				},
			},
		},
	}
	wantDevice := structured.MakeDeviceID("driver.example.com", "pool-1", "dev-1")

	if err := draA1.ResourceClaims().SignalClaimPendingAllocation(claimUID, allocatedClaim); err != nil {
		t.Fatalf("draA1 SignalClaimPendingAllocation() error = %v", err)
	}

	// Profiles within the same NewFrameworkMap must share in-flight allocations.
	for name, mgr := range map[string]fwk.SharedDRAManager{"draA1": draA1, "draA2": draA2} {
		gotPending := mgr.ResourceClaims().GetPendingAllocation(claimUID)
		if diff := cmp.Diff(allocatedClaim.Status.Allocation, gotPending); diff != "" {
			t.Errorf("%s GetPendingAllocation() mismatch (-want +got):\n%s", name, diff)
		}
		gotDevices, err := mgr.ResourceClaims().ListAllAllocatedDevices()
		if err != nil {
			t.Fatalf("%s ListAllAllocatedDevices() error = %v", name, err)
		}
		if diff := cmp.Diff(sets.New(wantDevice), gotDevices); diff != "" {
			t.Errorf("%s ListAllAllocatedDevices() mismatch (-want +got):\n%s", name, diff)
		}
	}

	// A separate NewFrameworkMap must have isolated in-flight allocations.
	if got := draB.ResourceClaims().GetPendingAllocation(claimUID); got != nil {
		t.Errorf("draB GetPendingAllocation() = %v, want nil", got)
	}
	gotDevicesB, err := draB.ResourceClaims().ListAllAllocatedDevices()
	if err != nil {
		t.Fatalf("draB ListAllAllocatedDevices() error = %v", err)
	}
	if diff := cmp.Diff(sets.New[structured.DeviceID](), gotDevicesB); diff != "" {
		t.Errorf("draB ListAllAllocatedDevices() mismatch (-want +got):\n%s", diff)
	}
}

type stubDRAManager struct {
	fwk.SharedDRAManager
}

func TestNewFrameworkMapCustomDRAManagerOverride(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	client := fake.NewClientset()
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	comps, err := NewFrameworkComponents(ctx, client, informerFactory)
	if err != nil {
		t.Fatalf("NewFrameworkComponents() error = %v", err)
	}

	customDRA := &stubDRAManager{}
	recorderFactory := func(string) events.EventRecorderLogger {
		return events.NewFakeRecorder(10)
	}
	profiles, err := NewFrameworkMap(
		ctx,
		comps,
		recorderFactory,
		internalcache.NewEmptySnapshot(),
		frameworkruntime.WithSharedDRAManager(customDRA),
	)
	if err != nil {
		t.Fatalf("NewFrameworkMap() error = %v", err)
	}

	if got := profiles[v1.DefaultSchedulerName].SharedDRAManager(); got != customDRA {
		t.Errorf("SharedDRAManager() = %T, want custom stubDRAManager", got)
	}
}
