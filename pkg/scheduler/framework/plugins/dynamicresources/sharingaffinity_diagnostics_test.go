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

package dynamicresources

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/events"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/utils/ptr"
)

const (
	filterDiagnosticMismatch    = "incompatible with the device's sharing affinity lock"
	filterDiagnosticUnsatisfied = "claim configuration does not produce a value for every sharing affinity extractor of the pool"
	filterDiagnosticUnknown     = "device sharing affinity is unknown because of legacy or inconsistent allocations; it can only be re-evaluated after all active allocations on the device are removed"
)

type diagnosticRecorder struct{ calls int }

type diagnosticLockLog struct {
	*testing.T
	tracker *claimTracker
	calls   int
}

func (l *diagnosticLockLog) Log(args ...interface{}) {
	l.calls++
	if l.tracker.allocatedDevices.mutex.TryLock() {
		l.tracker.allocatedDevices.mutex.Unlock()
	} else {
		l.Error("allocation/diagnostic logging ran under accounting lock")
	}
	if l.tracker.inFlightMutex.TryLock() {
		l.tracker.inFlightMutex.Unlock()
	} else {
		l.Error("allocation/diagnostic logging ran under in-flight lock")
	}
	l.T.Log(args...)
}

func (r *diagnosticRecorder) Eventf(runtime.Object, runtime.Object, string, string, string, string, ...interface{}) {
	r.calls++
}
func (r *diagnosticRecorder) WithLogger(klog.Logger) events.EventRecorderLogger { return r }

type diagnosticHandle struct {
	fwk.Handle
	recorder *diagnosticRecorder
}

func (h diagnosticHandle) EventRecorder() events.EventRecorderLogger { return h.recorder }

func diagnosticFilter(t *testing.T, ctx context.Context, snapshot structured.AllocatedState, slices []*resourceapi.ResourceSlice, claim *resourceapi.ResourceClaim, class *resourceapi.DeviceClass, nodeName string, excluded sets.Set[poolName], reasons map[poolName]string) *fwk.Status {
	t.Helper()
	allocator, err := structured.NewAllocator(ctx, structured.Features{SharingAffinity: true, ConsumableCapacity: true},
		snapshot, filterAffinityClasses{class}, slices, cel.NewCache(10, cel.Features{}))
	if err != nil {
		t.Fatalf("NewAllocator: %v", err)
	}
	cycle, state := filterAffinityState([]*resourceapi.ResourceClaim{claim}, allocator)
	state.affinitySlices, state.affinityExcluded, state.affinityExcludedReasons = slices, excluded, reasons
	node := framework.NewNodeInfo()
	node.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}})
	recorder := &diagnosticRecorder{}
	plugin := &DynamicResources{
		fts: feature.Features{EnableDRASharingAffinity: true, EnableDRAConsumableCapacity: true},
		fh:  diagnosticHandle{recorder: recorder},
	}
	status := plugin.Filter(ctx, cycle, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: "default"}}, node)
	if recorder.calls != 0 {
		t.Errorf("Filter emitted %d Events; scheduler must own attempt-level reporting", recorder.calls)
	}
	if !status.IsSuccess() && len(state.nodeAllocations) != 0 {
		t.Error("failed Filter published a node allocation")
	}
	if status.IsSuccess() {
		plan := state.nodeAllocations[nodeName]
		if len(plan.allocationResults) != 1 || len(plan.allocationResults[0].Devices.Results) != 1 {
			t.Errorf("successful Filter did not retain exactly one allocation: %+v", plan)
		}
	}
	return status
}

func TestSharingAffinityDiagnostics(t *testing.T) {
	for _, tc := range []struct {
		name, message, countKey string
		status                  fwk.Code
		cause                   error
	}{
		{"mismatch", filterDiagnosticMismatch, "mismatch", fwk.Unschedulable, nil},
		{"unsatisfied", filterDiagnosticUnsatisfied, "unsatisfied", fwk.Unschedulable, nil},
		{"unknown", filterDiagnosticUnknown, "unknown", fwk.Unschedulable, nil},
		{"missing-field", "missing field", "", fwk.Error, sharingaffinity.ErrMissingField},
		{"non-string", "cel expression returned non-string", "", fwk.Error, sharingaffinity.ErrNonStringReturn},
	} {
		t.Run(tc.name, func(t *testing.T) {
			setSharingAffinityAccountingGates(t, true, true)
			ctx, logs := sharingAffinityTestLogContext(t, 5)
			claim, slice, class := filterAffinityFixture()
			snapshot := structured.AllocatedState{
				DeviceSharingAffinity: structured.SharingAffinityState{DeviceStates: structured.AffinityStateCollection{}},
			}
			for i := 0; i < 3; i++ {
				name := fmt.Sprintf("device-%d", i)
				if i == 0 {
					slice.Spec.Devices[0].Name = name
				} else {
					slice.Spec.Devices = append(slice.Spec.Devices, resourceapi.Device{Name: name, AllowMultipleAllocations: ptr.To(true)})
				}
				id := structured.MakeDeviceID(slice.Spec.Driver, slice.Spec.Pool.Name, name)
				switch tc.name {
				case "mismatch":
					snapshot.DeviceSharingAffinity.DeviceStates[id] = structured.DeviceAffinityState{Status: structured.AffinityStatusLocked, LockedAffinity: map[string]string{"subnet": "other-secret"}}
				case "unknown":
					snapshot.DeviceSharingAffinity.DeviceStates[id] = structured.DeviceAffinityState{Status: structured.AffinityStatusUnreconstructable}
				}
			}
			switch tc.name {
			case "unsatisfied":
				slice.Spec.SharingAffinity = append(slice.Spec.SharingAffinity, sharingAffinityTestExtractor("sibling", `""`))
			case "missing-field":
				slice.Spec.SharingAffinity[0].Expression = `parameters.missing`
			case "non-string":
				slice.Spec.SharingAffinity[0].Expression = `parameters.subnet`
				claim.Spec.Devices.Config[0] = sharingAffinityTestConfig(sharingAffinityTestDriver, `{"apiVersion":"config.example.com/v1","kind":"Network","subnet":42}`)
			}
			before := snapshot.DeviceSharingAffinity.DeviceStates.Clone()
			for range 2 {
				status := diagnosticFilter(t, ctx, snapshot, []*resourceapi.ResourceSlice{slice}, claim, class, "node", nil, nil)
				if status.Code() != tc.status || !strings.Contains(status.Message(), tc.message) {
					t.Errorf("Filter = %v, want %v with %q", status, tc.status, tc.message)
				}
				if tc.cause != nil && !errors.Is(status.AsError(), tc.cause) {
					t.Errorf("runtime error lost identity: %v, want %v", status.AsError(), tc.cause)
				}
				for _, secret := range []string{"opaque-secret", "other-secret", slice.Spec.SharingAffinity[0].Expression} {
					if strings.Contains(status.Message(), secret) || strings.Contains(logs.String(), secret) {
						t.Errorf("status or V(5) logs disclose %q", secret)
					}
				}
			}
			if !reflect.DeepEqual(snapshot.DeviceSharingAffinity.DeviceStates, before) {
				t.Error("Filter mutated lock snapshot")
			}
			if tc.countKey != "" {
				summaries := 0
				for _, entry := range logs.Data() {
					if entry.Message != "Sharing affinity rejected candidate devices" {
						continue
					}
					summaries++
					values := sharingAffinityTestLogValues(entry)
					for _, key := range []string{"mismatch", "unsatisfied", "unknown"} {
						want := 0
						if key == tc.countKey {
							want = 3
						}
						if got := values[key]; got != want {
							t.Errorf("summary %s = %v, want %d", key, got, want)
						}
					}
					for _, key := range []string{"pod", "node"} {
						if _, found := values[key]; !found {
							t.Errorf("summary omits %s", key)
						}
					}
				}
				if summaries != 2 {
					t.Errorf("summary count = %d, want one per Filter invocation (2), not per device", summaries)
				}
			}
		})
	}
	t.Run("two-nodes-rejected-and-feasible", func(t *testing.T) {
		setSharingAffinityAccountingGates(t, true, true)
		ctx, _ := sharingAffinityTestLogContext(t, 5)
		claim, rejected, class := filterAffinityFixture()
		rejected.Spec.AllNodes, rejected.Spec.NodeName = nil, ptr.To("rejected")
		feasible := rejected.DeepCopy()
		feasible.Name, feasible.Spec.Pool.Name, feasible.Spec.NodeName = "feasible", "feasible", ptr.To("feasible")
		snapshot := structured.AllocatedState{
			DeviceSharingAffinity: structured.SharingAffinityState{
				DeviceStates: structured.AffinityStateCollection{
					structured.MakeDeviceID(rejected.Spec.Driver, rejected.Spec.Pool.Name, "device"): {
						Status: structured.AffinityStatusLocked, LockedAffinity: map[string]string{"subnet": "other-secret"},
					},
				},
			},
		}
		for _, name := range []string{"rejected", "feasible"} {
			status := diagnosticFilter(t, ctx, snapshot, []*resourceapi.ResourceSlice{rejected, feasible}, claim, class, name, nil, nil)
			if name == "rejected" {
				if status.Code() != fwk.Unschedulable || !strings.Contains(status.Message(), filterDiagnosticMismatch) {
					t.Errorf("rejected node = %v", status)
				}
			} else if !status.IsSuccess() {
				t.Errorf("alternative node became infeasible: %v", status)
			}
		}
	})
	t.Run("unknown/partial-then-full-drain", testDiagnosticOwnerDrain)
	for _, reason := range []string{"notReady", "listerAhead"} {
		t.Run("pending-not-unknown/"+reason, func(t *testing.T) { testDiagnosticPending(t, reason) })
	}
}

func testDiagnosticOwnerDrain(t *testing.T) {
	setSharingAffinityAccountingGates(t, true, true)
	c := newSharingAffinityAccountingTracker(t, true)
	a, b := sharingAffinityAccountingClaim("a", "blue"), sharingAffinityAccountingClaim("b", "red")
	c.allocatedDevices.onAdd(a)
	c.allocatedDevices.onAdd(b)
	lockLog := &diagnosticLockLog{T: t, tracker: c}
	logger := ktesting.NewLogger(lockLog, ktesting.NewConfig(ktesting.Verbosity(5)))
	ctx := klog.NewContext(t.Context(), logger)
	claim, _, class := filterAffinityFixture()
	claim.Spec.Devices.Config = sharingAffinityAccountingClaim("pending", "blue").Spec.Devices.Config
	slices := newSharingAffinityAccounting().slices.(accountingSliceLister).slices
	for _, stage := range []string{"both-owners", "partial-drain", "full-drain"} {
		t.Run(stage, func(t *testing.T) {
			switch stage {
			case "partial-drain":
				c.allocatedDevices.onDelete(b)
			case "full-drain":
				c.allocatedDevices.onDelete(a)
			}
			before := gatherSharingAffinityAccounting(t, c)
			revision := c.allocatedDevices.Revision()
			status := diagnosticFilter(t, ctx, *before, slices, claim, class, "node", nil, nil)
			if stage == "full-drain" {
				if !status.IsSuccess() {
					t.Errorf("fully drained device = %v, want Success", status)
				}
			} else if status.Code() != fwk.Unschedulable || !strings.Contains(status.Message(), filterDiagnosticUnknown) {
				t.Errorf("%s Filter = %v, want sticky unknown diagnostic", stage, status)
			}
			after := gatherSharingAffinityAccounting(t, c)
			if !reflect.DeepEqual(before, after) || revision != c.allocatedDevices.Revision() {
				t.Error("diagnostic Filter changed owner locks, revision or capacity")
			}
		})
	}
	if lockLog.calls == 0 {
		t.Error("lock assertions did not observe allocation logs")
	}
}

func testDiagnosticPending(t *testing.T, reason string) {
	slices := poolEventsSlices()
	c, lister := newPoolEventsUnitTracker(t, slices...)
	c.publishMetadata(true, nil)
	key := poolName{accountingDriver, accountingPool}
	poolID := schedulerapi.PoolID{Driver: key.driver, Pool: key.pool}
	device := structured.MakeDeviceID(key.driver, key.pool, accountingDevice)
	owner := sharingAffinityAccountingClaim("legacy", "blue")
	owner.Spec.Devices.Config = nil
	// Seed the same captured-state seam as the readiness regression tests. Both
	// unknown ownership and pending readiness must be present, not just a string.
	a := c.allocatedDevices
	a.mutex.Lock()
	a.affinityDevices[device] = &deviceAffinity{ambiguous: true}
	view := c.metadata.pools[key]
	recordKey := ownerPoolKey{uid: owner.UID, pool: key}
	a.owners[recordKey] = &ownerPoolRecord{claim: owner, claimRV: owner.ResourceVersion, view: view, epoch: view.epoch, pending: reason == "notReady"}
	a.mutex.Unlock()
	if reason == "listerAhead" {
		for i := range slices {
			slices[i] = slices[i].DeepCopy()
			slices[i].Spec.Pool.Generation++
		}
		lister.set(slices...)
	}
	claim, _, class := filterAffinityFixture()
	claim.Spec.Devices.Config = sharingAffinityAccountingClaim("pending", "blue").Spec.Devices.Config
	ctx, _ := sharingAffinityTestLogContext(t, 5)
	pending, err := c.gatherAllocatedStateForSlices(slices)
	if err != nil {
		t.Fatal(err)
	}
	if pending.reasons[key] != reason || !pending.excluded.Has(key) || !pending.state.DeviceSharingAffinity.StalePools.Has(poolID) {
		t.Fatalf("snapshot exclusion = %v (%v), want %s", pending.excluded, pending.reasons, reason)
	}
	if pending.state.DeviceSharingAffinity.DeviceStates[device].Status != structured.AffinityStatusUnreconstructable {
		t.Fatal("pending fixture lost unknown ownership")
	}
	status := diagnosticFilter(t, ctx, *pending.state, slices, claim, class, "node", pending.excluded, pending.reasons)
	want := fmt.Sprintf("sharing affinity state for pool %s/%s not yet reconciled", key.driver, key.pool)
	if status.Code() != fwk.Unschedulable || status.Message() != want || strings.Contains(status.Message(), "unknown") {
		t.Errorf("pending Filter = %v, want %q (readiness takes precedence)", status, want)
	}
	// Restore the captured publication synchronously. No timing-dependent waits.
	slices = poolEventsSlices()
	lister.set(slices...)
	a.mutex.Lock()
	a.owners[recordKey].pending = false
	a.mutex.Unlock()
	ready, err := c.gatherAllocatedStateForSlices(slices)
	if err != nil {
		t.Fatal(err)
	}
	if ready.excluded.Has(key) {
		t.Fatalf("settled fixture still excluded: %v", ready.reasons)
	}
	status = diagnosticFilter(t, ctx, *ready.state, slices, claim, class, "node", ready.excluded, ready.reasons)
	if status.Code() != fwk.Unschedulable || !strings.Contains(status.Message(), filterDiagnosticUnknown) || strings.Contains(status.Message(), "not yet reconciled") {
		t.Errorf("settled unknown Filter = %v, want unknown diagnostic", status)
	}
}
