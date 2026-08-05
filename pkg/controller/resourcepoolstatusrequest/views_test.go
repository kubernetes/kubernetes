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

package resourcepoolstatusrequest

import (
	"strings"
	"testing"

	resourcev1 "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
)

func qty(s string) resource.Quantity { return resource.MustParse(s) }

func counterSet(name string, counters map[string]string) resourcev1.CounterSet {
	c := make(map[string]resourcev1.Counter, len(counters))
	for k, v := range counters {
		c[k] = resourcev1.Counter{Value: qty(v)}
	}
	return resourcev1.CounterSet{Name: name, Counters: c}
}

func consumes(set string, counters map[string]string) resourcev1.DeviceCounterConsumption {
	c := make(map[string]resourcev1.Counter, len(counters))
	for k, v := range counters {
		c[k] = resourcev1.Counter{Value: qty(v)}
	}
	return resourcev1.DeviceCounterConsumption{CounterSet: set, Counters: c}
}

func inUseSet(names ...string) map[string]struct{} {
	s := make(map[string]struct{}, len(names))
	for _, n := range names {
		s[n] = struct{}{}
	}
	return s
}

// partitionDevice builds a counter-consuming device with its grouping attribute
// and partition type already resolved (as the controller does before calling
// the view functions).
func partitionDevice(name, partitionType string, cost ...resourcev1.DeviceCounterConsumption) deviceRecord {
	return deviceRecord{
		name:             name,
		partitionAttr:    "gpu.example.com/profile",
		partitionType:    partitionType,
		hasPartitionType: true,
		consumesCounters: cost,
	}
}

func TestComputePartitionSummary(t *testing.T) {
	// One 80Gi counter set. Full costs 80Gi, Half costs 40Gi.
	sc := []resourcev1.CounterSet{counterSet("gpu-0", map[string]string{"memory": "80Gi"})}

	testCases := map[string]struct {
		devices []deviceRecord
		inUse   map[string]struct{}
		want    map[string]int32 // type -> allocatable
		total   map[string]int32 // type -> total
		wantErr string
	}{
		"all-fresh-independent-per-type": {
			devices: []deviceRecord{
				partitionDevice("full-0", "Full", consumes("gpu-0", map[string]string{"memory": "80Gi"})),
				partitionDevice("half-0", "Half", consumes("gpu-0", map[string]string{"memory": "40Gi"})),
				partitionDevice("half-1", "Half", consumes("gpu-0", map[string]string{"memory": "40Gi"})),
			},
			inUse: inUseSet(),
			// Each type measured independently against the full 80Gi baseline.
			want:  map[string]int32{"Full": 1, "Half": 2},
			total: map[string]int32{"Full": 1, "Half": 2},
		},
		"in-use-full-consumes-counter": {
			devices: []deviceRecord{
				partitionDevice("full-0", "Full", consumes("gpu-0", map[string]string{"memory": "80Gi"})),
				partitionDevice("half-0", "Half", consumes("gpu-0", map[string]string{"memory": "40Gi"})),
				partitionDevice("half-1", "Half", consumes("gpu-0", map[string]string{"memory": "40Gi"})),
			},
			inUse: inUseSet("full-0"),
			// full-0 in use drains the counter; no headroom for either type.
			want:  map[string]int32{"Full": 0, "Half": 0},
			total: map[string]int32{"Full": 1, "Half": 2},
		},
		"fresh-device-clamp-binds-before-counter": {
			devices: []deviceRecord{
				partitionDevice("full-0", "Full", consumes("gpu-1", map[string]string{"memory": "80Gi"})),
				partitionDevice("full-1", "Full", consumes("gpu-1", map[string]string{"memory": "80Gi"})),
				partitionDevice("full-2", "Full", consumes("gpu-1", map[string]string{"memory": "80Gi"})),
			},
			inUse: inUseSet(),
			// 800Gi counter fits 10 Fulls, but only 3 fresh devices exist.
			want:  map[string]int32{"Full": 3},
			total: map[string]int32{"Full": 3},
		},
		// A counter-consuming device whose slice resolved a grouping attribute
		// but that lacks the attribute value is flagged; a non-partition device
		// missing it would instead be skipped.
		"missing-attribute": {
			devices: []deviceRecord{
				partitionDevice("full-0", "Full", consumes("gpu-0", map[string]string{"memory": "80Gi"})),
				{
					name:             "mystery",
					partitionAttr:    "gpu.example.com/profile",
					consumesCounters: []resourcev1.DeviceCounterConsumption{consumes("gpu-0", map[string]string{"memory": "40Gi"})},
					hasPartitionType: false,
				},
			},
			inUse:   inUseSet(),
			wantErr: prefixPartitionTypeMissing,
		},
		"cost-mismatch-within-type": {
			devices: []deviceRecord{
				partitionDevice("half-0", "Half", consumes("gpu-0", map[string]string{"memory": "40Gi"})),
				partitionDevice("half-1", "Half", consumes("gpu-0", map[string]string{"memory": "50Gi"})),
			},
			inUse:   inUseSet(),
			wantErr: prefixPartitionCostMismatch,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			in := poolViewInput{
				driver:         "gpu.example.com",
				poolName:       "pool-0",
				devices:        tc.devices,
				sharedCounters: chooseCounters(name, sc),
				inUse:          tc.inUse,
			}
			got, gotErr := computePartitionSummary(in)
			if tc.wantErr != "" {
				if !strings.HasPrefix(gotErr, tc.wantErr) {
					t.Fatalf("want error prefix %q, got %q", tc.wantErr, gotErr)
				}
				return
			}
			if gotErr != "" {
				t.Fatalf("unexpected error: %s", gotErr)
			}
			for _, p := range got {
				if want, ok := tc.want[p.Type]; ok && ptr.Deref(p.Allocatable, 0) != want {
					t.Errorf("type %s: allocatable = %d, want %d", p.Type, ptr.Deref(p.Allocatable, 0), want)
				}
				if want, ok := tc.total[p.Type]; ok && ptr.Deref(p.Total, 0) != want {
					t.Errorf("type %s: total = %d, want %d", p.Type, ptr.Deref(p.Total, 0), want)
				}
			}
			if len(got) != len(tc.total) {
				t.Errorf("got %d partition types, want %d", len(got), len(tc.total))
			}
		})
	}
}

// chooseCounters gives the 800Gi baseline only to the fresh-device-clamp case.
func chooseCounters(name string, dflt []resourcev1.CounterSet) []resourcev1.CounterSet {
	if strings.Contains(name, "clamp") {
		return []resourcev1.CounterSet{counterSet("gpu-1", map[string]string{"memory": "800Gi"})}
	}
	return dflt
}

// A device drawing from two counter sets is bounded by the tighter one.
func TestComputePartitionSummary_MultiCounterSet(t *testing.T) {
	sc := []resourcev1.CounterSet{
		counterSet("mem", map[string]string{"memory": "160Gi"}),
		counterSet("cores", map[string]string{"sm": "8"}),
	}
	full := func(name string) deviceRecord {
		return partitionDevice(name, "Full",
			consumes("mem", map[string]string{"memory": "80Gi"}),
			consumes("cores", map[string]string{"sm": "8"}),
		)
	}
	in := poolViewInput{
		driver: "gpu.example.com", poolName: "p", sharedCounters: sc,
		devices: []deviceRecord{full("full-0"), full("full-1")},
		inUse:   inUseSet(),
	}
	got, err := computePartitionSummary(in)
	if err != "" {
		t.Fatalf("unexpected error: %s", err)
	}
	// memory allows 2 (160/80), but the sm counter allows only 1 (8/8) -> min is 1.
	if len(got) != 1 || ptr.Deref(got[0].Allocatable, 0) != 1 {
		t.Errorf("allocatable = %+v, want 1 (bound by the sm counter)", got)
	}
}

// A device that is both partition-typed and shareable yields both views.
func TestComputePoolViews_Hybrid(t *testing.T) {
	in := poolViewInput{
		driver: "gpu.example.com", poolName: "p",
		sharedCounters: []resourcev1.CounterSet{counterSet("gpu-0", map[string]string{"memory": "80Gi"})},
		devices: []deviceRecord{{
			name:             "full-0",
			partitionAttr:    "gpu.example.com/profile",
			partitionType:    "Full",
			hasPartitionType: true,
			consumesCounters: []resourcev1.DeviceCounterConsumption{consumes("gpu-0", map[string]string{"memory": "80Gi"})},
			allowMultiple:    true,
			capacity:         map[resourcev1.QualifiedName]resourcev1.DeviceCapacity{"memory": {Value: qty("80Gi")}},
		}},
		inUse: inUseSet(),
	}
	ps, sh, err := computePoolViews(in)
	if err != "" {
		t.Fatalf("unexpected error: %s", err)
	}
	if len(ps) != 1 || sh == nil {
		t.Errorf("hybrid: want partitionSummary set and shareableSummary set; got ps=%d sh=%v", len(ps), sh)
	}
	if sh != nil && ptr.Deref(sh.FullyAvailableDevices, 0) != 1 {
		t.Errorf("shareable fullyAvailable = %d, want 1", ptr.Deref(sh.FullyAvailableDevices, 0))
	}
}

func TestComputeShareableSummary(t *testing.T) {
	devices := []deviceRecord{
		{
			name:          "shareable-0",
			allowMultiple: true,
			capacity:      map[resourcev1.QualifiedName]resourcev1.DeviceCapacity{"memory": {Value: qty("40Gi")}},
		},
		{
			name:          "shareable-1",
			allowMultiple: true,
			capacity:      map[resourcev1.QualifiedName]resourcev1.DeviceCapacity{"memory": {Value: qty("40Gi")}},
		},
		{name: "plain", allowMultiple: false},
	}
	in := poolViewInput{
		driver:           "gpu.example.com",
		poolName:         "pool-0",
		devices:          devices,
		inUse:            inUseSet("shareable-0"),
		consumedCapacity: map[resourcev1.QualifiedName]resource.Quantity{"memory": qty("10Gi")},
	}

	got, err := computeShareableSummary(in)
	if err != "" {
		t.Fatalf("unexpected error: %s", err)
	}
	if got == nil {
		t.Fatal("expected a shareable summary")
	}
	if ptr.Deref(got.FullyAvailableDevices, 0) != 1 {
		t.Errorf("fullyAvailableDevices = %d, want 1", ptr.Deref(got.FullyAvailableDevices, 0))
	}
	if ptr.Deref(got.PartiallyAvailableDevices, 0) != 1 {
		t.Errorf("partiallyAvailableDevices = %d, want 1", ptr.Deref(got.PartiallyAvailableDevices, 0))
	}
	if len(got.Capacity) != 1 {
		t.Fatalf("want 1 capacity key, got %d", len(got.Capacity))
	}
	cap := got.Capacity[0]
	if cap.Total.String() != "80Gi" || cap.Consumed.String() != "10Gi" || cap.Available.Cmp(qty("70Gi")) != 0 {
		t.Errorf("capacity = %+v, want total=80Gi consumed=10Gi available=70Gi", cap)
	}
}

// Shareable devices advertising multiple capacities: each key is summed across
// devices and reported independently, sorted by name.
func TestComputeShareableSummary_MultipleCapacities(t *testing.T) {
	devices := []deviceRecord{
		{
			name:          "shareable-0",
			allowMultiple: true,
			capacity: map[resourcev1.QualifiedName]resourcev1.DeviceCapacity{
				"memory": {Value: qty("40Gi")},
				"cores":  {Value: qty("4")},
			},
		},
		{
			name:          "shareable-1",
			allowMultiple: true,
			capacity: map[resourcev1.QualifiedName]resourcev1.DeviceCapacity{
				"memory": {Value: qty("40Gi")},
				"cores":  {Value: qty("4")},
			},
		},
	}
	in := poolViewInput{
		driver:   "gpu.example.com",
		poolName: "pool-0",
		devices:  devices,
		inUse:    inUseSet("shareable-0"),
		consumedCapacity: map[resourcev1.QualifiedName]resource.Quantity{
			"memory": qty("10Gi"),
			"cores":  qty("1"),
		},
	}

	got, err := computeShareableSummary(in)
	if err != "" {
		t.Fatalf("unexpected error: %s", err)
	}
	if got == nil {
		t.Fatal("expected a shareable summary")
	}
	if ptr.Deref(got.FullyAvailableDevices, 0) != 1 || ptr.Deref(got.PartiallyAvailableDevices, 0) != 1 {
		t.Errorf("full/partial = %d/%d, want 1/1", ptr.Deref(got.FullyAvailableDevices, 0), ptr.Deref(got.PartiallyAvailableDevices, 0))
	}
	byKey := map[string][3]string{}
	for _, c := range got.Capacity {
		byKey[c.Name] = [3]string{c.Total.String(), c.Consumed.String(), c.Available.String()}
	}
	if want := [3]string{"8", "1", "7"}; byKey["cores"] != want {
		t.Errorf("cores {total,consumed,available} = %v, want %v", byKey["cores"], want)
	}
	if want := [3]string{"80Gi", "10Gi", "70Gi"}; byKey["memory"] != want {
		t.Errorf("memory {total,consumed,available} = %v, want %v", byKey["memory"], want)
	}
}

func TestComputeShareableSummary_NoShareableDevices(t *testing.T) {
	in := poolViewInput{devices: []deviceRecord{{name: "plain"}}, inUse: inUseSet()}
	got, err := computeShareableSummary(in)
	if err != "" || got != nil {
		t.Errorf("want nil summary and no error, got %+v / %q", got, err)
	}
}

// A resolved grouping attribute produces a partitionSummary; without one, a
// partitionable pool reports no partition view (option 2, no counterSets fallback).
func TestComputePoolViews_PartitionAttributeGatesView(t *testing.T) {
	sc := []resourcev1.CounterSet{counterSet("gpu-0", map[string]string{"memory": "80Gi"})}

	// With a grouping attribute -> partitionSummary.
	typed := poolViewInput{
		driver: "gpu.example.com", poolName: "p", sharedCounters: sc,
		devices: []deviceRecord{partitionDevice("d0", "Full", consumes("gpu-0", map[string]string{"memory": "80Gi"}))},
		inUse:   inUseSet(),
	}
	ps, _, err := computePoolViews(typed)
	if err != "" {
		t.Fatalf("typed: unexpected error %s", err)
	}
	if len(ps) == 0 {
		t.Errorf("typed pool: want partitionSummary set, got ps=%d", len(ps))
	}

	// Without a grouping attribute -> no partition view.
	noAttr := poolViewInput{
		driver: "gpu.example.com", poolName: "p", sharedCounters: sc,
		devices: []deviceRecord{{name: "d0", consumesCounters: []resourcev1.DeviceCounterConsumption{consumes("gpu-0", map[string]string{"memory": "10Gi"})}}},
		inUse:   inUseSet(),
	}
	ps2, _, err := computePoolViews(noAttr)
	if err != "" {
		t.Fatalf("no-attr: unexpected error %s", err)
	}
	if ps2 != nil {
		t.Errorf("no-attr pool: want no partition view, got ps=%d", len(ps2))
	}
}

func TestResolveDevicePartitions(t *testing.T) {
	const driver = "gpu.example.com"
	// A device carrying profile=Full and shape=Small, so either attribute resolves.
	deviceAttrs := map[resourcev1.QualifiedName]resourcev1.DeviceAttribute{
		"gpu.example.com/profile": {StringValue: ptr.To("Full")},
		"gpu.example.com/shape":   {StringValue: ptr.To("Small")},
	}
	testCases := map[string]struct {
		requestAttr *string
		devices     []deviceRecord
		wantAttr    []string // resolved partitionAttr per device
		wantType    []string // resolved partitionType per device ("" when unresolved)
	}{
		"slice declaration wins over request default": {
			requestAttr: ptr.To("gpu.example.com/shape"),
			devices:     []deviceRecord{{name: "d0", partitionAttr: "gpu.example.com/profile", attributes: deviceAttrs}},
			wantAttr:    []string{"gpu.example.com/profile"},
			wantType:    []string{"Full"},
		},
		"request default fills devices when no slice declares": {
			requestAttr: ptr.To("gpu.example.com/shape"),
			devices:     []deviceRecord{{name: "d0", attributes: deviceAttrs}},
			wantAttr:    []string{"gpu.example.com/shape"},
			wantType:    []string{"Small"},
		},
		"request default ignored when any slice declares": {
			requestAttr: ptr.To("gpu.example.com/shape"),
			devices: []deviceRecord{
				{name: "d0", partitionAttr: "gpu.example.com/profile", attributes: deviceAttrs},
				{name: "d1", attributes: deviceAttrs},
			},
			wantAttr: []string{"gpu.example.com/profile", ""},
			wantType: []string{"Full", ""},
		},
		// liggitt's case: slices declaring differing attributes are legitimate,
		// so each device keeps its own rather than being flagged a conflict.
		"differing slice attributes are each kept": {
			devices: []deviceRecord{
				{name: "d0", partitionAttr: "gpu.example.com/profile", attributes: deviceAttrs},
				{name: "d1", partitionAttr: "gpu.example.com/shape", attributes: deviceAttrs},
			},
			wantAttr: []string{"gpu.example.com/profile", "gpu.example.com/shape"},
			wantType: []string{"Full", "Small"},
		},
		"neither slice nor request names one": {
			devices:  []deviceRecord{{name: "d0", attributes: deviceAttrs}},
			wantAttr: []string{""},
			wantType: []string{""},
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			in := poolViewInput{driver: driver, devices: tc.devices}
			resolveDevicePartitions(tc.requestAttr, &in)
			for i := range in.devices {
				d := in.devices[i]
				if d.partitionAttr != tc.wantAttr[i] {
					t.Errorf("device %d: partitionAttr = %q, want %q", i, d.partitionAttr, tc.wantAttr[i])
				}
				gotType := ""
				if d.hasPartitionType {
					gotType = d.partitionType
				}
				if gotType != tc.wantType[i] {
					t.Errorf("device %d: type = %q (has=%v), want %q", i, d.partitionType, d.hasPartitionType, tc.wantType[i])
				}
			}
		})
	}
}

func TestResolvePartitionType(t *testing.T) {
	attrs := map[resourcev1.QualifiedName]resourcev1.DeviceAttribute{
		"profile":                     {StringValue: ptr.To("Full")},
		"other.com/x":                 {StringValue: ptr.To("ignore")},
		"gpu.example.com/numeric-key": {IntValue: ptr.To(int64(3))},
	}
	// Bare "profile" resolves against the driver domain.
	if v, ok := resolvePartitionType("gpu.example.com", attrs, "gpu.example.com/profile"); !ok || v != "Full" {
		t.Errorf("driver-domain resolution: got (%q,%v), want (Full,true)", v, ok)
	}
	// A non-string attribute is treated as absent.
	if _, ok := resolvePartitionType("gpu.example.com", attrs, "gpu.example.com/numeric-key"); ok {
		t.Error("non-string attribute should resolve as absent")
	}
	// Unknown attribute.
	if _, ok := resolvePartitionType("gpu.example.com", attrs, "gpu.example.com/missing"); ok {
		t.Error("missing attribute should resolve as absent")
	}
	// A foreign-domain name only matches an explicit key, never a bare one.
	if v, ok := resolvePartitionType("gpu.example.com", attrs, "other.com/x"); !ok || v != "ignore" {
		t.Errorf("foreign-domain resolution: got (%q,%v), want (ignore,true)", v, ok)
	}
	// When both the explicit and bare form of the same attribute are present,
	// the explicit key wins so the result does not depend on map iteration order.
	both := map[resourcev1.QualifiedName]resourcev1.DeviceAttribute{
		"profile":                 {StringValue: ptr.To("Bare")},
		"gpu.example.com/profile": {StringValue: ptr.To("Explicit")},
	}
	for range 20 {
		if v, ok := resolvePartitionType("gpu.example.com", both, "gpu.example.com/profile"); !ok || v != "Explicit" {
			t.Fatalf("colliding keys: got (%q,%v), want (Explicit,true)", v, ok)
		}
	}
}
