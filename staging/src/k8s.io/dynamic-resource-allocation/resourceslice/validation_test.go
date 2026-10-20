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

package resourceslice

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
)

func TestValidateDriverResources(t *testing.T) {
	const (
		driverName = "driver.example.com"
		otherName  = "other.example.com"
		poolName   = "pool"
		deviceName = "device"
	)

	device := func(modify func(*resourceapi.Device)) resourceapi.Device {
		d := resourceapi.Device{Name: deviceName}
		modify(&d)
		return d
	}
	resources := func(d resourceapi.Device) *DriverResources {
		return &DriverResources{
			Pools: map[string]Pool{
				poolName: {
					Slices: []Slice{
						{Devices: []resourceapi.Device{d}},
					},
				},
			},
		}
	}

	// wantErr is the exact error message expected from validateDriverResources,
	// empty if no error is expected.
	for name, tc := range map[string]struct {
		device  resourceapi.Device
		wantErr string
	}{
		"unqualified-attribute": {
			device: device(func(d *resourceapi.Device) {
				d.Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"foo": {StringValue: new("bar")},
				}
			}),
		},
		"foreign-domain-attribute": {
			device: device(func(d *resourceapi.Device) {
				d.Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					resourceapi.QualifiedName(otherName + "/foo"): {StringValue: new("bar")},
				}
			}),
		},
		"driver-qualified-attribute": {
			device: device(func(d *resourceapi.Device) {
				d.Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					resourceapi.QualifiedName(driverName + "/foo"): {StringValue: new("bar")},
				}
			}),
			wantErr: `pool "pool": device "device": attribute "driver.example.com/foo": use unqualified "foo", the driver name is added implicitly`,
		},
		"unqualified-capacity": {
			device: device(func(d *resourceapi.Device) {
				d.Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
					"foo": {Value: resource.MustParse("1")},
				}
			}),
		},
		"foreign-domain-capacity": {
			device: device(func(d *resourceapi.Device) {
				d.Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
					resourceapi.QualifiedName(otherName + "/foo"): {Value: resource.MustParse("1")},
				}
			}),
		},
		"driver-qualified-capacity": {
			device: device(func(d *resourceapi.Device) {
				d.Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
					resourceapi.QualifiedName(driverName + "/foo"): {Value: resource.MustParse("1")},
				}
			}),
			wantErr: `pool "pool": device "device": capacity "driver.example.com/foo": use unqualified "foo", the driver name is added implicitly`,
		},
		// Attributes are validated before capacities, so the attribute error wins.
		"driver-qualified-attribute-and-unqualified-capacity": {
			device: device(func(d *resourceapi.Device) {
				d.Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					resourceapi.QualifiedName(driverName + "/foo"): {StringValue: new("bar")},
				}
				d.Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
					"foo": {Value: resource.MustParse("1")},
				}
			}),
			wantErr: `pool "pool": device "device": attribute "driver.example.com/foo": use unqualified "foo", the driver name is added implicitly`,
		},
		// Attributes are validated before capacities, so the attribute error wins
		// even though the capacity is also invalid.
		"driver-qualified-attribute-and-driver-qualified-capacity": {
			device: device(func(d *resourceapi.Device) {
				d.Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					resourceapi.QualifiedName(driverName + "/foo"): {StringValue: new("bar")},
				}
				d.Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
					resourceapi.QualifiedName(driverName + "/foo"): {Value: resource.MustParse("1")},
				}
			}),
			wantErr: `pool "pool": device "device": attribute "driver.example.com/foo": use unqualified "foo", the driver name is added implicitly`,
		},
		// A domain that merely has the driver name as a prefix or suffix of a
		// larger domain is not the same domain and must not be rejected.
		"suffix-sub-domain-attribute": {
			device: device(func(d *resourceapi.Device) {
				d.Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					resourceapi.QualifiedName(driverName + ".evil/foo"): {StringValue: new("bar")},
				}
			}),
		},
		"suffix-modified-domain-attribute": {
			device: device(func(d *resourceapi.Device) {
				d.Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					resourceapi.QualifiedName(driverName + "x/foo"): {StringValue: new("bar")},
				}
			}),
		},
		"prefix-sub-domain-attribute": {
			device: device(func(d *resourceapi.Device) {
				d.Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					resourceapi.QualifiedName("evil." + driverName + "/foo"): {StringValue: new("bar")},
				}
			}),
		},
		"prefix-modified-domain-attribute": {
			device: device(func(d *resourceapi.Device) {
				d.Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					resourceapi.QualifiedName("x" + driverName + "/foo"): {StringValue: new("bar")},
				}
			}),
		},
		// Cut splits on the first slash, so a multi-segment name after the
		// driver domain is still rejected.
		"driver-qualified-attribute-with-extra-slash": {
			device: device(func(d *resourceapi.Device) {
				d.Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					resourceapi.QualifiedName(driverName + "/a/b"): {StringValue: new("bar")},
				}
			}),
			wantErr: `pool "pool": device "device": attribute "driver.example.com/a/b": use unqualified "a/b", the driver name is added implicitly`,
		},
		// A name that merely has a "/" prefixed with the empty string, i.e. an
		// empty domain, is not the same as being qualified with the driver name
		// and must not be rejected here.
		"empty-domain-attribute": {
			device: device(func(d *resourceapi.Device) {
				d.Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"/foo": {StringValue: new("bar")},
				}
			}),
		},
	} {
		t.Run(name, func(t *testing.T) {
			err := validateDriverResources(driverName, resources(tc.device), nil)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("expected no error, got: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected error %q, got none", tc.wantErr)
			}
			if err.Error() != tc.wantErr {
				t.Fatalf("expected error:\n  %s\ngot:\n  %s", tc.wantErr, err.Error())
			}
		})
	}
}

// Validation must cover the complete pool, not just each API object.
func TestSharingAffinityPoolValidation(t *testing.T) {
	for _, tc := range sharingAffinityValidationCases(t) {
		t.Run(tc.name, func(t *testing.T) {
			for _, driver := range []string{"driver.example.com", ""} {
				t.Run("driver="+driver, func(t *testing.T) {
					for attempt := 0; attempt < 3; attempt++ {
						err := validatePool(driver, "pool", tc.pool, nil)
						checkSharingAffinityValidationError(t, err, tc)
					}
				})
			}
		})
	}
	t.Run("distinct-pools-same-names", func(t *testing.T) {
		pool := sharingAffinityPool(sharingAffinityEntry("subnet", `"blue"`))
		err := validateDriverResources("driver.example.com", &DriverResources{
			Pools: map[string]Pool{"pool-a": pool, "pool-b": pool},
		}, nil)
		if err != nil {
			t.Fatalf("independent pools rejected: %v", err)
		}
	})
	for _, tc := range sharingAffinityRatchetCases(t) {
		t.Run("ratchet-"+tc.name, func(t *testing.T) {
			err := validatePool("driver.example.com", "pool", tc.pool, tc.old)
			checkSharingAffinityValidationError(t, err, tc)
		})
	}
}

type sharingAffinityValidationCase struct {
	name      string
	pool      Pool
	old       []*resourceapi.ResourceSlice
	wantText  string
	wantExact string
	wantKind  error
}

func checkSharingAffinityValidationError(t *testing.T, err error, tc sharingAffinityValidationCase) {
	t.Helper()
	if tc.wantText == "" && tc.wantExact == "" && tc.wantKind == nil {
		if err != nil {
			t.Fatalf("valid pool rejected: %v", err)
		}
		return
	}
	if err == nil {
		t.Fatal("invalid pool accepted")
	}
	if tc.wantText != "" && !strings.Contains(err.Error(), tc.wantText) {
		t.Errorf("error = %v, want %q", err, tc.wantText)
	}
	if tc.wantExact != "" && err.Error() != tc.wantExact {
		t.Errorf("error = %q, want %q", err, tc.wantExact)
	}
	if tc.wantKind != nil && !errors.Is(err, tc.wantKind) {
		t.Errorf("error = %v, want errors.Is(%v)", err, tc.wantKind)
	}
	if tc.wantKind == nil && errors.Is(err, sharingaffinity.ErrInvalidDefinition) {
		t.Errorf("preflight error unexpectedly classified as CEL failure: %v", err)
	}
	if tc.wantKind == sharingaffinity.ErrInvalidDefinition {
		var compileErr *sharingaffinity.CompileError
		if !errors.As(err, &compileErr) {
			t.Errorf("compiler cause not reachable: %v", err)
		}
		var union []resourceapi.SharingAffinityExtractor
		for _, slice := range tc.pool.Slices {
			union = append(union, slice.SharingAffinity...)
		}
		want := sharingaffinity.ValidatePublishedExtractorSet(union, nil)
		inner := errors.Unwrap(err)
		if want == nil || inner == nil || inner.Error() != want.Error() || !errors.Is(inner, sharingaffinity.ErrInvalidDefinition) {
			t.Errorf("wrapped error = %v, want publisher compiler error %v", inner, want)
		}
		result := sharingaffinity.ValidateExpression(union[len(union)-1].Expression, environment.NewExpressions)
		if result.Error == nil {
			t.Error("admission compiler unexpectedly accepted invalid fixture")
		}
	}
}

func sharingAffinityEntry(name, expression string) resourceapi.SharingAffinityExtractor {
	return resourceapi.SharingAffinityExtractor{Name: name, APIGroup: "*", Kind: "*", Expression: expression}
}

func sharingAffinityPool(entries ...resourceapi.SharingAffinityExtractor) Pool {
	pool := Pool{AllNodes: true}
	for _, entry := range entries {
		pool.Slices = append(pool.Slices, Slice{SharingAffinity: []resourceapi.SharingAffinityExtractor{entry}})
	}
	return pool
}

func sharingAffinityValidationCases(t *testing.T) []sharingAffinityValidationCase {
	t.Helper()
	entry := sharingAffinityEntry("subnet", `"blue"`)
	at, over := poolCostBoundaryExtractors(t)
	cases := []sharingAffinityValidationCase{
		{name: "valid-multi-slice", pool: sharingAffinityPool(entry, sharingAffinityEntry("zone", `"west"`))},
		{name: "nil-affinity", pool: Pool{Slices: []Slice{{}}}},
		{name: "empty-affinity", pool: Pool{Slices: []Slice{{SharingAffinity: []resourceapi.SharingAffinityExtractor{}}}}},
		{name: "duplicate-within-slice", pool: Pool{Slices: []Slice{{SharingAffinity: []resourceapi.SharingAffinityExtractor{entry, entry}}}},
			wantExact: `pool "pool": duplicate sharing affinity key "subnet" in slice #0 (first declared in slice #0); key names must be unique across all slices of a pool`},
		{name: "duplicate-across-slices", pool: sharingAffinityPool(entry, entry),
			wantExact: `pool "pool": duplicate sharing affinity key "subnet" in slice #1 (first declared in slice #0); key names must be unique across all slices of a pool`},
		{name: "cost-at-limit-accepted", pool: sharingAffinityPool(at...)},
		{name: "cost-over-limit-across-individually-valid-slices", pool: sharingAffinityPool(over...), wantText: "combined estimated cost exceeds 1000000", wantKind: sharingaffinity.ErrCostExceeded},
		{name: "devices-counters-no-affinity-unchanged", pool: Pool{Slices: []Slice{{Devices: []resourceapi.Device{{Name: "device"}}, SharedCounters: []resourceapi.CounterSet{{Name: "counter"}}}}}},
	}
	for _, entries := range [][]resourceapi.SharingAffinityExtractor{at, over} {
		for _, entry := range entries {
			if err := sharingaffinity.ValidatePublishedExtractorSet([]resourceapi.SharingAffinityExtractor{entry}, nil); err != nil {
				t.Fatalf("boundary slice must be individually valid: %v", err)
			}
		}
	}
	for _, tc := range []struct {
		name string
		edit func(*resourceapi.SharingAffinityExtractor)
		text string
		kind error
	}{
		{"expression-length-at-limit", func(e *resourceapi.SharingAffinityExtractor) { e.Expression = `"` + strings.Repeat("a", 10238) + `"` }, "", nil},
		{"expression-length-over-limit", func(e *resourceapi.SharingAffinityExtractor) { e.Expression = `"` + strings.Repeat("a", 10239) + `"` }, "expression length 10241 exceeds 10240 bytes", nil},
		{"oversized-syntax-error-length-first", func(e *resourceapi.SharingAffinityExtractor) { e.Expression = strings.Repeat("[", 10241) }, "expression length 10241 exceeds 10240 bytes", nil},
		{"empty-name", func(e *resourceapi.SharingAffinityExtractor) { e.Name = "" }, "name", nil},
		{"invalid-name", func(e *resourceapi.SharingAffinityExtractor) { e.Name = "NOT_VALID" }, "name", nil},
		{"long-name", func(e *resourceapi.SharingAffinityExtractor) { e.Name = strings.Repeat("a", 64) }, "name", nil},
		{"empty-apigroup", func(e *resourceapi.SharingAffinityExtractor) { e.APIGroup = "" }, "apiGroup", nil},
		{"invalid-apigroup", func(e *resourceapi.SharingAffinityExtractor) { e.APIGroup = "BAD_GROUP" }, "apiGroup", nil},
		{"long-apigroup", func(e *resourceapi.SharingAffinityExtractor) { e.APIGroup = strings.Repeat("a", 254) }, "apiGroup", nil},
		{"empty-kind", func(e *resourceapi.SharingAffinityExtractor) { e.Kind = "" }, "kind", nil},
		{"invalid-kind", func(e *resourceapi.SharingAffinityExtractor) { e.Kind = "Bad_Kind" }, "kind", nil},
		{"numeric-kind", func(e *resourceapi.SharingAffinityExtractor) { e.Kind = "1NIC" }, "kind", nil},
		{"mixed-case-kind-at-label-limit", func(e *resourceapi.SharingAffinityExtractor) { e.Kind = "N" + strings.Repeat("ic", 31) }, "", nil},
		{"mixed-case-kind-over-label-limit", func(e *resourceapi.SharingAffinityExtractor) { e.Kind = "Ni" + strings.Repeat("ic", 31) }, "invalid kind: value exceeds 63 characters", nil},
		{"valid-group-kind", func(e *resourceapi.SharingAffinityExtractor) { e.APIGroup, e.Kind = "network.example.com", "NICConfig" }, "", nil},
		{"empty-expression", func(e *resourceapi.SharingAffinityExtractor) { e.Expression = "" }, "expression", nil},
		{"syntax-error", func(e *resourceapi.SharingAffinityExtractor) { e.Expression = "parameters[" }, "CEL expression could not be compiled", sharingaffinity.ErrInvalidDefinition},
		{"type-error-non-string", func(e *resourceapi.SharingAffinityExtractor) { e.Expression = "42" }, "CEL expression could not be compiled", sharingaffinity.ErrInvalidDefinition},
		{"single-expression-over-cost", func(e *resourceapi.SharingAffinityExtractor) {
			e.Expression = `parameters.s.contains(parameters.pattern) ? "x" : ""`
		}, "combined estimated cost exceeds 1000000", sharingaffinity.ErrCostExceeded},
	} {
		e := entry
		tc.edit(&e)
		cases = append(cases, sharingAffinityValidationCase{name: tc.name, pool: sharingAffinityPool(e), wantText: tc.text, wantKind: tc.kind})
	}
	nine := Slice{}
	for i := 0; i < 9; i++ {
		nine.SharingAffinity = append(nine.SharingAffinity, sharingAffinityEntry(fmt.Sprintf("key-%d", i), `"blue"`))
	}
	cases = append(cases, sharingAffinityValidationCase{name: "too-many-entries-per-slice", pool: Pool{Slices: []Slice{nine}}, wantText: "8"})
	for _, tc := range []struct {
		name  string
		slice Slice
	}{
		{"mixed-devices-affinity", Slice{Devices: []resourceapi.Device{{Name: "device"}}}},
		{"mixed-empty-devices-affinity", Slice{Devices: []resourceapi.Device{}}},
		{"mixed-counters-affinity", Slice{SharedCounters: []resourceapi.CounterSet{{Name: "counter"}}}},
		{"mixed-empty-counters-affinity", Slice{SharedCounters: []resourceapi.CounterSet{}}},
		{"mixed-all-three", Slice{Devices: []resourceapi.Device{}, SharedCounters: []resourceapi.CounterSet{}}},
	} {
		tc.slice.SharingAffinity = []resourceapi.SharingAffinityExtractor{entry}
		cases = append(cases, sharingAffinityValidationCase{name: tc.name, pool: Pool{Slices: []Slice{tc.slice}},
			wantText: "sharingAffinity cannot be combined with devices or sharedCounters in one slice"})
		tc.slice.SharingAffinity = []resourceapi.SharingAffinityExtractor{}
		cases = append(cases, sharingAffinityValidationCase{name: tc.name + "-empty-affinity", pool: Pool{Slices: []Slice{tc.slice}},
			wantText: "sharingAffinity cannot be combined with devices or sharedCounters in one slice"})
	}
	return cases
}

// The same per-object fixtures exercise pure validation and actual controller
// matching. A declaration on some other object must never grant a ratchet.
func sharingAffinityRatchetCases(t *testing.T) []sharingAffinityValidationCase {
	t.Helper()
	entry := sharingAffinityEntry("subnet", `"blue"`)
	long := entry
	long.Expression = `"` + strings.Repeat("a", 10239) + `"`
	badName := entry
	badName.Name = "INVALID"
	badGroup := entry
	badGroup.APIGroup = "INVALID"
	badKind := entry
	badKind.Kind = "INVALID_KIND"
	badCEL := entry
	badCEL.Expression = "parameters["
	storedOnly := entry
	storedOnly.Expression = `cidr("192.168.0.0/24").isMask() ? "x" : ""`
	if result := sharingaffinity.ValidateExpression(storedOnly.Expression, environment.StoredExpressions); result.Error != nil || result.MaxCost == 0 {
		t.Fatalf("stored-only fixture must compile with positive cost: %+v", result)
	}
	if result := sharingaffinity.ValidateExpression(storedOnly.Expression, environment.NewExpressions); result.Error == nil {
		t.Fatal("stored-only fixture is now available for new expressions; choose a rollback-only expression")
	}
	var cases []sharingAffinityValidationCase
	for _, tc := range []struct {
		name    string
		entries []resourceapi.SharingAffinityExtractor
	}{
		{"long-expression", []resourceapi.SharingAffinityExtractor{long}},
		{"invalid-name", []resourceapi.SharingAffinityExtractor{badName}},
		{"invalid-group", []resourceapi.SharingAffinityExtractor{badGroup}},
		{"invalid-kind", []resourceapi.SharingAffinityExtractor{badKind}},
		{"uncompilable", []resourceapi.SharingAffinityExtractor{badCEL}},
		{"stored-only-expression", []resourceapi.SharingAffinityExtractor{storedOnly}},
		{"nine-entries", func() []resourceapi.SharingAffinityExtractor {
			var entries []resourceapi.SharingAffinityExtractor
			for i := 0; i < 9; i++ {
				entries = append(entries, sharingAffinityEntry(fmt.Sprintf("key-%d", i), `"blue"`))
			}
			return entries
		}()},
	} {
		pool := Pool{AllNodes: true, Generation: 2, Slices: []Slice{{SharingAffinity: tc.entries}}}
		old := sharingAffinityStoredSlices(pool, 1)
		cases = append(cases, sharingAffinityValidationCase{name: "stored-legacy-unchanged-" + tc.name, pool: pool, old: old})
		changed := *pool.DeepCopy()
		changed.Slices = append(changed.Slices, Slice{SharingAffinity: []resourceapi.SharingAffinityExtractor{sharingAffinityEntry("zone", `"west"`)}})
		cases = append(cases, sharingAffinityValidationCase{name: "stored-legacy-other-slice-changed-" + tc.name, pool: changed, old: old})
		if len(tc.entries) == 9 {
			reordered := *pool.DeepCopy()
			reordered.Slices[0].SharingAffinity[0], reordered.Slices[0].SharingAffinity[1] = reordered.Slices[0].SharingAffinity[1], reordered.Slices[0].SharingAffinity[0]
			cases = append(cases, sharingAffinityValidationCase{name: "changed-nine-entry-list-rechecked", pool: reordered, old: old, wantText: "8"})
		}
	}
	pool := sharingAffinityPool(long, sharingAffinityEntry("zone", `"west"`))
	old := sharingAffinityStoredSlices(pool, 1)
	moved := *pool.DeepCopy()
	moved.Slices[0], moved.Slices[1] = moved.Slices[1], moved.Slices[0]
	cases = append(cases, sharingAffinityValidationCase{name: "redistribution-moves-legacy", pool: moved, old: old, wantText: "expression length 10241 exceeds 10240 bytes"})
	cases = append(cases, sharingAffinityValidationCase{name: "replacement-after-deletion", pool: pool, old: []*resourceapi.ResourceSlice{nil, old[1]}, wantText: "expression length 10241 exceeds 10240 bytes"})
	for _, field := range []string{"expression", "kind", "apiGroup", "name", "devices", "counters"} {
		changed := *pool.DeepCopy()
		text := ""
		switch field {
		case "expression":
			changed.Slices[0].SharingAffinity[0].Expression += " "
			text = "expression length 10242 exceeds 10240 bytes"
		case "kind":
			changed.Slices[0].SharingAffinity[0].Kind = "Bad_Kind"
			text = "kind"
		case "apiGroup":
			changed.Slices[0].SharingAffinity[0].APIGroup = "BAD_GROUP"
			text = "apiGroup"
		case "name":
			changed.Slices[0].SharingAffinity[0].Name = "renamed"
			text = "expression length 10241 exceeds 10240 bytes"
		case "devices":
			changed.Slices[0].Devices = []resourceapi.Device{}
			text = "sharingAffinity cannot be combined"
		case "counters":
			changed.Slices[0].SharedCounters = []resourceapi.CounterSet{}
			text = "sharingAffinity cannot be combined"
		}
		cases = append(cases, sharingAffinityValidationCase{name: "stored-" + field + "-changed", pool: changed, old: old, wantText: text})
	}
	validKind := *pool.DeepCopy()
	validKind.Slices[0].SharingAffinity[0].Kind = "NICConfig"
	cases = append(cases, sharingAffinityValidationCase{name: "stored-kind-only-long-expression-not-rechecked", pool: validKind, old: old})
	duplicate := sharingAffinityPool(entry, entry)
	cases = append(cases, sharingAffinityValidationCase{name: "duplicate-insertion-same-union", pool: duplicate,
		old: sharingAffinityStoredSlices(sharingAffinityPool(entry), 1), wantText: "duplicate sharing affinity key"})
	at, over := poolCostBoundaryExtractors(t)
	for _, bump := range []int64{0, 2} {
		overPool := sharingAffinityPool(over...)
		overPool.Generation = bump
		cases = append(cases, sharingAffinityValidationCase{name: fmt.Sprintf("unchanged-over-budget-generation-%d", bump), pool: overPool,
			old: sharingAffinityStoredSlices(overPool, 1), wantKind: sharingaffinity.ErrCostExceeded, wantText: "1000000"})
	}
	cases = append(cases, sharingAffinityValidationCase{name: "stored-plus-new-over-budget", pool: sharingAffinityPool(over...),
		old: sharingAffinityStoredSlices(sharingAffinityPool(at[:2]...), 1), wantKind: sharingaffinity.ErrCostExceeded, wantText: "1000000"})
	cases = append(cases, sharingAffinityValidationCase{name: "stored-plus-new-at-budget", pool: sharingAffinityPool(at...),
		old: sharingAffinityStoredSlices(sharingAffinityPool(at[:2]...), 1)})
	removed := *pool.DeepCopy()
	removed.Slices[0].SharingAffinity = nil
	cases = append(cases, sharingAffinityValidationCase{name: "remove-last-declaration-from-slice", pool: removed, old: old})
	return cases
}

// Keep this construction identical to sharingaffinity.estimatedCostBoundaryExtractors:
// literal budgets in fixtures silently go stale when CEL estimation changes.
func poolCostBoundaryExtractors(t *testing.T) (at, over []resourceapi.SharingAffinityExtractor) {
	t.Helper()
	cost := func(expression string) uint64 {
		t.Helper()
		result := sharingaffinity.ValidateExpression(expression, environment.NewExpressions)
		if result.Error != nil {
			t.Fatalf("boundary fixture compilation failed: %v", result.Error)
		}
		if len(expression) > resourceapi.SharingAffinityCELExpressionMaxLength {
			t.Fatalf("boundary fixture length %d exceeds expression byte limit", len(expression))
		}
		return result.MaxCost
	}
	regex := func(input, pattern int) string {
		return fmt.Sprintf("'%s'.matches('%s') ? 'x' : 'x'", strings.Repeat("a", input), strings.Repeat("a", pattern))
	}
	limit := uint64(resourceapi.SharingAffinityCELMaxCost)
	low, high := 1, 5000
	for low < high {
		mid := (low + high + 1) / 2
		if 2*cost(regex(4000, mid)) <= limit-10 {
			low = mid
		} else {
			high = mid - 1
		}
	}
	base := regex(4000, low)
	remainder := limit - 2*cost(base)
	low, high = 0, 10000
	for low < high {
		mid := (low + high) / 2
		if cost(regex(mid, 1)) < remainder {
			low = mid + 1
		} else {
			high = mid
		}
	}
	filler := regex(low, 1)
	if got := 2*cost(base) + cost(filler); got != limit {
		t.Fatalf("could not construct exact aggregate boundary: got %d, want %d", got, limit)
	}
	for i, expression := range []string{base, base, filler} {
		if got := cost(expression); got >= limit {
			t.Fatalf("individual fixture cost %d must be below %d", got, limit)
		}
		at = append(at, sharingAffinityEntry(fmt.Sprintf("key-%d", i), expression))
	}
	over = append([]resourceapi.SharingAffinityExtractor(nil), at...)
	over[2].Expression = "(" + filler + ") + ''"
	if got := 2*cost(base) + cost(over[2].Expression); got != limit+1 {
		t.Fatalf("could not construct one-unit-over boundary: got %d, want %d", got, limit+1)
	}
	return at, over
}
