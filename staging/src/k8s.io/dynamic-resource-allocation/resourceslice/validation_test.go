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
	apiservercel "k8s.io/apiserver/pkg/cel"
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
			err := validateDriverResources(driverName, resources(tc.device))
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

// Validation covers pool invariants without repeating per-object admission.
func TestSharingAffinityPoolValidation(t *testing.T) {
	cases := sharingAffinityValidationCases(t)
	for _, tc := range sharingAffinitySliceValidationCases() {
		tc.wantText = ""
		cases = append(cases, tc)
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for _, driver := range []string{"driver.example.com", ""} {
				t.Run("driver="+driver, func(t *testing.T) {
					for attempt := 0; attempt < 3; attempt++ {
						err := validatePool(driver, "pool", tc.pool)
						checkSharingAffinityValidationError(t, err, tc)
					}
				})
			}
		})
	}
	t.Run("distinct-pools-same-names", func(t *testing.T) {
		pool := poolWithSharingAffinities(sharingAffinityEntry("subnet", `"blue"`))
		err := validateDriverResources("driver.example.com", &DriverResources{
			Pools: map[string]Pool{"pool-a": pool, "pool-b": pool},
		})
		if err != nil {
			t.Fatalf("independent pools rejected: %v", err)
		}
	})
}

func TestSharingAffinitySliceValidation(t *testing.T) {
	for _, tc := range sharingAffinitySliceValidationCases() {
		t.Run(tc.name, func(t *testing.T) {
			for _, stored := range []bool{false, true} {
				t.Run(fmt.Sprintf("stored=%t", stored), func(t *testing.T) {
					var old *resourceapi.ResourceSlice
					want := tc.wantText
					if stored {
						old = MakeResourceSlice().SharingAffinity(tc.pool.Slices[0].SharingAffinity...).Obj().DeepCopy()
						want = tc.wantUnchangedText
					}
					err := validateSliceSharingAffinity(tc.pool.Slices[0], old)
					if want == "" {
						if err != nil {
							t.Fatalf("unchanged declaration rejected: %v", err)
						}
					} else if err == nil || !strings.Contains(err.Error(), want) {
						t.Fatalf("error = %v, want %q", err, want)
					}
				})
			}
		})
	}

	entry := sharingAffinityEntry("subnet", `"blue"`)
	for _, tc := range []struct {
		name string
		edit func(*resourceapi.SharingAffinityExtractor)
		want string
	}{
		{"wildcard-matchers", func(e *resourceapi.SharingAffinityExtractor) {}, ""},
		{"name-at-limit", func(e *resourceapi.SharingAffinityExtractor) { e.Name = strings.Repeat("a", 63) }, ""},
		{"group-at-limit", func(e *resourceapi.SharingAffinityExtractor) { e.APIGroup = strings.Repeat("a", 253) }, ""},
		{"kind-at-limit", func(e *resourceapi.SharingAffinityExtractor) { e.Kind = "N" + strings.Repeat("ic", 31) }, ""},
		{"valid-group-kind", func(e *resourceapi.SharingAffinityExtractor) { e.APIGroup, e.Kind = "network.example.com", "NICConfig" }, ""},
		{"expression-at-byte-limit", func(e *resourceapi.SharingAffinityExtractor) {
			e.Expression = `"` + strings.Repeat("\u00e9", (resourceapi.SharingAffinityCELExpressionMaxLength-2)/2) + `"`
		}, ""},
		{"expression-over-byte-limit", func(e *resourceapi.SharingAffinityExtractor) {
			e.Expression = `"` + strings.Repeat("\u00e9", resourceapi.SharingAffinityCELExpressionMaxLength/2) + `"`
		}, "expression length 10242 exceeds 10240 bytes"},
		{"empty-expression", func(e *resourceapi.SharingAffinityExtractor) { e.Expression = "" }, "expression is required"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			desired := entry
			tc.edit(&desired)
			err := validateSliceSharingAffinity(Slice{SharingAffinity: []resourceapi.SharingAffinityExtractor{desired}}, nil)
			if tc.want == "" {
				if err != nil {
					t.Fatalf("valid declaration rejected: %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}

	legacy := entry
	legacy.Name, legacy.APIGroup, legacy.Kind = "OLD_NAME", "OLD_GROUP", "Old_Kind"
	legacy.Expression = `"` + strings.Repeat("a", resourceapi.SharingAffinityCELExpressionMaxLength-1) + `"`
	other := sharingAffinityEntry("other", `"red"`)
	for _, tc := range []struct {
		name string
		edit func(*resourceapi.SharingAffinityExtractor)
		want string
	}{
		{"unchanged-fields-survive-list-reordering", func(e *resourceapi.SharingAffinityExtractor) {}, ""},
		{"repair-name", func(e *resourceapi.SharingAffinityExtractor) { e.Name = "subnet" }, "invalid apiGroup"},
		{"change-invalid-name", func(e *resourceapi.SharingAffinityExtractor) { e.Name = "NEW_NAME" }, "invalid name"},
		{"repair-group", func(e *resourceapi.SharingAffinityExtractor) { e.APIGroup = "*" }, ""},
		{"change-invalid-group", func(e *resourceapi.SharingAffinityExtractor) { e.APIGroup = "NEW_GROUP" }, "invalid apiGroup"},
		{"repair-kind", func(e *resourceapi.SharingAffinityExtractor) { e.Kind = "*" }, ""},
		{"change-invalid-kind", func(e *resourceapi.SharingAffinityExtractor) { e.Kind = "New_Kind" }, "invalid kind"},
		{"repair-expression", func(e *resourceapi.SharingAffinityExtractor) { e.Expression = `"red"` }, ""},
		{"change-oversized-expression", func(e *resourceapi.SharingAffinityExtractor) {
			e.Expression = `"` + strings.Repeat("b", resourceapi.SharingAffinityCELExpressionMaxLength-1) + `"`
		}, "expression length"},
		{"clear-expression", func(e *resourceapi.SharingAffinityExtractor) { e.Expression = "" }, "expression is required"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			old := &resourceapi.ResourceSlice{Spec: resourceapi.ResourceSliceSpec{
				SharingAffinity: []resourceapi.SharingAffinityExtractor{other, legacy},
			}}
			desired := legacy
			tc.edit(&desired)
			err := validateSliceSharingAffinity(Slice{SharingAffinity: []resourceapi.SharingAffinityExtractor{desired, other}}, old)
			if tc.want == "" {
				if err != nil {
					t.Fatalf("unchanged fields rejected: %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}

	t.Run("entry-count-ratcheting", func(t *testing.T) {
		var entries []resourceapi.SharingAffinityExtractor
		for i := 0; i <= resourceapi.SharingAffinityMaxEntries; i++ {
			entries = append(entries, sharingAffinityEntry(fmt.Sprintf("key-%d", i), `"blue"`))
		}
		old := &resourceapi.ResourceSlice{Spec: resourceapi.ResourceSliceSpec{SharingAffinity: entries}}
		changed := old.DeepCopy().Spec.SharingAffinity
		changed[0].Expression = `"red"`
		if err := validateSliceSharingAffinity(Slice{SharingAffinity: changed}, old); err == nil || !strings.Contains(err.Error(), "entries 9 exceeds 8") {
			t.Fatalf("changed oversized list error = %v", err)
		}
		if err := validateSliceSharingAffinity(Slice{SharingAffinity: changed[:resourceapi.SharingAffinityMaxEntries]}, old); err != nil {
			t.Fatalf("list reduced to the limit rejected: %v", err)
		}
	})
}

func TestValidateSharingAffinityAPIGroupMatcher(t *testing.T) {
	for _, tc := range []struct {
		name, value, want string
	}{
		{"empty", "", "value is required"},
		{"wildcard", "*", ""},
		{"subdomain", "network.example.com", ""},
		{"numeric-prefix", "3rd-party.example.com", ""},
		{"at-character-limit", strings.Repeat("a", 253), ""},
		{"over-character-limit", strings.Repeat("a", 254), "value exceeds 253 characters"},
		{"uppercase", "Network.Example.com", "must consist of"},
		{"partial-wildcard", "*.example.com", "must consist of"},
		{"wildcard-suffix", "network*", "must consist of"},
		{"underscore", "network_group", "must consist of"},
		{"unicode-at-character-limit", strings.Repeat("\u00e9", 253), "must consist of"},
		{"unicode-over-character-limit", strings.Repeat("\u00e9", 254), "value exceeds 253 characters"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := validateSharingAffinityAPIGroupMatcher(tc.value)
			if tc.want == "" {
				if err != nil {
					t.Fatalf("valid API group rejected: %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestValidateSharingAffinityKindMatcher(t *testing.T) {
	for _, tc := range []struct {
		name, value, want string
	}{
		{"empty", "", "value is required"},
		{"wildcard", "*", ""},
		{"mixed-case", "NICConfig", ""},
		{"label-with-hyphen", "NICConfig-2", ""},
		{"at-character-limit", strings.Repeat("N", 63), ""},
		{"over-character-limit", strings.Repeat("N", 64), "value exceeds 63 characters"},
		{"qualified-kind", "example.NIC", "must consist of"},
		{"numeric-prefix", "1NIC", "must consist of"},
		{"partial-wildcard", "NIC*", "must consist of"},
		{"underscore", "NIC_Config", "must consist of"},
		{"unicode-at-character-limit", strings.Repeat("\u00e9", 63), "must consist of"},
		{"unicode-over-character-limit", strings.Repeat("\u00e9", 64), "value exceeds 63 characters"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := validateSharingAffinityKindMatcher(tc.value)
			if tc.want == "" {
				if err != nil {
					t.Fatalf("valid kind rejected: %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}

type sharingAffinityValidationCase struct {
	name              string
	pool              Pool
	wantText          string
	wantUnchangedText string
	wantExact         string
	wantKind          error
	wantCompile       bool
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
	var compileErr *apiservercel.Error
	if got := errors.As(err, &compileErr); got != tc.wantCompile {
		t.Errorf("compiler cause reachable = %t, want %t: %v", got, tc.wantCompile, err)
	}
	if tc.wantKind != nil {
		var union []resourceapi.SharingAffinityExtractor
		for _, slice := range tc.pool.Slices {
			union = append(union, slice.SharingAffinity...)
		}
		want := sharingaffinity.ValidateExpressions(union)
		inner := errors.Unwrap(err)
		if want == nil || inner == nil || inner.Error() != want.Error() || !errors.Is(inner, sharingaffinity.ErrInvalidDefinition) {
			t.Errorf("wrapped error = %v, want pool validation error %v", inner, want)
		}
		if tc.wantCompile {
			result := sharingaffinity.ValidateExpression(union[len(union)-1].Expression, environment.StoredExpressions)
			if result.Error == nil {
				t.Error("pool compiler unexpectedly accepted invalid fixture")
			}
		}
	}
}

func sharingAffinityEntry(name, expression string) resourceapi.SharingAffinityExtractor {
	return resourceapi.SharingAffinityExtractor{Name: name, APIGroup: "*", Kind: "*", Expression: expression}
}

func poolWithSharingAffinities(entries ...resourceapi.SharingAffinityExtractor) Pool {
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
		{name: "valid-multi-slice", pool: poolWithSharingAffinities(entry, sharingAffinityEntry("zone", `"west"`))},
		{name: "nil-affinity", pool: Pool{Slices: []Slice{{}}}},
		{name: "empty-affinity", pool: Pool{Slices: []Slice{{SharingAffinity: []resourceapi.SharingAffinityExtractor{}}}}},
		{name: "duplicate-within-slice", pool: Pool{Slices: []Slice{{SharingAffinity: []resourceapi.SharingAffinityExtractor{entry, entry}}}},
			wantExact: `pool "pool": duplicate sharing affinity key "subnet"`},
		{name: "duplicate-across-slices", pool: poolWithSharingAffinities(entry, entry),
			wantExact: `pool "pool": duplicate sharing affinity key "subnet"`},
		{name: "cost-at-limit-accepted", pool: poolWithSharingAffinities(at...)},
		{name: "cost-over-limit-across-individually-valid-slices", pool: poolWithSharingAffinities(over...), wantText: "combined estimated cost exceeds 1000000", wantKind: sharingaffinity.ErrCostExceeded},
		{name: "devices-counters-no-affinity-unchanged", pool: Pool{Slices: []Slice{{Devices: []resourceapi.Device{{Name: "device"}}, SharedCounters: []resourceapi.CounterSet{{Name: "counter"}}}}}},
	}
	for _, entries := range [][]resourceapi.SharingAffinityExtractor{at, over} {
		for _, entry := range entries {
			if err := sharingaffinity.ValidateExpressions([]resourceapi.SharingAffinityExtractor{entry}); err != nil {
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
		{"oversized-syntax-error", func(e *resourceapi.SharingAffinityExtractor) { e.Expression = strings.Repeat("[", 10241) }, "CEL expression could not be compiled", sharingaffinity.ErrInvalidDefinition},
		{"mixed-case-kind-at-label-limit", func(e *resourceapi.SharingAffinityExtractor) { e.Kind = "N" + strings.Repeat("ic", 31) }, "", nil},
		{"valid-group-kind", func(e *resourceapi.SharingAffinityExtractor) { e.APIGroup, e.Kind = "network.example.com", "NICConfig" }, "", nil},
		{"stored-environment-function", func(e *resourceapi.SharingAffinityExtractor) {
			e.Expression = `cidr("192.168.0.0/24").isMask() ? "x" : ""`
		}, "", nil},
		{"empty-expression", func(e *resourceapi.SharingAffinityExtractor) { e.Expression = "" }, "CEL expression could not be compiled", sharingaffinity.ErrInvalidDefinition},
		{"syntax-error", func(e *resourceapi.SharingAffinityExtractor) { e.Expression = "parameters[" }, "CEL expression could not be compiled", sharingaffinity.ErrInvalidDefinition},
		{"type-error-non-string", func(e *resourceapi.SharingAffinityExtractor) { e.Expression = "42" }, "CEL expression could not be compiled", sharingaffinity.ErrInvalidDefinition},
		{"single-expression-over-cost", func(e *resourceapi.SharingAffinityExtractor) {
			e.Expression = `parameters.s.contains(parameters.pattern) ? "x" : ""`
		}, "combined estimated cost exceeds 1000000", sharingaffinity.ErrCostExceeded},
	} {
		e := entry
		tc.edit(&e)
		cases = append(cases, sharingAffinityValidationCase{
			name: tc.name, pool: poolWithSharingAffinities(e), wantText: tc.text, wantKind: tc.kind,
			wantCompile: tc.kind == sharingaffinity.ErrInvalidDefinition,
		})
	}
	var nine []resourceapi.SharingAffinityExtractor
	for i := 0; i <= resourceapi.SharingAffinityMaxEntries; i++ {
		nine = append(nine, sharingAffinityEntry(fmt.Sprintf("key-%d", i), `"blue"`))
	}
	cases = append(cases, sharingAffinityValidationCase{name: "more-than-eight-entries-across-slices", pool: poolWithSharingAffinities(nine...)})
	return cases
}

// These declarations compile within the pool budget but violate per-slice rules.
func sharingAffinitySliceValidationCases() []sharingAffinityValidationCase {
	entry := sharingAffinityEntry("subnet", `"blue"`)
	var cases []sharingAffinityValidationCase
	for _, tc := range []struct {
		name string
		edit func(*resourceapi.SharingAffinityExtractor)
		want string
	}{
		{"expression-length-over-limit", func(e *resourceapi.SharingAffinityExtractor) {
			e.Expression = `"` + strings.Repeat("a", resourceapi.SharingAffinityCELExpressionMaxLength-1) + `"`
		}, "expression length 10241 exceeds 10240 bytes"},
		{"empty-name", func(e *resourceapi.SharingAffinityExtractor) { e.Name = "" }, "invalid name"},
		{"invalid-name", func(e *resourceapi.SharingAffinityExtractor) { e.Name = "NOT_VALID" }, "invalid name"},
		{"long-name", func(e *resourceapi.SharingAffinityExtractor) { e.Name = strings.Repeat("a", 64) }, "invalid name"},
		{"empty-apigroup", func(e *resourceapi.SharingAffinityExtractor) { e.APIGroup = "" }, "invalid apiGroup"},
		{"invalid-apigroup", func(e *resourceapi.SharingAffinityExtractor) { e.APIGroup = "BAD_GROUP" }, "invalid apiGroup"},
		{"uppercase-apigroup", func(e *resourceapi.SharingAffinityExtractor) { e.APIGroup = "Network.Example.com" }, "invalid apiGroup"},
		{"long-apigroup", func(e *resourceapi.SharingAffinityExtractor) { e.APIGroup = strings.Repeat("a", 254) }, "invalid apiGroup"},
		{"empty-kind", func(e *resourceapi.SharingAffinityExtractor) { e.Kind = "" }, "invalid kind"},
		{"invalid-kind", func(e *resourceapi.SharingAffinityExtractor) { e.Kind = "Bad_Kind" }, "invalid kind"},
		{"qualified-kind", func(e *resourceapi.SharingAffinityExtractor) { e.Kind = "example.NIC" }, "invalid kind"},
		{"numeric-kind", func(e *resourceapi.SharingAffinityExtractor) { e.Kind = "1NIC" }, "invalid kind"},
		{"long-kind", func(e *resourceapi.SharingAffinityExtractor) { e.Kind = "Ni" + strings.Repeat("ic", 31) }, "invalid kind"},
	} {
		e := entry
		tc.edit(&e)
		cases = append(cases, sharingAffinityValidationCase{name: tc.name, pool: poolWithSharingAffinities(e), wantText: tc.want})
	}
	nine := Slice{}
	for i := 0; i <= resourceapi.SharingAffinityMaxEntries; i++ {
		nine.SharingAffinity = append(nine.SharingAffinity, sharingAffinityEntry(fmt.Sprintf("key-%d", i), `"blue"`))
	}
	cases = append(cases, sharingAffinityValidationCase{name: "too-many-entries", pool: Pool{AllNodes: true, Slices: []Slice{nine}}, wantText: "sharing affinity entries 9 exceeds 8"})
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
		const want = "sharingAffinity cannot be combined with devices or sharedCounters"
		tc.slice.SharingAffinity = []resourceapi.SharingAffinityExtractor{entry}
		cases = append(cases, sharingAffinityValidationCase{name: tc.name, pool: Pool{AllNodes: true, Slices: []Slice{tc.slice}}, wantText: want, wantUnchangedText: want})
		tc.slice.SharingAffinity = []resourceapi.SharingAffinityExtractor{}
		cases = append(cases, sharingAffinityValidationCase{name: tc.name + "-empty-affinity", pool: Pool{AllNodes: true, Slices: []Slice{tc.slice}}, wantText: want, wantUnchangedText: want})
	}
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
