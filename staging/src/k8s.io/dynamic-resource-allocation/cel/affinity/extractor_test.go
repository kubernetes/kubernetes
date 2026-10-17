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

package affinity

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/runtime"
	dracel "k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

const testDriver = "dra.example.com"

func TestCompileObjectOnlyEnvironment(t *testing.T) {
	// the extractor CEL environment binds only object, never device.
	if _, err := compileExpression(`object.kind`, compileOptions{}); err != nil {
		t.Fatalf("compileExpression(object.kind) error = %v, want nil", err)
	}
	if _, err := compileExpression(`device.attributes["dra.example.com"].model`, compileOptions{}); err == nil || !strings.Contains(err.Error(), "device") {
		t.Fatalf("compileExpression(device...) error = %v, want an error mentioning device", err)
	}
}

func TestCompileSucceedsAcrossGenerations(t *testing.T) {
	// programs compile successfully on repeated Compiles and after a
	// generation bump. Caching is keyed by expression text (see
	// TestCompileCachesProgramsByExpression), not by generation.
	ctx := testContext(t)
	extractor := newTestExtractor()
	pool := testPool(7, []resourceapi.SharingAffinityExtractor{
		{CEL: map[string]string{"subnet": `object.spec.subnet`}},
	})
	if err := extractor.Compile(ctx, pool); err != nil {
		t.Fatalf("first Compile() error = %v, want nil", err)
	}
	if err := extractor.Compile(ctx, pool); err != nil {
		t.Fatalf("second Compile() error = %v, want nil", err)
	}
	pool.Key.Generation++
	if err := extractor.Compile(ctx, pool); err != nil {
		t.Fatalf("Compile() after generation bump error = %v, want nil", err)
	}
}

func TestExtractSelectorFiltering(t *testing.T) {
	// selector matching gates which extractor groups apply.
	ctx := testContext(t)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{
		{
			Selector: selector(`device.attributes["dra.example.com"].model == "match"`),
			CEL:      map[string]string{"subnet": `object.spec.subnet`},
		},
	})
	configs := []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"blue"}}`)}

	got, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, deviceWithModel("match"), configs)
	if err != nil {
		t.Fatalf("Extract() matching selector error = %v, want nil", err)
	}
	if want := map[string]string{"subnet": "blue"}; !reflect.DeepEqual(got, want) {
		t.Errorf("Extract() matching selector = %#v, want %#v", got, want)
	}

	got, err = newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, deviceWithModel("skip"), configs)
	if err != nil {
		t.Fatalf("Extract() non-matching selector error = %v, want nil", err)
	}
	if len(got) != 0 {
		t.Errorf("Extract() non-matching selector = %#v, want empty map", got)
	}
}

func TestExtractStrictGating(t *testing.T) {
	// when one key in a group is produced, all keys in that group are required.
	ctx := testContext(t)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{
		{CEL: map[string]string{
			"subnet": `object.spec.subnet`,
			"pkey":   `""`,
		}},
	})

	_, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, testDevice(), []resourceapi.DeviceClaimConfiguration{
		opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"blue"}}`),
	})
	if !errors.Is(err, ErrMissingField) {
		t.Fatalf("Extract() error = %v, want ErrMissingField", err)
	}
}

func TestExtractEmptyReturn(t *testing.T) {
	// empty string return means no affinity contribution from that object.
	ctx := testContext(t)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{
		{CEL: map[string]string{"subnet": `""`}},
	})

	got, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, testDevice(), []resourceapi.DeviceClaimConfiguration{
		opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"blue"}}`),
	})
	if err != nil {
		t.Fatalf("Extract() error = %v, want nil", err)
	}
	if len(got) != 0 {
		t.Errorf("Extract() = %#v, want empty map", got)
	}
}

func TestExtractMissingField(t *testing.T) {
	// missing object fields are classifiable claim/slice extraction failures.
	ctx := testContext(t)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{
		{CEL: map[string]string{"subnet": `object.spec.noSuchField`}},
	})

	_, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, testDevice(), []resourceapi.DeviceClaimConfiguration{
		opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"blue"}}`),
	})
	if !errors.Is(err, ErrMissingField) {
		t.Fatalf("Extract() error = %v, want ErrMissingField", err)
	}
}

func TestExtractNonStringReturn(t *testing.T) {
	// non-string CEL results are reported separately from eval failures.
	ctx := testContext(t)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{
		{CEL: map[string]string{"subnet": `42`}},
	})

	_, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, testDevice(), []resourceapi.DeviceClaimConfiguration{
		opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"blue"}}`),
	})
	if !errors.Is(err, ErrNonStringReturn) {
		t.Fatalf("Extract() error = %v, want ErrNonStringReturn", err)
	}
}

func TestExtractCostExceeded(t *testing.T) {
	// extractor evaluation uses the DRA CEL runtime cost budget.
	ctx := testContext(t)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{
		{CEL: map[string]string{"subnet": `[1, 2, 3, 4, 5].exists(i, [1, 2, 3, 4, 5].exists(j, string(i) + string(j) == object.spec.subnet)) ? object.spec.subnet : ""`}},
	})

	_, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, testDevice(), []resourceapi.DeviceClaimConfiguration{
		opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"never"}}`),
	})
	if !errors.Is(err, ErrCostExceeded) {
		t.Fatalf("Extract() error = %v, want ErrCostExceeded", err)
	}
}

func TestExtractDuplicateKeySameValue(t *testing.T) {
	// two applicable extractor groups may publish the same key
	// when they agree on the value.
	ctx := testContext(t)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{
		{CEL: map[string]string{"subnet": `object.spec.subnet`}},
		{CEL: map[string]string{"subnet": `object.spec.subnet`}},
	})

	got, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, testDevice(), []resourceapi.DeviceClaimConfiguration{
		opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"blue"}}`),
	})
	if err != nil {
		t.Fatalf("Extract() error = %v, want nil", err)
	}
	if want := map[string]string{"subnet": "blue"}; !reflect.DeepEqual(got, want) {
		t.Errorf("Extract() = %#v, want %#v", got, want)
	}
}

func TestExtractClaimSelfInconsistent(t *testing.T) {
	// one claim/request cannot produce conflicting values for the same key.
	ctx := testContext(t)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{
		{CEL: map[string]string{"subnet": `object.spec.subnet`}},
	})

	_, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, testDevice(), []resourceapi.DeviceClaimConfiguration{
		opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"blue"}}`),
		opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"green"}}`),
	})
	if !errors.Is(err, ErrClaimSelfInconsistent) {
		t.Fatalf("Extract() error = %v, want ErrClaimSelfInconsistent", err)
	}
}

func TestExtractMultiVersionGuard(t *testing.T) {
	// drivers can guard versions by returning "" for unsupported objects.
	ctx := testContext(t)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{
		{CEL: map[string]string{"subnet": `object.apiVersion == "example.com/v1" && object.kind == "Config" ? object.spec.subnet : ""`}},
	})

	got, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, testDevice(), []resourceapi.DeviceClaimConfiguration{
		opaqueConfig(nil, `{"apiVersion":"example.com/v1beta1","kind":"Config","spec":{"subnet":"old"}}`),
		opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"blue"}}`),
	})
	if err != nil {
		t.Fatalf("Extract() error = %v, want nil", err)
	}
	if want := map[string]string{"subnet": "blue"}; !reflect.DeepEqual(got, want) {
		t.Errorf("Extract() = %#v, want %#v", got, want)
	}
}

func TestExtractInScopeRequestConfigs(t *testing.T) {
	// request scoping honors global, top-level, exact subrequest, and unrelated configs.
	ctx := testContext(t)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{
		{CEL: map[string]string{
			"subnet": `object.spec.subnet`,
			"pkey":   `object.spec.pkey`,
		}},
	})

	got, err := newTestExtractor().Extract(ctx, pool, Request{Name: "slice", ParentName: "main"}, testDevice(), []resourceapi.DeviceClaimConfiguration{
		opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"blue","pkey":"p0"}}`),
		opaqueConfig([]string{"main"}, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"blue","pkey":"p0"}}`),
		opaqueConfig([]string{"main/slice"}, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"blue","pkey":"p0"}}`),
		opaqueConfig([]string{"other"}, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"green","pkey":"p9"}}`),
	})
	if err != nil {
		t.Fatalf("Extract() error = %v, want nil", err)
	}
	if want := map[string]string{"subnet": "blue", "pkey": "p0"}; !reflect.DeepEqual(got, want) {
		t.Errorf("Extract() = %#v, want %#v", got, want)
	}
}

func TestErrorsAreClassifiable(t *testing.T) {
	// exported typed errors must work with errors.Is.
	for _, kind := range []error{
		ErrClaimSelfInconsistent,
		ErrSliceKeyCollision,
		ErrCELEval,
		ErrNonStringReturn,
		ErrCostExceeded,
		ErrMissingField,
	} {
		t.Run(kind.Error(), func(t *testing.T) {
			err := &Error{Kind: kind}
			if !errors.Is(err, kind) {
				t.Errorf("errors.Is(%T{Kind: %q}, %q) = false, want true", err, kind, kind)
			}
		})
	}
}

func TestCompileCachesProgramsByExpression(t *testing.T) {
	// CEL programs are cached by expression text and reused whenever the same
	// expression recurs — within a generation, on a repeated Compile, and across
	// a generation bump — because compilation depends only on the expression and
	// the fixed affinity CEL environment, not on the pool generation.
	ctx := testContext(t)
	extractor := newTestExtractor()
	pool := testPool(3, []resourceapi.SharingAffinityExtractor{
		{CEL: map[string]string{"subnet": `object.spec.subnet`}},
	})
	if err := extractor.Compile(ctx, pool); err != nil {
		t.Fatalf("Compile() error = %v, want nil", err)
	}
	first := extractor.get(`object.spec.subnet`)
	if first == nil {
		t.Fatalf("get(expression) = nil after Compile, want cached program")
	}
	if err := extractor.Compile(ctx, pool); err != nil {
		t.Fatalf("second Compile() error = %v, want nil", err)
	}
	if second := extractor.get(`object.spec.subnet`); second != first {
		t.Errorf("program recompiled within generation: got %p, want cached %p", second, first)
	}
	// The cache key is the expression, not the generation: bumping the pool
	// generation while the expression is unchanged reuses the same program.
	pool.Key.Generation++
	if err := extractor.Compile(ctx, pool); err != nil {
		t.Fatalf("Compile() after generation bump error = %v, want nil", err)
	}
	if reused := extractor.get(`object.spec.subnet`); reused != first {
		t.Errorf("generation bump recompiled unchanged expression: got %p, want cached %p", reused, first)
	}
}

func TestDisjointSelectorsShareKeyDifferentExpression(t *testing.T) {
	// KEP-5981 "Shared Key Across Disjoint Selectors (allowed)": two groups may
	// define the same key with DIFFERENT expressions when their selectors are
	// disjoint, because only the matching group applies to any single device.
	// The cache is keyed by expression text, so both expressions coexist and
	// Compile succeeds (the old pool-global compile-time guard is gone).
	ctx := testContext(t)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{
		{Selector: selector(`device.attributes["dra.example.com"].model == "a"`), CEL: map[string]string{"subnet": `object.spec.subnet`}},
		{Selector: selector(`device.attributes["dra.example.com"].model == "b"`), CEL: map[string]string{"subnet": `object.spec.other`}},
	})
	extractor := newTestExtractor()
	if err := extractor.Compile(ctx, pool); err != nil {
		t.Fatalf("Compile() error = %v, want nil (disjoint selectors may share a key)", err)
	}

	config := opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"blue","other":"green"}}`)

	// A device matching model "a" resolves subnet via the first group's expression.
	got, err := extractor.Extract(ctx, pool, Request{Name: "req"}, deviceWithModel("a"), []resourceapi.DeviceClaimConfiguration{config})
	if err != nil {
		t.Fatalf("Extract() model=a error = %v, want nil", err)
	}
	if want := map[string]string{"subnet": "blue"}; !reflect.DeepEqual(got, want) {
		t.Errorf("Extract() model=a = %#v, want %#v", got, want)
	}

	// A device matching model "b" resolves subnet via the second group's expression.
	got, err = extractor.Extract(ctx, pool, Request{Name: "req"}, deviceWithModel("b"), []resourceapi.DeviceClaimConfiguration{config})
	if err != nil {
		t.Fatalf("Extract() model=b error = %v, want nil", err)
	}
	if want := map[string]string{"subnet": "green"}; !reflect.DeepEqual(got, want) {
		t.Errorf("Extract() model=b = %#v, want %#v", got, want)
	}
}

func TestExtractSliceKeyCollision(t *testing.T) {
	// KEP-5981 authoritative runtime collision: when two applicable extractors
	// produce the same key with DIFFERENT values for one candidate device, the
	// device is not viable and Extract returns ErrSliceKeyCollision. With the
	// expression-keyed cache this per-device check is the primary collision guard.
	ctx := testContext(t)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{
		{CEL: map[string]string{"subnet": `object.spec.subnet`}},
		{CEL: map[string]string{"subnet": `object.spec.other`}},
	})
	_, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, testDevice(), []resourceapi.DeviceClaimConfiguration{
		opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"blue","other":"green"}}`),
	})
	if !errors.Is(err, ErrSliceKeyCollision) {
		t.Fatalf("Extract() error = %v, want ErrSliceKeyCollision", err)
	}
}

func TestExtractInvalidOpaqueJSON(t *testing.T) {
	// Edge case: malformed opaque parameters are a classifiable ErrCELEval failure.
	ctx := testContext(t)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{
		{CEL: map[string]string{"subnet": `object.spec.subnet`}},
	})
	_, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, testDevice(), []resourceapi.DeviceClaimConfiguration{
		opaqueConfig(nil, `{"spec": not-json}`),
	})
	if !errors.Is(err, ErrCELEval) {
		t.Fatalf("Extract() error = %v, want ErrCELEval", err)
	}
}

func TestExtractSelectorCompileError(t *testing.T) {
	// Edge case: a malformed slice-authored selector surfaces as ErrCELEval.
	ctx := testContext(t)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{
		{Selector: selector(`this is not valid cel`), CEL: map[string]string{"subnet": `object.spec.subnet`}},
	})
	_, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, testDevice(), []resourceapi.DeviceClaimConfiguration{
		opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"blue"}}`),
	})
	if !errors.Is(err, ErrCELEval) {
		t.Fatalf("Extract() error = %v, want ErrCELEval", err)
	}
}

func TestExtractNoExtractors(t *testing.T) {
	// Edge case: a pool with no extractors yields an empty map and nil error.
	ctx := testContext(t)
	pool := testPool(1, nil)
	got, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, testDevice(), []resourceapi.DeviceClaimConfiguration{
		opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"blue"}}`),
	})
	if err != nil {
		t.Fatalf("Extract() error = %v, want nil", err)
	}
	if len(got) != 0 {
		t.Errorf("Extract() = %#v, want empty map", got)
	}
}

func TestExtractNonOpaqueConfigIgnored(t *testing.T) {
	// Edge case: this phase only reads opaque configs; non-opaque ones are ignored.
	ctx := testContext(t)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{
		{CEL: map[string]string{"subnet": `object.spec.subnet`}},
	})
	got, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, testDevice(), []resourceapi.DeviceClaimConfiguration{
		{DeviceConfiguration: resourceapi.DeviceConfiguration{}},
	})
	if err != nil {
		t.Fatalf("Extract() error = %v, want nil", err)
	}
	if len(got) != 0 {
		t.Errorf("Extract() = %#v, want empty map", got)
	}
}

func TestExtractOpaqueDriverMismatchIgnored(t *testing.T) {
	// Edge case: opaque configs addressed to another driver are skipped.
	ctx := testContext(t)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{
		{CEL: map[string]string{"subnet": `object.spec.subnet`}},
	})
	cfg := opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"blue"}}`)
	cfg.Opaque.Driver = "other.example.com"
	got, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, testDevice(), []resourceapi.DeviceClaimConfiguration{cfg})
	if err != nil {
		t.Fatalf("Extract() error = %v, want nil", err)
	}
	if len(got) != 0 {
		t.Errorf("Extract() = %#v, want empty map", got)
	}
}

func TestExtractMultipleObjectsSameKeySameValue(t *testing.T) {
	// Edge case: repeated identical values for one key across a claim's own objects
	// are accepted (not ErrClaimSelfInconsistent).
	ctx := testContext(t)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{
		{CEL: map[string]string{"subnet": `object.spec.subnet`}},
	})
	got, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, testDevice(), []resourceapi.DeviceClaimConfiguration{
		opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"blue"}}`),
		opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","spec":{"subnet":"blue"}}`),
	})
	if err != nil {
		t.Fatalf("Extract() error = %v, want nil", err)
	}
	if want := map[string]string{"subnet": "blue"}; !reflect.DeepEqual(got, want) {
		t.Errorf("Extract() = %#v, want %#v", got, want)
	}
}

func testContext(t *testing.T) context.Context {
	t.Helper()
	_, ctx := ktesting.NewTestContext(t)
	return ctx
}

func newTestExtractor() *Extractor {
	return NewExtractor(32, dracel.NewCache(32, dracel.Features{}))
}

func testPool(generation int64, extractors []resourceapi.SharingAffinityExtractor) PoolExtractors {
	return PoolExtractors{
		Key: PoolKey{
			Driver:     testDriver,
			Name:       "pool-a",
			Generation: generation,
		},
		Extractors: extractors,
	}
}

func testDevice() CandidateDevice {
	return deviceWithModel("match")
}

func deviceWithModel(model string) CandidateDevice {
	return CandidateDevice{
		ID:     "device-a",
		Driver: testDriver,
		Device: resourceapi.Device{
			Name: "device-a",
			Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				"model": {StringValue: ptr.To(model)},
			},
			AllowMultipleAllocations: ptr.To(true),
		},
	}
}

func selector(expression string) *resourceapi.DeviceSelector {
	return &resourceapi.DeviceSelector{
		CEL: &resourceapi.CELDeviceSelector{Expression: expression},
	}
}

func opaqueConfig(requests []string, raw string) resourceapi.DeviceClaimConfiguration {
	return resourceapi.DeviceClaimConfiguration{
		Requests: requests,
		DeviceConfiguration: resourceapi.DeviceConfiguration{
			Opaque: &resourceapi.OpaqueDeviceConfiguration{
				Driver: testDriver,
				Parameters: runtime.RawExtension{
					Raw: []byte(raw),
				},
			},
		},
	}
}

const integrationDriver = "gpu.example.com"

// poolFromSlice mirrors how scheduler wiring will derive a PoolExtractors from a
// real ResourceSlice object: pool identity comes from spec.driver and spec.pool,
// and the extractors come straight off spec.sharingAffinity.
func poolFromSlice(slice *resourceapi.ResourceSlice) PoolExtractors {
	return PoolExtractors{
		Key: PoolKey{
			Driver:     slice.Spec.Driver,
			Name:       slice.Spec.Pool.Name,
			Generation: slice.Spec.Pool.Generation,
		},
		Extractors: slice.Spec.SharingAffinity,
	}
}

// candidatesFromSlice turns each real device in a slice into the CandidateDevice
// view the extractor consumes.
func candidatesFromSlice(slice *resourceapi.ResourceSlice) []CandidateDevice {
	out := make([]CandidateDevice, 0, len(slice.Spec.Devices))
	for _, device := range slice.Spec.Devices {
		out = append(out, CandidateDevice{
			ID:     slice.Spec.Pool.Name + "/" + device.Name,
			Driver: slice.Spec.Driver,
			Device: device,
		})
	}
	return out
}

func opaque(raw string) resourceapi.DeviceClaimConfiguration {
	return resourceapi.DeviceClaimConfiguration{
		DeviceConfiguration: resourceapi.DeviceConfiguration{
			Opaque: &resourceapi.OpaqueDeviceConfiguration{
				Driver:     integrationDriver,
				Parameters: runtime.RawExtension{Raw: []byte(raw)},
			},
		},
	}
}

func device(name, model string) resourceapi.Device {
	return resourceapi.Device{
		Name: name,
		Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			"model": {StringValue: ptr.To(model)},
		},
		AllowMultipleAllocations: ptr.To(true),
	}
}

// TestExtractAcrossRealSliceAndSharedSelectorCache verifies the full data flow
// from a real ResourceSlice through the public extractor API, using a single
// shared dracel.Cache for selector CEL across every device. Different devices
// match different selector-scoped groups, so this exercises the
// affinity-extractor <-> selector-evaluator boundary as the scheduler would.
func TestExtractAcrossRealSliceAndSharedSelectorCache(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	// Shared selector cache, reused across all devices and Extract calls: the
	// affinity extractor evaluates slice-authored selectors through the real DRA
	// selector CEL compiler/evaluator.
	selectorCache := dracel.NewCache(32, dracel.Features{})
	extractor := NewExtractor(32, selectorCache)

	slice := &resourceapi.ResourceSlice{
		Spec: resourceapi.ResourceSliceSpec{
			Driver: integrationDriver,
			Pool:   resourceapi.ResourcePool{Name: "rack-0", Generation: 5},
			Devices: []resourceapi.Device{
				device("gpu-0", "a100"),
				device("gpu-1", "h100"),
			},
			SharingAffinity: []resourceapi.SharingAffinityExtractor{
				{
					// Only applies to a100 devices.
					Selector: &resourceapi.DeviceSelector{CEL: &resourceapi.CELDeviceSelector{
						Expression: `device.attributes["gpu.example.com"].model == "a100"`,
					}},
					CEL: map[string]string{"nvlink-domain": `object.spec.nvlinkDomain`},
				},
				{
					// Only applies to h100 devices, publishing a different key.
					Selector: &resourceapi.DeviceSelector{CEL: &resourceapi.CELDeviceSelector{
						Expression: `device.attributes["gpu.example.com"].model == "h100"`,
					}},
					CEL: map[string]string{"nvswitch-domain": `object.spec.nvswitchDomain`},
				},
			},
		},
	}

	pool := poolFromSlice(slice)

	// Pre-compiling the pool must succeed against the real CEL runtime.
	if err := extractor.Compile(ctx, pool); err != nil {
		t.Fatalf("Compile() error = %v, want nil", err)
	}

	configs := []resourceapi.DeviceClaimConfiguration{
		opaque(`{"apiVersion":"gpu.example.com/v1","kind":"Config","spec":{"nvlinkDomain":"dom-a","nvswitchDomain":"sw-7"}}`),
	}

	req := Request{Name: "gpu"}
	candidates := candidatesFromSlice(slice)

	got := map[string]map[string]string{}
	// Drive every device through the shared extractor/selector cache, just like
	// the scheduler iterating a slice's devices for one request.
	for _, candidate := range candidates {
		result, err := extractor.Extract(ctx, pool, req, candidate, configs)
		if err != nil {
			t.Fatalf("Extract(%s) error = %v, want nil", candidate.ID, err)
		}
		got[candidate.Device.Name] = result
	}

	want := map[string]map[string]string{
		// a100 matches only the first group's selector.
		"gpu-0": {"nvlink-domain": "dom-a"},
		// h100 matches only the second group's selector.
		"gpu-1": {"nvswitch-domain": "sw-7"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Extract() across slice = %#v, want %#v", got, want)
	}

	// Re-running after a generation bump must still work end to end (new cache
	// scope) and produce identical results, proving the public Compile/Extract
	// contract is stable across generations.
	slice.Spec.Pool.Generation++
	bumped := poolFromSlice(slice)
	if err := extractor.Compile(ctx, bumped); err != nil {
		t.Fatalf("Compile() after generation bump error = %v, want nil", err)
	}
	for _, candidate := range candidatesFromSlice(slice) {
		result, err := extractor.Extract(ctx, bumped, req, candidate, configs)
		if err != nil {
			t.Fatalf("Extract(%s) after bump error = %v, want nil", candidate.ID, err)
		}
		if !reflect.DeepEqual(result, want[candidate.Device.Name]) {
			t.Errorf("Extract(%s) after bump = %#v, want %#v", candidate.ID, result, want[candidate.Device.Name])
		}
	}
}

// TestExtractTypedErrorPropagation verifies that extraction failures surface as
// classifiable typed errors carrying the rich *Error context an external
// consumer (the scheduler) needs to attribute the failure to a slice/claim/device.
func TestExtractTypedErrorPropagation(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	extractor := NewExtractor(32, dracel.NewCache(32, dracel.Features{}))

	slice := &resourceapi.ResourceSlice{
		Spec: resourceapi.ResourceSliceSpec{
			Driver:  integrationDriver,
			Pool:    resourceapi.ResourcePool{Name: "rack-1", Generation: 1},
			Devices: []resourceapi.Device{device("gpu-0", "a100")},
			SharingAffinity: []resourceapi.SharingAffinityExtractor{
				{CEL: map[string]string{"nvlink-domain": `object.spec.nvlinkDomain`}},
			},
		},
	}
	pool := poolFromSlice(slice)
	candidate := candidatesFromSlice(slice)[0]

	for _, tc := range []struct {
		name    string
		configs []resourceapi.DeviceClaimConfiguration
		want    error
	}{
		{
			name: "self-inconsistent claim",
			configs: []resourceapi.DeviceClaimConfiguration{
				opaque(`{"spec":{"nvlinkDomain":"dom-a"}}`),
				opaque(`{"spec":{"nvlinkDomain":"dom-b"}}`),
			},
			want: ErrClaimSelfInconsistent,
		},
		{
			name: "missing field",
			configs: []resourceapi.DeviceClaimConfiguration{
				opaque(`{"spec":{"unrelated":"x"}}`),
			},
			want: ErrMissingField,
		},
		{
			name: "malformed opaque parameters",
			configs: []resourceapi.DeviceClaimConfiguration{
				opaque(`{"spec": not-json}`),
			},
			want: ErrCELEval,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := extractor.Extract(ctx, pool, Request{Name: "gpu"}, candidate, tc.configs)
			if !errors.Is(err, tc.want) {
				t.Fatalf("Extract() error = %v, want errors.Is(..., %v)", err, tc.want)
			}
			// The error must expose the structured *Error context the scheduler
			// will use to attribute the failure.
			var affErr *Error
			if !errors.As(err, &affErr) {
				t.Fatalf("Extract() error = %v, want a *Error", err)
			}
			if affErr.Pool != pool.Key {
				t.Errorf("affErr.Pool = %#v, want %#v", affErr.Pool, pool.Key)
			}
			if affErr.Device != candidate.ID {
				t.Errorf("affErr.Device = %q, want %q", affErr.Device, candidate.ID)
			}
		})
	}
}

// TestExtractCostBudgetSharedWithSelectorRuntime verifies that extractor CEL
// evaluation is bounded by the same DRA runtime cost budget as selector CEL,
// surfaced through the public API as ErrCostExceeded.
func TestExtractCostBudgetSharedWithSelectorRuntime(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	extractor := NewExtractor(32, dracel.NewCache(32, dracel.Features{}))

	slice := &resourceapi.ResourceSlice{
		Spec: resourceapi.ResourceSliceSpec{
			Driver:  integrationDriver,
			Pool:    resourceapi.ResourcePool{Name: "rack-2", Generation: 1},
			Devices: []resourceapi.Device{device("gpu-0", "a100")},
			SharingAffinity: []resourceapi.SharingAffinityExtractor{
				{CEL: map[string]string{
					"nvlink-domain": `[1, 2, 3, 4, 5].exists(i, [1, 2, 3, 4, 5].exists(j, string(i) + string(j) == object.spec.nvlinkDomain)) ? object.spec.nvlinkDomain : ""`,
				}},
			},
		},
	}
	pool := poolFromSlice(slice)
	candidate := candidatesFromSlice(slice)[0]

	_, err := extractor.Extract(ctx, pool, Request{Name: "gpu"}, candidate, []resourceapi.DeviceClaimConfiguration{
		opaque(`{"spec":{"nvlinkDomain":"never"}}`),
	})
	if !errors.Is(err, ErrCostExceeded) {
		t.Fatalf("Extract() error = %v, want ErrCostExceeded", err)
	}
}
