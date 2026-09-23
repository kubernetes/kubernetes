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

package structured

import (
	"context"
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/utils/ptr"
)

// fakeClassLister implements DeviceClassLister over a fixed map.
type fakeClassLister map[string]*resourceapi.DeviceClass

func (f fakeClassLister) List() ([]*resourceapi.DeviceClass, error) {
	var out []*resourceapi.DeviceClass
	for _, class := range f {
		out = append(out, class)
	}
	return out, nil
}

func (f fakeClassLister) Get(name string) (*resourceapi.DeviceClass, error) {
	class, ok := f[name]
	if !ok {
		return nil, fmt.Errorf("class %q not found", name)
	}
	return class, nil
}

// makeConstraintFixture produces the smallest objects that make Allocate
// succeed with one device: one class, one node, one slice with one device,
// and one claim requesting exactly that device.
func makeConstraintFixture() (fakeClassLister, *v1.Node, []*resourceapi.ResourceSlice, []*resourceapi.ResourceClaim) {
	className := "gpu"
	driverName := "gpu-driver"

	class := &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{Name: className},
		Spec: resourceapi.DeviceClassSpec{
			Selectors: []resourceapi.DeviceSelector{{
				CEL: &resourceapi.CELDeviceSelector{
					Expression: fmt.Sprintf(`device.driver == %q`, driverName),
				},
			}},
		},
	}

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node-1"},
	}

	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{Name: "slice-1"},
		Spec: resourceapi.ResourceSliceSpec{
			Driver: driverName,
			Pool: resourceapi.ResourcePool{
				Name:               "pool",
				ResourceSliceCount: 1,
				Generation:         1,
			},
			NodeName: ptr.To("node-1"),
			Devices:  []resourceapi.Device{{Name: "device-1"}},
		},
	}

	claim := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "claim-1", Namespace: "default"},
		Spec: resourceapi.ResourceClaimSpec{
			Devices: resourceapi.DeviceClaim{
				Requests: []resourceapi.DeviceRequest{{
					Name: "gpu-request",
					Exactly: &resourceapi.ExactDeviceRequest{
						DeviceClassName: className,
						AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
						Count:           1,
					},
				}},
			},
		},
	}

	return fakeClassLister{className: class}, node, []*resourceapi.ResourceSlice{slice}, []*resourceapi.ResourceClaim{claim}
}

// recordingConstraint records every Add and Remove call.
type recordingConstraint struct {
	added   []DeviceAllocation
	removed []DeviceAllocation
}

func (c *recordingConstraint) Add(allocation DeviceAllocation) bool {
	c.added = append(c.added, allocation)
	return true
}

func (c *recordingConstraint) Remove(allocation DeviceAllocation) {
	c.removed = append(c.removed, allocation)
}

func TestAllocationConstraintSeesFinalValues(t *testing.T) {
	classLister, node, slices, claims := makeConstraintFixture()
	recorder := &recordingConstraint{}

	allocator, err := NewAllocator(context.Background(),
		Features{ConsumableCapacity: true},
		AllocatedState{},
		classLister, slices, cel.NewCache(10, cel.Features{}),
		WithAllocationConstraints(func(ctx context.Context, node *v1.Node,
			claims []*resourceapi.ResourceClaim) ([]AllocationConstraint, error) {
			return []AllocationConstraint{recorder}, nil
		}),
	)
	if err != nil {
		t.Fatalf("NewAllocator: %v", err)
	}

	result, err := allocator.Allocate(context.Background(), node, claims)
	if err != nil {
		t.Fatalf("Allocate: %v", err)
	}
	if len(result) != 1 {
		t.Fatalf("want 1 result, got %d", len(result))
	}

	if len(recorder.added) == 0 {
		t.Fatal("constraint Add was never called")
	}
	got := recorder.added[len(recorder.added)-1]

	if got.Claim.Name != "claim-1" {
		t.Errorf("Claim.Name = %q, want claim-1", got.Claim.Name)
	}
	if got.Request != "gpu-request" {
		t.Errorf("Request = %q, want gpu-request", got.Request)
	}
	if got.DeviceClassName != "gpu" {
		t.Errorf("DeviceClassName = %q, want gpu", got.DeviceClassName)
	}
	if got.Device.String() != "gpu-driver/pool/device-1" {
		t.Errorf("Device = %q, want gpu-driver/pool/device-1", got.Device.String())
	}
	if got.AdminAccess {
		t.Errorf("AdminAccess = true, want false")
	}
}

func TestAllocationConstraintAddRemoveSymmetry(t *testing.T) {
	classLister, node, slices, claims := makeConstraintFixture()
	recorder := &recordingConstraint{}

	allocator, err := NewAllocator(context.Background(),
		Features{ConsumableCapacity: true}, AllocatedState{},
		classLister, slices, cel.NewCache(10, cel.Features{}),
		WithAllocationConstraints(func(ctx context.Context, node *v1.Node,
			claims []*resourceapi.ResourceClaim) ([]AllocationConstraint, error) {
			return []AllocationConstraint{recorder}, nil
		}),
	)
	if err != nil {
		t.Fatalf("NewAllocator: %v", err)
	}

	_, err = allocator.Allocate(context.Background(), node, claims)
	if err != nil {
		t.Fatalf("Allocate: %v", err)
	}

	// A successful allocation records Add without a matching Remove:
	// the pair is only guaranteed for backtrack, not for the final result.
	if len(recorder.added) == 0 {
		t.Fatal("Add was never called")
	}
}

// rejectingConstraint rejects every candidate after recording it.
type rejectingConstraint struct {
	added []DeviceAllocation
}

func (c *rejectingConstraint) Add(allocation DeviceAllocation) bool {
	c.added = append(c.added, allocation)
	return false
}

func (c *rejectingConstraint) Remove(allocation DeviceAllocation) {}

func TestAllocationConstraintRejectionFailsAllocation(t *testing.T) {
	classLister, node, slices, claims := makeConstraintFixture()
	rejector := &rejectingConstraint{}

	allocator, err := NewAllocator(context.Background(),
		Features{ConsumableCapacity: true}, AllocatedState{},
		classLister, slices, cel.NewCache(10, cel.Features{}),
		WithAllocationConstraints(func(ctx context.Context, node *v1.Node,
			claims []*resourceapi.ResourceClaim) ([]AllocationConstraint, error) {
			return []AllocationConstraint{rejector}, nil
		}),
	)
	if err != nil {
		t.Fatalf("NewAllocator: %v", err)
	}

	result, err := allocator.Allocate(context.Background(), node, claims)
	if err != nil {
		t.Fatalf("Allocate returned error, want nil result: %v", err)
	}
	if result != nil {
		t.Fatalf("want nil result when all candidates are rejected, got %+v", result)
	}
	if len(rejector.added) == 0 {
		t.Fatal("rejecting constraint was never consulted")
	}
}

func TestNoConstraintProviderUnchanged(t *testing.T) {
	classLister, node, slices, claims := makeConstraintFixture()

	withOption, err := NewAllocator(context.Background(),
		Features{ConsumableCapacity: true}, AllocatedState{},
		classLister, slices, cel.NewCache(10, cel.Features{}),
		WithAllocationConstraints(nil),
	)
	if err != nil {
		t.Fatalf("NewAllocator with option: %v", err)
	}
	withoutOption, err := NewAllocator(context.Background(),
		Features{ConsumableCapacity: true}, AllocatedState{},
		classLister, slices, cel.NewCache(10, cel.Features{}),
	)
	if err != nil {
		t.Fatalf("NewAllocator without option: %v", err)
	}

	r1, err1 := withOption.Allocate(context.Background(), node, claims)
	r2, err2 := withoutOption.Allocate(context.Background(), node, claims)

	if (err1 == nil) != (err2 == nil) {
		t.Errorf("error mismatch: with option %v, without %v", err1, err2)
	}
	if len(r1) != len(r2) {
		t.Errorf("result length mismatch: with option %d, without %d", len(r1), len(r2))
	}
}

// budgetConstraint counts devices and rejects beyond a fixed limit.
type budgetConstraint struct {
	limit   int
	current int
}

func (c *budgetConstraint) Add(allocation DeviceAllocation) bool {
	if c.current >= c.limit {
		return false
	}
	c.current++
	return true
}

func (c *budgetConstraint) Remove(allocation DeviceAllocation) {
	c.current--
}

func TestAllocationConstraintBudgetBacktracks(t *testing.T) {
	// Two devices, budget of one.
	classLister, node, slices, claims := makeConstraintFixture()
	slices[0].Spec.Devices = append(slices[0].Spec.Devices, resourceapi.Device{Name: "device-2"})
	claims[0].Spec.Devices.Requests[0].Exactly.Count = 2
	budget := &budgetConstraint{limit: 1}

	allocator, err := NewAllocator(context.Background(),
		Features{ConsumableCapacity: true}, AllocatedState{},
		classLister, slices, cel.NewCache(10, cel.Features{}),
		WithAllocationConstraints(func(ctx context.Context, node *v1.Node,
			claims []*resourceapi.ResourceClaim) ([]AllocationConstraint, error) {
			return []AllocationConstraint{budget}, nil
		}),
	)
	if err != nil {
		t.Fatalf("NewAllocator: %v", err)
	}

	result, err := allocator.Allocate(context.Background(), node, claims)
	if err != nil {
		t.Fatalf("Allocate: %v", err)
	}
	if result != nil {
		t.Fatalf("want nil result when budget prevents allocating 2 devices, got %+v", result)
	}
}
