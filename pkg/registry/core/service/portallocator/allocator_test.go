/*
Copyright 2015 The Kubernetes Authors.

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

package portallocator

import (
	"strconv"
	"testing"

	"k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/component-base/metrics/testutil"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestAllocate(t *testing.T) {
	pr, err := net.ParsePortRange("10000-10200")
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewInMemory(*pr)
	if err != nil {
		t.Fatal(err)
	}
	if f := r.Free(); f != 201 {
		t.Errorf("unexpected free %d", f)
	}
	if f := r.Used(); f != 0 {
		t.Errorf("unexpected used %d", f)
	}
	found := sets.NewString()
	count := 0
	for r.Free() > 0 {
		p, err := r.AllocateNext()
		if err != nil {
			t.Fatalf("error @ %d: %v", count, err)
		}
		count++
		if !pr.Contains(p) {
			t.Fatalf("allocated %d which is outside of %v", p, pr)
		}
		if found.Has(strconv.Itoa(p)) {
			t.Fatalf("allocated %d twice @ %d", p, count)
		}
		found.Insert(strconv.Itoa(p))
	}
	if _, err := r.AllocateNext(); err != ErrFull {
		t.Fatal(err)
	}

	released := 10005
	if err := r.Release(released); err != nil {
		t.Fatal(err)
	}
	if f := r.Free(); f != 1 {
		t.Errorf("unexpected free %d", f)
	}
	if f := r.Used(); f != 200 {
		t.Errorf("unexpected used %d", f)
	}
	p, err := r.AllocateNext()
	if err != nil {
		t.Fatal(err)
	}
	if released != p {
		t.Errorf("unexpected %d : %d", p, released)
	}

	if err := r.Release(released); err != nil {
		t.Fatal(err)
	}

	err = r.Allocate(1)
	if _, ok := err.(*ErrNotInRange); !ok {
		t.Fatal(err)
	}

	if err := r.Allocate(10001); err != ErrAllocated {
		t.Fatal(err)
	}

	err = r.Allocate(20000)
	if _, ok := err.(*ErrNotInRange); !ok {
		t.Fatal(err)
	}

	err = r.Allocate(10201)
	if _, ok := err.(*ErrNotInRange); !ok {
		t.Fatal(err)
	}
	if f := r.Free(); f != 1 {
		t.Errorf("unexpected free %d", f)
	}
	if f := r.Used(); f != 200 {
		t.Errorf("unexpected used %d", f)
	}
	if err := r.Allocate(released); err != nil {
		t.Fatal(err)
	}
	if f := r.Free(); f != 0 {
		t.Errorf("unexpected free %d", f)
	}
	if f := r.Used(); f != 201 {
		t.Errorf("unexpected used %d", f)
	}
}

func TestAllocateReserved(t *testing.T) {
	pr, err := net.ParsePortRange("30000-30128")
	if err != nil {
		t.Fatal(err)
	}

	r, err := NewInMemory(*pr)
	if err != nil {
		t.Fatal(err)
	}
	// allocate all ports on the dynamic block
	// dynamic block size is min(max(16,128/32),128) = 16
	dynamicOffset := calculateRangeOffset(*pr)
	dynamicBlockSize := pr.Size - dynamicOffset
	for i := 0; i < dynamicBlockSize; i++ {
		if _, err := r.AllocateNext(); err != nil {
			t.Errorf("Unexpected error trying to allocate: %v", err)
		}
	}
	for i := dynamicOffset; i < pr.Size; i++ {
		port := i + pr.Base
		if !r.Has(port) {
			t.Errorf("Port %d expected to be allocated", port)
		}
	}
	if f := r.Free(); f != dynamicOffset {
		t.Errorf("expected %d free ports, got %d", dynamicOffset, f)
	}
	// allocate all ports on the static block
	for i := 0; i < dynamicOffset; i++ {
		port := i + pr.Base
		if err := r.Allocate(port); err != nil {
			t.Errorf("Unexpected error trying to allocate Port %d: %v", port, err)
		}
	}
	if f := r.Free(); f != 0 {
		t.Errorf("expected free equal to 0 got: %d", f)
	}
	// release one port in the allocated block and another a new one randomly
	if err := r.Release(30053); err != nil {
		t.Fatalf("Unexpected error trying to release port 30053: %v", err)
	}
	if _, err := r.AllocateNext(); err != nil {
		t.Error(err)
	}
	if f := r.Free(); f != 0 {
		t.Errorf("expected free equal to 0 got: %d", f)
	}
}

func TestForEach(t *testing.T) {
	pr, err := net.ParsePortRange("10000-10200")
	if err != nil {
		t.Fatal(err)
	}

	testCases := []sets.Int{
		sets.NewInt(),
		sets.NewInt(10000),
		sets.NewInt(10000, 10200),
		sets.NewInt(10000, 10099, 10200),
	}

	for i, tc := range testCases {
		r, err := NewInMemory(*pr)
		if err != nil {
			t.Fatal(err)
		}

		for port := range tc {
			if err := r.Allocate(port); err != nil {
				t.Errorf("[%d] error allocating port %v: %v", i, port, err)
			}
			if !r.Has(port) {
				t.Errorf("[%d] expected port %v allocated", i, port)
			}
		}

		calls := sets.NewInt()
		r.ForEach(func(port int) {
			calls.Insert(port)
		})
		if len(calls) != len(tc) {
			t.Errorf("[%d] expected %d calls, got %d", i, len(tc), len(calls))
		}
		if !calls.Equal(tc) {
			t.Errorf("[%d] expected calls to equal testcase: %v vs %v", i, calls.List(), tc.List())
		}
	}
}

func TestSnapshot(t *testing.T) {
	pr, err := net.ParsePortRange("10000-10200")
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewInMemory(*pr)
	if err != nil {
		t.Fatal(err)
	}
	ports := []int{}
	for i := 0; i < 10; i++ {
		port, err := r.AllocateNext()
		if err != nil {
			t.Fatal(err)
		}
		ports = append(ports, port)
	}

	var dst api.RangeAllocation
	err = r.Snapshot(&dst)
	if err != nil {
		t.Fatal(err)
	}

	pr2, err := net.ParsePortRange(dst.Range)
	if err != nil {
		t.Fatal(err)
	}

	if pr.String() != pr2.String() {
		t.Fatalf("mismatched networks: %s : %s", pr, pr2)
	}

	otherPr, err := net.ParsePortRange("200-300")
	if err != nil {
		t.Fatal(err)
	}
	_, err = NewInMemory(*otherPr)
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Restore(*otherPr, dst.Data); err != ErrMismatchedNetwork {
		t.Fatal(err)
	}
	other, err := NewInMemory(*pr2)
	if err != nil {
		t.Fatal(err)
	}
	if err := other.Restore(*pr2, dst.Data); err != nil {
		t.Fatal(err)
	}

	for _, n := range ports {
		if !other.Has(n) {
			t.Errorf("restored range does not have %d", n)
		}
	}
	if other.Free() != r.Free() {
		t.Errorf("counts do not match: %d", other.Free())
	}
}

func TestNewFromSnapshot(t *testing.T) {
	pr, err := net.ParsePortRange("200-300")
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewInMemory(*pr)
	if err != nil {
		t.Fatal(err)
	}
	allocated := []int{}
	for i := 0; i < 50; i++ {
		p, err := r.AllocateNext()
		if err != nil {
			t.Fatal(err)
		}
		allocated = append(allocated, p)
	}

	snapshot := api.RangeAllocation{}
	if err = r.Snapshot(&snapshot); err != nil {
		t.Fatal(err)
	}

	r, err = NewFromSnapshot(&snapshot)
	if err != nil {
		t.Fatal(err)
	}

	if x := r.Free(); x != 51 {
		t.Fatalf("expected 51 free ports, got %d", x)
	}
	if x := r.Used(); x != 50 {
		t.Fatalf("expected 50 used port, got %d", x)
	}

	for _, p := range allocated {
		if !r.Has(p) {
			t.Fatalf("expected port to be allocated, but it was not")
		}
	}
}

func Test_calculateRangeOffset(t *testing.T) {
	type args struct {
		pr net.PortRange
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{
			name: "default node port range",
			args: args{
				pr: net.PortRange{
					Base: 30000,
					Size: 2768,
				},
			},
			want: 86,
		},
		{
			name: "very small node port range",
			args: args{
				pr: net.PortRange{
					Base: 30000,
					Size: 10,
				},
			},
			want: 0,
		},
		{
			name: "small node port range (lower boundary)",
			args: args{
				pr: net.PortRange{
					Base: 30000,
					Size: 16,
				},
			},
			want: 0,
		},
		{
			name: "small node port range",
			args: args{
				pr: net.PortRange{
					Base: 30000,
					Size: 128,
				},
			},
			want: 16,
		},
		{
			name: "medium node port range",
			args: args{
				pr: net.PortRange{
					Base: 30000,
					Size: 2048,
				},
			},
			want: 64,
		},
		{
			name: "large node port range (upper boundary)",
			args: args{
				pr: net.PortRange{
					Base: 30000,
					Size: 4096,
				},
			},
			want: 128,
		},
		{
			name: "large node port range",
			args: args{
				pr: net.PortRange{
					Base: 30000,
					Size: 8192,
				},
			},
			want: 128,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := calculateRangeOffset(tt.args.pr); got != tt.want {
				t.Errorf("calculateRangeOffset() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNodePortMetrics(t *testing.T) {
	clearMetrics()
	// create node port allocator
	portRange := "30000-32767"
	pr, err := net.ParsePortRange(portRange)
	if err != nil {
		t.Fatal(err)
	}

	a, err := NewInMemory(*pr)
	if err != nil {
		t.Fatalf("unexpected error creating nodeport allocator: %v", err)
	}
	a.EnableMetrics()

	// Check initial state
	em := testMetrics{
		free:      0,
		used:      0,
		allocated: 0,
		errors:    0,
	}
	expectMetrics(t, em)

	// allocate 2 ports
	found := sets.NewInt()
	for i := 0; i < 2; i++ {
		port, err := a.AllocateNext()
		if err != nil {
			t.Fatal(err)
		}
		if found.Has(port) {
			t.Fatalf("already reserved: %d", port)
		}
		found.Insert(port)
	}

	em = testMetrics{
		free:      2768 - 2,
		used:      2,
		allocated: 2,
		errors:    0,
	}
	expectMetrics(t, em)

	// try to allocate the same ports
	for s := range found {
		if !a.Has(s) {
			t.Fatalf("missing: %d", s)
		}
		if err := a.Allocate(s); err != ErrAllocated {
			t.Fatal(err)
		}
	}
	em = testMetrics{
		free:      2768 - 2,
		used:      2,
		allocated: 2,
		errors:    2,
	}
	expectMetrics(t, em)

	// release the ports allocated
	for s := range found {
		if !a.Has(s) {
			t.Fatalf("missing: %d", s)
		}
		if err := a.Release(s); err != nil {
			t.Fatal(err)
		}
	}
	em = testMetrics{
		free:      2768,
		used:      0,
		allocated: 2,
		errors:    2,
	}
	expectMetrics(t, em)

	// allocate 3000 ports for each allocator
	// the full range and 232 more (2768 + 232 = 3000)
	for i := 0; i < 3000; i++ {
		a.AllocateNext()
	}
	em = testMetrics{
		free:      0,
		used:      2768,
		allocated: 2768 + 2, // this is a counter, we already had 2 allocations and we did 2768 more
		errors:    232 + 2,  // this is a counter, we already had 2 errors and we did 232 more
	}
	expectMetrics(t, em)
}

func TestNodePortAllocatedMetrics(t *testing.T) {
	clearMetrics()

	// create NodePort allocator
	portRange := "30000-32767"
	pr, err := net.ParsePortRange(portRange)
	if err != nil {
		t.Fatal(err)
	}

	a, err := NewInMemory(*pr)
	if err != nil {
		t.Fatalf("unexpected error creating nodeport allocator: %v", err)
	}
	a.EnableMetrics()

	em := testMetrics{
		free:      0,
		used:      0,
		allocated: 0,
		errors:    0,
	}
	expectMetrics(t, em)

	// allocate 2 dynamic port
	found := sets.NewInt()
	for i := 0; i < 2; i++ {
		port, err := a.AllocateNext()
		if err != nil {
			t.Fatal(err)
		}
		if found.Has(port) {
			t.Fatalf("already reserved: %d", port)
		}
		found.Insert(port)
	}

	dynamicAllocated, err := testutil.GetCounterMetricValue(nodePortAllocations.WithLabelValues("dynamic"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", nodePortAllocations.Name, err)
	}
	if dynamicAllocated != 2 {
		t.Fatalf("Expected 2 received %f", dynamicAllocated)
	}

	// try to allocate the same ports
	for s := range found {
		if !a.Has(s) {
			t.Fatalf("missing: %d", s)
		}
		if err := a.Allocate(s); err != ErrAllocated {
			t.Fatal(err)
		}
	}

	staticErrors, err := testutil.GetCounterMetricValue(nodePortAllocationErrors.WithLabelValues("static"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", nodePortAllocationErrors.Name, err)
	}
	if staticErrors != 2 {
		t.Fatalf("Expected 2 received %f", staticErrors)
	}
}

func TestMetricsDisabled(t *testing.T) {
	clearMetrics()

	// create NodePort allocator
	portRange := "30000-32766"
	pr, err := net.ParsePortRange(portRange)
	if err != nil {
		t.Fatal(err)
	}

	a, err := NewInMemory(*pr)
	if err != nil {
		t.Fatalf("unexpected error creating nodeport allocator: %v", err)
	}
	a.EnableMetrics()

	// create metrics disabled allocator with same port range
	// this metrics should be ignored
	b, err := NewInMemory(*pr)
	if err != nil {
		t.Fatalf("unexpected error creating nodeport allocator: %v", err)
	}

	// Check initial state
	em := testMetrics{
		free:      0,
		used:      0,
		allocated: 0,
		errors:    0,
	}
	expectMetrics(t, em)

	// allocate in metrics enabled allocator
	for i := 0; i < 100; i++ {
		_, err := a.AllocateNext()
		if err != nil {
			t.Fatal(err)
		}
	}
	em = testMetrics{
		free:      2767 - 100,
		used:      100,
		allocated: 100,
		errors:    0,
	}
	expectMetrics(t, em)

	// allocate in metrics disabled allocator
	for i := 0; i < 200; i++ {
		_, err := b.AllocateNext()
		if err != nil {
			t.Fatal(err)
		}
	}
	// the metrics should not be changed
	expectMetrics(t, em)
}

// Metrics helpers
func clearMetrics() {
	nodePortAllocated.Set(0)
	nodePortAvailable.Set(0)
	nodePortAllocations.Reset()
	nodePortAllocationErrors.Reset()
}

type testMetrics struct {
	free      float64
	used      float64
	allocated float64
	errors    float64
}

func expectMetrics(t *testing.T, em testMetrics) {
	var m testMetrics
	var err error
	m.free, err = testutil.GetGaugeMetricValue(nodePortAvailable)
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", nodePortAvailable.Name, err)
	}
	m.used, err = testutil.GetGaugeMetricValue(nodePortAllocated)
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", nodePortAllocated.Name, err)
	}
	staticAllocated, err := testutil.GetCounterMetricValue(nodePortAllocations.WithLabelValues("static"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", nodePortAllocations.Name, err)
	}
	staticErrors, err := testutil.GetCounterMetricValue(nodePortAllocationErrors.WithLabelValues("static"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", nodePortAllocationErrors.Name, err)
	}
	dynamicAllocated, err := testutil.GetCounterMetricValue(nodePortAllocations.WithLabelValues("dynamic"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", nodePortAllocations.Name, err)
	}
	dynamicErrors, err := testutil.GetCounterMetricValue(nodePortAllocationErrors.WithLabelValues("dynamic"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", nodePortAllocationErrors.Name, err)
	}

	m.allocated = staticAllocated + dynamicAllocated
	m.errors = staticErrors + dynamicErrors

	if m != em {
		t.Fatalf("metrics error: expected %v, received %v", em, m)
	}
}
