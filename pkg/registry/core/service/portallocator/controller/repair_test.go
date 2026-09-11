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

package controller

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/net"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-base/metrics/testutil"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
)

type mockRangeRegistry struct {
	getCalled bool
	item      *api.RangeAllocation
	err       error

	updateCalled bool
	updated      *api.RangeAllocation
	updateErr    error
	updateErrors []error
}

func (r *mockRangeRegistry) Get() (*api.RangeAllocation, error) {
	r.getCalled = true
	return r.item, r.err
}

func (r *mockRangeRegistry) CreateOrUpdate(alloc *api.RangeAllocation) error {
	r.updateCalled = true
	r.updated = alloc
	if len(r.updateErrors) > 0 {
		err := r.updateErrors[0]
		r.updateErrors = r.updateErrors[1:]
		return err
	}
	return r.updateErr
}

func newTestRepair(t *testing.T, fakeClient *fake.Clientset, portRange net.PortRange, registry *mockRangeRegistry) *testRepair {
	t.Helper()

	// The fake tracker never stamps resource versions on objects, so the
	// informer's own store RV is not meaningful here. Both sides of the
	// freshness comparison are driven explicitly by the test instead.
	tr := &testRepair{
		cacheRV: &testRV{rv: "10"},
		liveRV:  &testRV{rv: "10"},
	}
	defaultReaction := clienttesting.ObjectReaction(fakeClient.Tracker())
	fakeClient.PrependReactor("list", "services", func(action clienttesting.Action) (bool, runtime.Object, error) {
		if err := tr.liveRV.getErr(); err != nil {
			return true, nil, err
		}
		handled, obj, err := defaultReaction(action)
		if list, ok := obj.(*corev1.ServiceList); ok && err == nil {
			list.ResourceVersion = tr.liveRV.get()
			// Only the freshness probe uses Limit=1; every other list reads storage.
			if action.(clienttesting.ListActionImpl).GetListOptions().Limit != 1 {
				list.Items = append(list.Items, tr.storageOnly()...)
			}
		}
		return handled, obj, err
	})

	informerFactory := informers.NewSharedInformerFactory(fakeClient, 0)
	serviceInformer := informerFactory.Core().V1().Services()
	tr.Repair = NewRepair(100*time.Millisecond, fakeClient.CoreV1(), serviceInformer, fakeClient.EventsV1(), portRange, registry)
	tr.Repair.serviceStore = &cache.FakeCustomStore{LastStoreSyncResourceVersionFunc: tr.cacheRV.get}

	stopCh := make(chan struct{})
	t.Cleanup(func() { close(stopCh) })
	informerFactory.Start(stopCh)
	if !cache.WaitForCacheSync(stopCh, tr.servicesSynced) {
		t.Fatal("failed to sync Service informer cache")
	}

	return tr
}

// testRepair exposes the two resource versions the freshness check compares:
// cacheRV is what the informer store reports, liveRV is what a LIST returns.
// Services added with addToStorageOnly are returned by LISTs against storage
// but never reach the informer, standing in for a lagging cache.
type testRepair struct {
	*Repair
	cacheRV *testRV
	liveRV  *testRV

	mu          sync.Mutex
	storageSvcs []corev1.Service
}

func (r *testRepair) addToStorageOnly(svc corev1.Service) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.storageSvcs = append(r.storageSvcs, svc)
}

func (r *testRepair) storageOnly() []corev1.Service {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]corev1.Service(nil), r.storageSvcs...)
}

type testRV struct {
	mu  sync.Mutex
	rv  string
	err error
}

func (r *testRV) set(rv string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.rv = rv
}

func (r *testRV) setErr(err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.err = err
}

func (r *testRV) get() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.rv
}

func (r *testRV) getErr() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.err
}

func TestRepair(t *testing.T) {
	clearMetrics()
	fakeClient := fake.NewSimpleClientset()
	registry := &mockRangeRegistry{
		item: &api.RangeAllocation{Range: "100-200"},
	}
	pr, _ := net.ParsePortRange(registry.item.Range)
	r := newTestRepair(t, fakeClient, *pr, registry)

	if err := r.runOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !registry.updateCalled || registry.updated == nil || registry.updated.Range != pr.String() || registry.updated != registry.item {
		t.Errorf("unexpected registry: %#v", registry)
	}
	repairErrors, err := testutil.GetCounterMetricValue(nodePortRepairReconcileErrors)
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", nodePortRepairReconcileErrors.Name, err)
	}
	if repairErrors != 0 {
		t.Fatalf("0 error expected, got %v", repairErrors)
	}

	registry = &mockRangeRegistry{
		item:      &api.RangeAllocation{Range: "100-200"},
		updateErr: fmt.Errorf("test error"),
	}
	r = newTestRepair(t, fakeClient, *pr, registry)
	if err := r.runOnce(context.Background()); !strings.Contains(err.Error(), ": test error") {
		t.Fatal(err)
	}
	repairErrors, err = testutil.GetCounterMetricValue(nodePortRepairReconcileErrors)
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", nodePortRepairReconcileErrors.Name, err)
	}
	if repairErrors != 1 {
		t.Fatalf("1 error expected, got %v", repairErrors)
	}
}

func TestRepairLeak(t *testing.T) {
	clearMetrics()

	pr, _ := net.ParsePortRange("100-200")
	previous, err := portallocator.NewInMemory(*pr)
	if err != nil {
		t.Fatal(err)
	}
	previous.Allocate(111)

	var dst api.RangeAllocation
	err = previous.Snapshot(&dst)
	if err != nil {
		t.Fatal(err)
	}

	fakeClient := fake.NewSimpleClientset()
	registry := &mockRangeRegistry{
		item: &api.RangeAllocation{
			ObjectMeta: metav1.ObjectMeta{
				ResourceVersion: "1",
			},
			Range: dst.Range,
			Data:  dst.Data,
		},
	}

	r := newTestRepair(t, fakeClient, *pr, registry)
	// Run through the "leak detection holdoff" loops.
	for i := 0; i < (numRepairsBeforeLeakCleanup - 1); i++ {
		if err := r.runOnce(context.Background()); err != nil {
			t.Fatal(err)
		}
		after, err := portallocator.NewFromSnapshot(registry.updated)
		if err != nil {
			t.Fatal(err)
		}
		if !after.Has(111) {
			t.Errorf("expected portallocator to still have leaked port")
		}
	}
	// Run one more time to actually remove the leak.
	if err := r.runOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	after, err := portallocator.NewFromSnapshot(registry.updated)
	if err != nil {
		t.Fatal(err)
	}
	if after.Has(111) {
		t.Errorf("expected portallocator to not have leaked port")
	}
	em := testMetrics{
		leak:       1,
		repair:     0,
		outOfRange: 0,
		duplicate:  0,
		unknown:    0,
	}
	expectMetrics(t, em)
}

// leakedRegistry returns a registry whose stored bitmap has the given ports
// allocated, with no Service referencing them.
func leakedRegistry(t *testing.T, pr net.PortRange, ports ...int) *mockRangeRegistry {
	t.Helper()
	previous, err := portallocator.NewInMemory(pr)
	if err != nil {
		t.Fatal(err)
	}
	for _, port := range ports {
		if err := previous.Allocate(port); err != nil {
			t.Fatal(err)
		}
	}
	var dst api.RangeAllocation
	if err := previous.Snapshot(&dst); err != nil {
		t.Fatal(err)
	}
	return &mockRangeRegistry{
		item: &api.RangeAllocation{
			ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"},
			Range:      dst.Range,
			Data:       dst.Data,
		},
	}
}

func updatedHas(t *testing.T, registry *mockRangeRegistry, port int) bool {
	t.Helper()
	after, err := portallocator.NewFromSnapshot(registry.updated)
	if err != nil {
		t.Fatal(err)
	}
	return after.Has(port)
}

func expectDeferred(t *testing.T, reason string, want float64) {
	t.Helper()
	got, err := testutil.GetCounterMetricValue(nodePortRepairLeakCleanupDeferred.WithLabelValues(reason))
	if err != nil {
		t.Fatalf("failed to get %s value: %v", nodePortRepairLeakCleanupDeferred.Name, err)
	}
	if got != want {
		t.Fatalf("expected %s{reason=%q} == %v, got %v", nodePortRepairLeakCleanupDeferred.Name, reason, want, got)
	}
}

// A port with no Service in the informer is only released once the informer
// is proven to have caught up with storage; until then the run still succeeds
// but the leak countdown must not even start.
func TestRepairLeakDeferredUntilCacheFresh(t *testing.T) {
	pr, _ := net.ParsePortRange("100-200")
	runs := numRepairsBeforeLeakCleanup + 1

	testCases := []struct {
		name   string
		reason string
		setup  func(r *testRepair)
	}{
		{
			name:   "informer behind storage",
			reason: "stale",
			setup: func(r *testRepair) {
				r.cacheRV.set("10")
				r.liveRV.set("20")
			},
		},
		{
			name:   "live read fails",
			reason: "error",
			setup:  func(r *testRepair) { r.liveRV.setErr(fmt.Errorf("storage is (re)initializing")) },
		},
		{
			name:   "resource versions are not comparable",
			reason: "error",
			setup:  func(r *testRepair) { r.cacheRV.set("not-a-number") },
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			clearMetrics()
			registry := leakedRegistry(t, *pr, 111)
			r := newTestRepair(t, fake.NewSimpleClientset(), *pr, registry)
			tc.setup(r)

			for i := range runs {
				if err := r.runOnce(context.Background()); err != nil {
					t.Fatalf("run %d: %v", i, err)
				}
				if !updatedHas(t, registry, 111) {
					t.Fatalf("run %d: port released while the informer was not proven fresh", i)
				}
				if len(r.leaks) != 0 {
					t.Fatalf("run %d: leak countdown started while the informer was not proven fresh: %v", i, r.leaks)
				}
			}
			expectDeferred(t, tc.reason, float64(runs))
			expectMetrics(t, testMetrics{})
		})
	}
}

// Once the informer catches up the normal holdoff applies from scratch: runs
// spent stale must not have consumed any of it.
func TestRepairLeakReleasedAfterCacheCatchesUp(t *testing.T) {
	clearMetrics()
	pr, _ := net.ParsePortRange("100-200")
	registry := leakedRegistry(t, *pr, 111)
	r := newTestRepair(t, fake.NewSimpleClientset(), *pr, registry)

	r.cacheRV.set("10")
	r.liveRV.set("20")
	for range numRepairsBeforeLeakCleanup + 1 {
		if err := r.runOnce(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	if !updatedHas(t, registry, 111) {
		t.Fatal("port released while stale")
	}

	r.cacheRV.set("20")
	for i := range numRepairsBeforeLeakCleanup - 1 {
		if err := r.runOnce(context.Background()); err != nil {
			t.Fatal(err)
		}
		if !updatedHas(t, registry, 111) {
			t.Fatalf("fresh run %d: port released before the holdoff expired", i)
		}
	}
	if err := r.runOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	if updatedHas(t, registry, 111) {
		t.Fatal("port still allocated after the holdoff expired on a fresh informer")
	}
	expectDeferred(t, "stale", float64(numRepairsBeforeLeakCleanup+1))
	expectMetrics(t, testMetrics{leak: 1})
}

// Staleness only gates releasing ports; repairing missing allocations for
// Services the informer does see must proceed regardless.
func TestRepairStaleCacheStillRepairsAllocations(t *testing.T) {
	clearMetrics()
	pr, _ := net.ParsePortRange("100-200")
	registry := leakedRegistry(t, *pr, 122)
	fakeClient := fake.NewSimpleClientset(&corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "one", Name: "one"},
		Spec:       corev1.ServiceSpec{Ports: []corev1.ServicePort{{NodePort: 111}}},
	})
	r := newTestRepair(t, fakeClient, *pr, registry)
	r.cacheRV.set("10")
	r.liveRV.set("20")

	if err := r.runOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !updatedHas(t, registry, 111) {
		t.Error("missing allocation for a visible Service was not repaired")
	}
	if !updatedHas(t, registry, 122) {
		t.Error("apparently leaked port released while stale")
	}
	expectDeferred(t, "stale", 1)
	expectMetrics(t, testMetrics{repair: 1})
}

// Without a store resource version (client-go AtomicFIFO disabled) staleness
// cannot be measured, so the loop lists storage as before and never consults
// the informer: a Service the informer never saw is still repaired, and a
// genuine leak is still released after the usual holdoff.
func TestRepairWithoutStoreResourceVersionReadsStorage(t *testing.T) {
	clearMetrics()
	pr, _ := net.ParsePortRange("100-200")
	registry := leakedRegistry(t, *pr, 122)
	r := newTestRepair(t, fake.NewSimpleClientset(), *pr, registry)
	r.cacheRV.set("")
	r.addToStorageOnly(corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "one", Name: "one"},
		Spec:       corev1.ServiceSpec{Ports: []corev1.ServicePort{{NodePort: 111}}},
	})

	for i := range numRepairsBeforeLeakCleanup - 1 {
		if err := r.runOnce(context.Background()); err != nil {
			t.Fatalf("run %d: %v", i, err)
		}
		if !updatedHas(t, registry, 111) {
			t.Fatalf("run %d: Service present only in storage was not repaired; lister used instead of storage", i)
		}
		if !updatedHas(t, registry, 122) {
			t.Fatalf("run %d: leaked port released before the holdoff expired", i)
		}
	}
	if err := r.runOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !updatedHas(t, registry, 111) {
		t.Error("Service present only in storage was not repaired; lister used instead of storage")
	}
	if updatedHas(t, registry, 122) {
		t.Error("leaked port still allocated after the holdoff expired")
	}
	for _, reason := range []string{"stale", "error"} {
		expectDeferred(t, reason, 0)
	}
	// The mock registry returns the snapshot the loop writes into, so the
	// repaired port is already stored on the following runs.
	expectMetrics(t, testMetrics{repair: 1, leak: 1})
}

func TestRepairWithExisting(t *testing.T) {
	clearMetrics()
	pr, _ := net.ParsePortRange("100-200")
	previous, err := portallocator.NewInMemory(*pr)
	if err != nil {
		t.Fatal(err)
	}

	var dst api.RangeAllocation
	err = previous.Snapshot(&dst)
	if err != nil {
		t.Fatal(err)
	}

	fakeClient := fake.NewSimpleClientset(
		&corev1.Service{
			ObjectMeta: metav1.ObjectMeta{Namespace: "one", Name: "one"},
			Spec: corev1.ServiceSpec{
				Ports: []corev1.ServicePort{{NodePort: 111}},
			},
		},
		&corev1.Service{
			ObjectMeta: metav1.ObjectMeta{Namespace: "two", Name: "two"},
			Spec: corev1.ServiceSpec{
				Ports: []corev1.ServicePort{{NodePort: 122}, {NodePort: 133}},
			},
		},
		&corev1.Service{ // outside range, will be dropped
			ObjectMeta: metav1.ObjectMeta{Namespace: "three", Name: "three"},
			Spec: corev1.ServiceSpec{
				Ports: []corev1.ServicePort{{NodePort: 201}},
			},
		},
		&corev1.Service{ // empty, ignored
			ObjectMeta: metav1.ObjectMeta{Namespace: "four", Name: "four"},
			Spec: corev1.ServiceSpec{
				Ports: []corev1.ServicePort{{}},
			},
		},
		&corev1.Service{ // duplicate, dropped
			ObjectMeta: metav1.ObjectMeta{Namespace: "five", Name: "five"},
			Spec: corev1.ServiceSpec{
				Ports: []corev1.ServicePort{{NodePort: 111}},
			},
		},
		&corev1.Service{
			ObjectMeta: metav1.ObjectMeta{Namespace: "six", Name: "six"},
			Spec: corev1.ServiceSpec{
				HealthCheckNodePort: 144,
			},
		},
	)

	registry := &mockRangeRegistry{
		item: &api.RangeAllocation{
			ObjectMeta: metav1.ObjectMeta{
				ResourceVersion: "1",
			},
			Range: dst.Range,
			Data:  dst.Data,
		},
	}
	r := newTestRepair(t, fakeClient, *pr, registry)
	if err := r.runOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	after, err := portallocator.NewFromSnapshot(registry.updated)
	if err != nil {
		t.Fatal(err)
	}
	if !after.Has(111) || !after.Has(122) || !after.Has(133) || !after.Has(144) {
		t.Errorf("unexpected portallocator state: %#v", after)
	}
	if free := after.Free(); free != 97 {
		t.Errorf("unexpected portallocator state: %d free", free)
	}
	em := testMetrics{
		leak:       0,
		repair:     4,
		outOfRange: 1,
		duplicate:  1,
		unknown:    0,
	}
	expectMetrics(t, em)
}

func TestCollectServiceNodePorts(t *testing.T) {
	tests := []struct {
		name        string
		serviceSpec corev1.ServiceSpec
		expected    []int
	}{
		{
			name: "no duplicated nodePorts",
			serviceSpec: corev1.ServiceSpec{
				Ports: []corev1.ServicePort{
					{NodePort: 111, Protocol: corev1.ProtocolTCP},
					{NodePort: 112, Protocol: corev1.ProtocolUDP},
					{NodePort: 113, Protocol: corev1.ProtocolUDP},
				},
			},
			expected: []int{111, 112, 113},
		},
		{
			name: "duplicated nodePort with TCP protocol",
			serviceSpec: corev1.ServiceSpec{
				Ports: []corev1.ServicePort{
					{NodePort: 111, Protocol: corev1.ProtocolTCP},
					{NodePort: 111, Protocol: corev1.ProtocolTCP},
					{NodePort: 112, Protocol: corev1.ProtocolUDP},
				},
			},
			expected: []int{111, 111, 112},
		},
		{
			name: "duplicated nodePort with UDP protocol",
			serviceSpec: corev1.ServiceSpec{
				Ports: []corev1.ServicePort{
					{NodePort: 111, Protocol: corev1.ProtocolUDP},
					{NodePort: 111, Protocol: corev1.ProtocolUDP},
					{NodePort: 112, Protocol: corev1.ProtocolTCP},
				},
			},
			expected: []int{111, 111, 112},
		},
		{
			name: "duplicated nodePort with different protocol",
			serviceSpec: corev1.ServiceSpec{
				Ports: []corev1.ServicePort{
					{NodePort: 111, Protocol: corev1.ProtocolTCP},
					{NodePort: 112, Protocol: corev1.ProtocolTCP},
					{NodePort: 111, Protocol: corev1.ProtocolUDP},
				},
			},
			expected: []int{111, 112},
		},
		{
			name: "no duplicated port(with health check port)",
			serviceSpec: corev1.ServiceSpec{
				Ports: []corev1.ServicePort{
					{NodePort: 111, Protocol: corev1.ProtocolTCP},
					{NodePort: 112, Protocol: corev1.ProtocolUDP},
				},
				HealthCheckNodePort: 113,
			},
			expected: []int{111, 112, 113},
		},
		{
			name: "nodePort has different protocol with duplicated health check port",
			serviceSpec: corev1.ServiceSpec{
				Ports: []corev1.ServicePort{
					{NodePort: 111, Protocol: corev1.ProtocolUDP},
					{NodePort: 112, Protocol: corev1.ProtocolTCP},
				},
				HealthCheckNodePort: 111,
			},
			expected: []int{111, 112},
		},
		{
			name: "nodePort has same protocol as duplicated health check port",
			serviceSpec: corev1.ServiceSpec{
				Ports: []corev1.ServicePort{
					{NodePort: 111, Protocol: corev1.ProtocolUDP},
					{NodePort: 112, Protocol: corev1.ProtocolTCP},
				},
				HealthCheckNodePort: 112,
			},
			expected: []int{111, 112, 112},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ports := collectServiceNodePorts(&corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "one", Name: "one"},
				Spec:       tc.serviceSpec,
			})
			sort.Ints(ports)
			if !reflect.DeepEqual(tc.expected, ports) {
				t.Fatalf("Invalid result\nexpected: %v\ngot: %v", tc.expected, ports)
			}
		})
	}
}

// Metrics helpers
func clearMetrics() {
	nodePortRepairPortErrors.Reset()
	nodePortRepairReconcileErrors.Reset()
	nodePortRepairLeakCleanupDeferred.Reset()
}

type testMetrics struct {
	leak       float64
	repair     float64
	outOfRange float64
	duplicate  float64
	unknown    float64
	full       float64
}

func expectMetrics(t *testing.T, em testMetrics) {
	var m testMetrics
	var err error

	m.leak, err = testutil.GetCounterMetricValue(nodePortRepairPortErrors.WithLabelValues("leak"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", nodePortRepairPortErrors.Name, err)
	}
	m.repair, err = testutil.GetCounterMetricValue(nodePortRepairPortErrors.WithLabelValues("repair"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", nodePortRepairPortErrors.Name, err)
	}
	m.outOfRange, err = testutil.GetCounterMetricValue(nodePortRepairPortErrors.WithLabelValues("outOfRange"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", nodePortRepairPortErrors.Name, err)
	}
	m.duplicate, err = testutil.GetCounterMetricValue(nodePortRepairPortErrors.WithLabelValues("duplicate"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", nodePortRepairPortErrors.Name, err)
	}
	m.unknown, err = testutil.GetCounterMetricValue(nodePortRepairPortErrors.WithLabelValues("unknown"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", nodePortRepairPortErrors.Name, err)
	}
	m.full, err = testutil.GetCounterMetricValue(nodePortRepairPortErrors.WithLabelValues("full"))
	if err != nil {
		t.Errorf("failed to get %s value, err: %v", nodePortRepairPortErrors.Name, err)
	}
	if m != em {
		t.Fatalf("metrics error: expected %v, received %v", em, m)
	}
}

// A fixed barrier must remain attainable when writes continue after the probe.
func TestRepairCacheCatchesUpDuringRun(t *testing.T) {
	clearMetrics()
	pr, _ := net.ParsePortRange("100-200")
	registry := leakedRegistry(t, *pr, 111)
	client := fake.NewSimpleClientset()
	r := newTestRepair(t, client, *pr, registry)
	r.cacheRV.set("10")
	r.liveRV.set("20")
	probed := make(chan struct{})
	client.PrependReactor("list", "services", func(action clienttesting.Action) (bool, runtime.Object, error) {
		if action.(clienttesting.ListActionImpl).GetListOptions().Limit != 1 {
			return false, nil, nil
		}
		close(probed)
		return true, &corev1.ServiceList{ListMeta: metav1.ListMeta{ResourceVersion: "20"}}, nil
	})
	done := make(chan error, 1)
	go func() { done <- r.runOnce(context.Background()) }()
	select {
	case <-probed:
	case <-time.After(5 * time.Second):
		t.Fatal("repair did not probe the Service collection")
	}
	// Storage moves ahead, but the cache need only reach the captured barrier.
	r.liveRV.set("30")
	r.cacheRV.set("20")
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("repair did not finish after the cache reached the barrier")
	}
	if _, found := r.leaks[111]; !found {
		t.Fatal("fresh cache did not advance leak detection")
	}
	expectDeferred(t, "stale", 0)
}

// A stalled freshness request must not block periodic repairs indefinitely.
func TestRepairWithStalledFreshnessRequest(t *testing.T) {
	clearMetrics()
	pr, _ := net.ParsePortRange("100-200")
	registry := leakedRegistry(t, *pr, 111)
	r := newTestRepair(t, fake.NewSimpleClientset(), *pr, registry)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		<-req.Context().Done()
	}))
	defer server.Close()
	client, err := kubernetes.NewForConfig(&rest.Config{Host: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	r.serviceClient = client.CoreV1()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := r.runOnce(ctx); err != nil {
		t.Fatalf("repair did not finish within one repair interval: %v", err)
	}
	if !updatedHas(t, registry, 111) {
		t.Fatal("stalled freshness check released a port")
	}
	if len(r.leaks) != 0 {
		t.Fatal("stalled freshness check advanced leak detection")
	}
	expectDeferred(t, "error", 1)
}

func TestRepairStaleCachePreservesLeakCountdown(t *testing.T) {
	clearMetrics()
	pr, _ := net.ParsePortRange("100-200")
	registry := leakedRegistry(t, *pr, 111)
	r := newTestRepair(t, fake.NewSimpleClientset(), *pr, registry)
	if err := r.runOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	remaining := r.leaks[111]
	r.liveRV.set("20")
	if err := r.runOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !updatedHas(t, registry, 111) || r.leaks[111] != remaining {
		t.Fatal("stale cache released a port or changed its existing leak countdown")
	}
}

func TestRepairCanceledWhileCacheIsStale(t *testing.T) {
	clearMetrics()
	pr, _ := net.ParsePortRange("100-200")
	registry := leakedRegistry(t, *pr, 111)
	client := fake.NewSimpleClientset()
	r := newTestRepair(t, client, *pr, registry)
	r.interval = time.Hour
	r.liveRV.set("20")
	probed := make(chan struct{})
	client.PrependReactor("list", "services", func(action clienttesting.Action) (bool, runtime.Object, error) {
		close(probed)
		return false, nil, nil
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- r.runOnce(ctx) }()
	select {
	case <-probed:
	case <-time.After(5 * time.Second):
		t.Fatal("repair did not probe the Service collection")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected context cancellation, got %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("repair did not stop after context cancellation")
	}
	if registry.updateCalled || len(r.leaks) != 0 {
		t.Fatal("canceled repair changed allocations or leak cleanup counters")
	}
}

// Freshness failures must not prevent the first repair from using the informer
// and reporting success within the caller's startup window.
func TestRepairFirstSuccessWithUnavailableFreshness(t *testing.T) {
	for _, scenario := range []string{"stale cache", "request fails", "request stalls", "cache sync uses freshness budget", "conflict retry"} {
		t.Run(scenario, func(t *testing.T) {
			clearMetrics()
			pr, _ := net.ParsePortRange("100-200")
			registry := leakedRegistry(t, *pr, 122)
			fakeClient := fake.NewSimpleClientset(&corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "one", Name: "one"},
				Spec:       corev1.ServiceSpec{Ports: []corev1.ServicePort{{NodePort: 111}}},
			})
			r := newTestRepair(t, fakeClient, *pr, registry)
			r.interval = time.Hour
			r.leaks[122] = 1
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				if req.URL.Query().Get("limit") != "1" {
					t.Error("initial repair issued a full Service LIST instead of using the informer")
					http.Error(w, "full LIST is unavailable", http.StatusTooManyRequests)
					return
				}
				switch scenario {
				case "request fails":
					http.Error(w, "Service storage unavailable", http.StatusServiceUnavailable)
				case "request stalls":
					<-req.Context().Done()
				default:
					w.Header().Set("Content-Type", "application/json")
					// The freshness deadline may cancel the request during the write.
					_, _ = fmt.Fprint(w, `{"apiVersion":"v1","kind":"ServiceList","metadata":{"resourceVersion":"20"},"items":[]}`)
				}
			}))
			defer server.Close()
			client, err := kubernetes.NewForConfig(&rest.Config{Host: server.URL})
			if err != nil {
				t.Fatal(err)
			}
			r.serviceClient = client.CoreV1()
			deadline := time.Now().Add(100 * time.Millisecond)
			if scenario == "cache sync uses freshness budget" {
				r.servicesSynced = func() bool { return time.Now().After(deadline) }
			}
			if scenario == "conflict retry" {
				registry.updateErrors = []error{apierrors.NewConflict(schema.GroupResource{Resource: "servicenodeportallocations"}, "nodeports", fmt.Errorf("allocation changed"))}
			}
			func() {
				stop := make(chan struct{})
				done := make(chan struct{})
				success := make(chan struct{})
				go func() {
					defer close(done)
					r.RunUntil(func() { close(success) }, stop, deadline)
				}()
				defer func() { close(stop); <-done }()
				select {
				case <-success:
				case <-time.After(5 * time.Second):
					t.Fatal("initial repair did not finish after the freshness deadline")
				}
			}()
			if !updatedHas(t, registry, 111) {
				t.Fatal("initial repair did not allocate the port of a Service visible in the informer")
			}
			if !updatedHas(t, registry, 122) || r.leaks[122] != 1 {
				t.Fatal("initial repair released a port or advanced leak cleanup without a fresh cache")
			}
			if !r.serviceCacheDeadline.IsZero() {
				t.Fatal("successful initial repair retained the startup freshness deadline")
			}
			if len(registry.updateErrors) != 0 {
				t.Fatal("initial repair did not attempt to persist allocations")
			}
		})
	}
}
