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
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"strings"
	"syscall"
	"testing"
	"testing/synctest"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/net"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes/fake"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/rest"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/util/retry"
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
	updateCalls  int
}

func (r *mockRangeRegistry) Get() (*api.RangeAllocation, error) {
	r.getCalled = true
	return r.item, r.err
}

func (r *mockRangeRegistry) CreateOrUpdate(alloc *api.RangeAllocation) error {
	r.updateCalls++
	r.updateCalled = true
	r.updated = alloc
	if len(r.updateErrors) > 0 {
		err := r.updateErrors[0]
		r.updateErrors = r.updateErrors[1:]
		return err
	}
	return r.updateErr
}

func TestRepairRetriesServiceList(t *testing.T) {
	resource := schema.GroupResource{Resource: "services"}
	for _, tc := range []struct {
		name      string
		err       error
		retriable bool
	}{
		{"conflict", apierrors.NewConflict(resource, "test", errors.New("conflict")), true},
		{"too many requests", apierrors.NewTooManyRequests("cache initializing", 0), true},
		{"server timeout", apierrors.NewServerTimeout(resource, "list", 0), true},
		{"timeout", apierrors.NewTimeoutError("timeout", 0), true},
		{"unavailable", apierrors.NewServiceUnavailable("unavailable"), true},
		{"internal", apierrors.NewInternalError(errors.New("internal")), true},
		{"EOF", io.EOF, true},
		{"unexpected EOF", io.ErrUnexpectedEOF, true},
		{"connection reset", syscall.ECONNRESET, true},
		{"forbidden", apierrors.NewForbidden(resource, "test", errors.New("forbidden")), false},
		{"unknown", errors.New("unknown"), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewSimpleClientset()
			calls := 0
			client.PrependReactor("list", "services", func(clienttesting.Action) (bool, runtime.Object, error) {
				calls++
				if calls == 1 {
					return true, nil, tc.err
				}
				return true, &corev1.ServiceList{}, nil
			})
			registry := &mockRangeRegistry{item: &api.RangeAllocation{Range: "100-200"}}
			pr, _ := net.ParsePortRange(registry.item.Range)
			r := NewRepair(0, client.CoreV1(), client.EventsV1(), *pr, registry)
			err := r.runOnce(context.Background())
			if tc.retriable {
				if err != nil || calls != 2 || !registry.updateCalled {
					t.Fatalf("expected successful retry and persistence, got calls=%d, persisted=%t, err=%v", calls, registry.updateCalled, err)
				}
			} else if !errors.Is(err, tc.err) || calls != 1 || registry.updateCalled {
				t.Fatalf("expected original error without retry or persistence, got calls=%d, persisted=%t, err=%v", calls, registry.updateCalled, err)
			}
		})
	}
}

func TestRepairRetriesDoNotAdvanceLeaks(t *testing.T) {
	for _, failures := range []int{retry.DefaultBackoff.Steps - 1, retry.DefaultBackoff.Steps} {
		t.Run(fmt.Sprintf("failed writes=%d", failures), func(t *testing.T) {
			client := fake.NewSimpleClientset()
			pr, _ := net.ParsePortRange("100-200")
			previous, err := portallocator.NewInMemory(*pr)
			if err != nil {
				t.Fatal(err)
			}
			if err := previous.Allocate(111); err != nil {
				t.Fatal(err)
			}
			snapshot := &api.RangeAllocation{}
			if err := previous.Snapshot(snapshot); err != nil {
				t.Fatal(err)
			}
			registry := &mockRangeRegistry{item: snapshot}
			for range failures {
				registry.updateErrors = append(registry.updateErrors, apierrors.NewServiceUnavailable("storage unavailable"))
			}
			r := NewRepair(0, client.CoreV1(), client.EventsV1(), *pr, registry)
			if err := r.runOnce(context.Background()); failures == retry.DefaultBackoff.Steps {
				if !apierrors.IsServiceUnavailable(err) || len(r.leaks) != 0 {
					t.Fatalf("expected exhausted retries without changing leak counts, got leaks=%v, err=%v", r.leaks, err)
				}
				if err := r.runOnce(context.Background()); err != nil {
					t.Fatal(err)
				}
			} else if err != nil {
				t.Fatal(err)
			}
			if registry.updateCalls != failures+1 {
				t.Fatalf("expected %d writes, got %d", failures+1, registry.updateCalls)
			}
			if got := r.leaks[111]; got != numRepairsBeforeLeakCleanup-2 {
				t.Fatalf("expected one successful leak observation, got count %d", got)
			}
			after, err := portallocator.NewFromSnapshot(registry.updated)
			if err != nil {
				t.Fatal(err)
			}
			if !after.Has(111) {
				t.Fatal("retries released a port before the leak observation period elapsed")
			}
		})
	}
}

func TestRepairRunUntil(t *testing.T) {
	// The default error handler's rate limiter retains wall-clock timestamps.
	// Disable it while using synctest's virtual clock.
	handlers := utilruntime.ErrorHandlers
	utilruntime.ErrorHandlers = nil
	defer func() { utilruntime.ErrorHandlers = handlers }()
	synctest.Test(t, func(t *testing.T) {
		client := fake.NewSimpleClientset()
		calls := 0
		var successfulLists []time.Time
		client.PrependReactor("list", "services", func(clienttesting.Action) (bool, runtime.Object, error) {
			calls++
			// Exhaust runOnce's retries to exercise the initial sync loop.
			if calls <= retry.DefaultBackoff.Steps {
				return true, nil, apierrors.NewTooManyRequests("cache initializing", 0)
			}
			successfulLists = append(successfulLists, time.Now())
			return true, &corev1.ServiceList{}, nil
		})
		registry := &mockRangeRegistry{item: &api.RangeAllocation{Range: "100-200"}}
		pr, _ := net.ParsePortRange(registry.item.Range)
		r := NewRepair(3*time.Minute, client.CoreV1(), client.EventsV1(), *pr, registry)
		stop := make(chan struct{})
		done := make(chan struct{})
		successes := 0
		var firstSuccess time.Time
		start := time.Now()
		go func() {
			defer close(done)
			r.RunUntil(func() {
				successes++
				firstSuccess = time.Now()
			}, stop)
		}()

		// Allow the post-start hook's one-minute timeout and one periodic repair.
		time.Sleep(time.Minute + r.interval)
		close(stop)
		<-done
		if successes != 1 || registry.updateCalls != 2 {
			t.Fatalf("expected initial and periodic repairs with one callback, got callbacks=%d, writes=%d", successes, registry.updateCalls)
		}
		if firstSuccess.Sub(start) >= time.Minute {
			t.Fatalf("initial repair took %v, exceeding the startup timeout", firstSuccess.Sub(start))
		}
		if calls != retry.DefaultBackoff.Steps+2 || len(successfulLists) != 2 {
			t.Fatalf("expected exhausted retries followed by two successful LISTs, got attempts=%d, successes=%d", calls, len(successfulLists))
		}
		if elapsed := successfulLists[1].Sub(successfulLists[0]); elapsed < r.interval {
			t.Fatalf("periodic repair ran after %v, before the %v interval", elapsed, r.interval)
		}
	})
}

func TestRepairRunUntilCancelsServiceList(t *testing.T) {
	started := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		close(started)
		<-req.Context().Done()
	}))
	defer server.Close()
	defer server.CloseClientConnections()
	client, err := corev1client.NewForConfig(&rest.Config{Host: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	registry := &mockRangeRegistry{item: &api.RangeAllocation{Range: "100-200"}}
	pr, _ := net.ParsePortRange(registry.item.Range)
	r := NewRepair(0, client, fake.NewSimpleClientset().EventsV1(), *pr, registry)
	stop := make(chan struct{})
	done := make(chan struct{})
	successes := 0
	go func() {
		defer close(done)
		r.RunUntil(func() { successes++ }, stop)
	}()
	select {
	case <-started:
	case <-time.After(wait.ForeverTestTimeout):
		close(stop)
		t.Fatal("repair did not issue a Service LIST")
	}
	close(stop)
	select {
	case <-done:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("stopping repair did not cancel the Service LIST")
	}
	if successes != 0 || registry.updateCalled {
		t.Fatalf("cancelled repair reported success or persisted allocations: successes=%d, persisted=%t", successes, registry.updateCalled)
	}
}

func TestRepair(t *testing.T) {
	clearMetrics()
	fakeClient := fake.NewSimpleClientset()
	registry := &mockRangeRegistry{
		item: &api.RangeAllocation{Range: "100-200"},
	}
	pr, _ := net.ParsePortRange(registry.item.Range)
	r := NewRepair(0, fakeClient.CoreV1(), fakeClient.EventsV1(), *pr, registry)

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
	r = NewRepair(0, fakeClient.CoreV1(), fakeClient.EventsV1(), *pr, registry)
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

	r := NewRepair(0, fakeClient.CoreV1(), fakeClient.EventsV1(), *pr, registry)
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
	r := NewRepair(0, fakeClient.CoreV1(), fakeClient.EventsV1(), *pr, registry)
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
