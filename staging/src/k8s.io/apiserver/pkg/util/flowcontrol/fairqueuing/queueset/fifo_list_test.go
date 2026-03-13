/*
Copyright 2021 The Kubernetes Authors.

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

package queueset

import (
	"math/rand"
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	fcrequest "k8s.io/apiserver/pkg/util/flowcontrol/request"
)

func TestFIFOWithEnqueueDequeueSingleRequest(t *testing.T) {
	req := &request{}

	list := newRequestFIFO()
	list.Enqueue(req)

	reqGot, ok := list.Dequeue()
	if !ok {
		t.Errorf("Expected true, but got: %t", ok)
	}
	if req != reqGot {
		t.Errorf("Expected dequued request: (%p), but got: (%p)", req, reqGot)
	}
	if list.Length() != 0 {
		t.Errorf("Expected length: %d, but got: %d)", 0, list.Length())
	}
}

func TestFIFOWithEnqueueDequeueMultipleRequests(t *testing.T) {
	arrival := []*request{{}, {}, {}, {}, {}, {}}

	list := newRequestFIFO()
	for i := range arrival {
		list.Enqueue(arrival[i])
	}

	dequeued := make([]*request, 0)
	for list.Length() > 0 {
		req, _ := list.Dequeue()
		dequeued = append(dequeued, req)
	}

	verifyOrder(t, arrival, dequeued)
}

func TestFIFOWithEnqueueDequeueSomeRequestsRemainInQueue(t *testing.T) {
	list := newRequestFIFO()

	arrival := []*request{{}, {}, {}, {}, {}, {}}
	half := len(arrival) / 2
	for i := range arrival {
		list.Enqueue(arrival[i])
	}

	dequeued := make([]*request, 0)
	for i := 0; i < half; i++ {
		req, _ := list.Dequeue()
		dequeued = append(dequeued, req)
	}

	verifyOrder(t, arrival[:half], dequeued)
}

func TestFIFOWithRemoveMultipleRequestsInArrivalOrder(t *testing.T) {
	list := newRequestFIFO()

	arrival := []*request{{}, {}, {}, {}, {}, {}}
	removeFn := make([]removeFromFIFOFunc, 0)
	for i := range arrival {
		removeFn = append(removeFn, list.Enqueue(arrival[i]))
	}

	expected := append([]*request{}, arrival...)
	for idx, f := range removeFn {
		if a := f(); a != arrival[idx] {
			t.Errorf("Removal %d returned %v instead of expected pointer", idx, a)
		}
		if a := f(); a != nil {
			t.Errorf("Redundant removal %d returned %v instead of expected nil", idx, a)
		}
		expected = expected[1:]
		actual := walkAll(list)
		verifyOrder(t, expected, actual)
	}

	if list.Length() != 0 {
		t.Errorf("Expected length: %d, but got: %d)", 0, list.Length())
	}
}

func TestFIFORemoveFromFIFOFunc(t *testing.T) {
	list := newRequestFIFO()
	reqWant := &request{}
	removeFn := list.Enqueue(reqWant)

	reqGot := removeFn()
	if reqWant != reqGot {
		t.Errorf("Expected request identity: %p, but got: %p)", reqWant, reqGot)
	}

	if got := removeFn(); got != nil {
		t.Errorf("Expected a nil request, but got: %v)", got)
	}
}

func TestFIFOWithRemoveMultipleRequestsInRandomOrder(t *testing.T) {
	list := newRequestFIFO()

	arrival := []*request{{}, {}, {}, {}, {}, {}}
	removeFn := make([]removeFromFIFOFunc, 0)
	for i := range arrival {
		removeFn = append(removeFn, list.Enqueue(arrival[i]))
	}

	expected := append([]*request{}, arrival...)
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for range arrival {
		idx := r.Intn(len(expected))
		t.Logf("Removing random index %d", idx)
		if e, a := expected[idx], removeFn[idx](); e != a {
			t.Errorf("Removal of %d returned %v instead of expected pointer %v", idx, a, e)
		}
		if e, a := (*request)(nil), removeFn[idx](); e != a {
			t.Errorf("Redundant removal of %d returned %v instead of expected nil pointer", idx, a)
		}
		expected = append(expected[:idx], expected[idx+1:]...)
		actual := walkAll(list)
		verifyOrder(t, expected, actual)
		removeFn = append(removeFn[:idx], removeFn[idx+1:]...)
	}
	if list.Length() != 0 {
		t.Errorf("Expected length: %d, but got: %d)", 0, list.Length())
	}
}

func TestFIFOWithRemoveIsIdempotent(t *testing.T) {
	list := newRequestFIFO()

	arrival := []*request{{}, {}, {}, {}}
	removeFn := make([]removeFromFIFOFunc, 0)
	for i := range arrival {
		removeFn = append(removeFn, list.Enqueue(arrival[i]))
	}

	// pick one request to be removed at random
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	randomIndex := r.Intn(len(removeFn))
	t.Logf("Random remove index: %d", randomIndex)

	// remove the request from the fifo twice, we expect it to be idempotent
	removeFn[randomIndex]()
	removeFn[randomIndex]()

	lengthExpected := len(arrival) - 1
	if lengthExpected != list.Length() {
		t.Errorf("Expected length: %d, but got: %d)", lengthExpected, list.Length())
	}

	orderExpected := append(arrival[0:randomIndex], arrival[randomIndex+1:]...)
	remainingRequests := walkAll(list)
	verifyOrder(t, orderExpected, remainingRequests)
}

func TestFIFOQueueWorkEstimate(t *testing.T) {
	qs := &queueSet{estimatedServiceDuration: time.Second}
	list := newRequestFIFO()

	update := func(we *queueSum, req *request, multiplier int) {
		we.InitialSeatsSum += multiplier * req.InitialSeats()
		we.MaxSeatsSum += multiplier * req.MaxSeats()
		we.TotalWorkSum += fcrequest.SeatSeconds(multiplier) * req.totalWork()
	}

	assert := func(t *testing.T, want, got *queueSum) {
		if !reflect.DeepEqual(want, got) {
			t.Errorf("Expected queue work estimate to match, diff: %s", cmp.Diff(want, got))
		}
	}

	newRequest := func(initialSeats, finalSeats uint64, additionalLatency time.Duration) *request {
		return &request{workEstimate: qs.completeWorkEstimate(&fcrequest.WorkEstimate{
			InitialSeats:      initialSeats,
			FinalSeats:        finalSeats,
			AdditionalLatency: additionalLatency,
		})}
	}
	arrival := []*request{
		newRequest(1, 3, time.Second),
		newRequest(2, 2, 2*time.Second),
		newRequest(3, 1, 3*time.Second),
	}
	removeFn := make([]removeFromFIFOFunc, 0)

	queueSumExpected := queueSum{}
	for i := range arrival {
		req := arrival[i]
		removeFn = append(removeFn, list.Enqueue(req))
		update(&queueSumExpected, req, 1)

		workEstimateGot := list.QueueSum()
		assert(t, &queueSumExpected, &workEstimateGot)
	}

	// NOTE: the test expects the request and the remove func to be at the same index
	for i := range removeFn {
		req := arrival[i]
		removeFn[i]()

		update(&queueSumExpected, req, -1)

		workEstimateGot := list.QueueSum()
		assert(t, &queueSumExpected, &workEstimateGot)

		// check idempotency
		removeFn[i]()

		workEstimateGot = list.QueueSum()
		assert(t, &queueSumExpected, &workEstimateGot)
	}

	// Check second type of idempotency: Dequeue + removeFn.
	for i := range arrival {
		req := arrival[i]
		removeFn[i] = list.Enqueue(req)

		update(&queueSumExpected, req, 1)
	}

	for i := range arrival {
		// we expect Dequeue to pop the oldest request that should
		// have the lowest index as well.
		req := arrival[i]

		if _, ok := list.Dequeue(); !ok {
			t.Errorf("Unexpected failed dequeue: %d", i)
		}

		update(&queueSumExpected, req, -1)

		queueSumGot := list.QueueSum()
		assert(t, &queueSumExpected, &queueSumGot)

		removeFn[i]()

		queueSumGot = list.QueueSum()
		assert(t, &queueSumExpected, &queueSumGot)
	}
}

func TestFIFOWithWalk(t *testing.T) {
	list := newRequestFIFO()

	arrival := []*request{{}, {}, {}, {}, {}, {}}
	for i := range arrival {
		list.Enqueue(arrival[i])
	}

	visited := walkAll(list)

	verifyOrder(t, arrival, visited)
}

func verifyOrder(t *testing.T, expected, actual []*request) {
	if len(expected) != len(actual) {
		t.Fatalf("Expected slice length: %d, but got: %d", len(expected), len(actual))
	}
	for i := range expected {
		if expected[i] != actual[i] {
			t.Errorf("Dequeue order mismatch, expected request: (%p), but got: (%p)", expected[i], actual[i])
		}
	}
}

func walkAll(l fifo) []*request {
	visited := make([]*request, 0)
	l.Walk(func(req *request) bool {
		visited = append(visited, req)
		return true
	})

	return visited
}
