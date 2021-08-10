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
	"testing"
	"time"

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

	dequeued := make([]*request, 0)
	for _, f := range removeFn {
		dequeued = append(dequeued, f())
	}

	if list.Length() != 0 {
		t.Errorf("Expected length: %d, but got: %d)", 0, list.Length())
	}

	verifyOrder(t, arrival, dequeued)
}

func TestFIFOWithRemoveMultipleRequestsInRandomOrder(t *testing.T) {
	list := newRequestFIFO()

	arrival := []*request{{}, {}, {}, {}, {}, {}}
	removeFn := make([]removeFromFIFOFunc, 0)
	for i := range arrival {
		removeFn = append(removeFn, list.Enqueue(arrival[i]))
	}

	dequeued := make([]*request, 0)
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	randomIndices := r.Perm(len(removeFn))
	t.Logf("Random remove order: %v", randomIndices)
	for i := range randomIndices {
		dequeued = append(dequeued, removeFn[i]())
	}

	if list.Length() != 0 {
		t.Errorf("Expected length: %d, but got: %d)", 0, list.Length())
	}

	verifyOrder(t, arrival, dequeued)
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

func TestFIFOSeatsSum(t *testing.T) {
	list := newRequestFIFO()

	newRequest := func(width uint) *request {
		return &request{width: fcrequest.Width{Seats: width}}
	}
	arrival := []*request{newRequest(1), newRequest(2), newRequest(3)}
	removeFn := make([]removeFromFIFOFunc, 0)

	seatsSum := 0
	for i := range arrival {
		removeFn = append(removeFn, list.Enqueue(arrival[i]))

		seatsSum += i + 1
		if list.SeatsSum() != seatsSum {
			t.Errorf("Expected seatsSum: %d, but got: %d", seatsSum, list.SeatsSum())
		}
	}

	for i := range removeFn {
		removeFn[i]()

		seatsSum -= i + 1
		if list.SeatsSum() != seatsSum {
			t.Errorf("Expected seatsSum: %d, but got: %d", seatsSum, list.SeatsSum())
		}

		// check idempotency
		removeFn[i]()
		if list.SeatsSum() != seatsSum {
			t.Errorf("Expected seatsSum: %d, but got: %d", seatsSum, list.SeatsSum())
		}
	}

	// Check second type of idempotency: Dequeue + removeFn.
	for i := range arrival {
		removeFn[i] = list.Enqueue(arrival[i])
		seatsSum += i + 1
	}

	for i := range arrival {
		if _, ok := list.Dequeue(); !ok {
			t.Errorf("Unexpected failed dequeue: %d", i)
		}
		seatsSum -= i + 1
		if list.SeatsSum() != seatsSum {
			t.Errorf("Expected seatsSum: %d, but got: %d", seatsSum, list.SeatsSum())
		}

		removeFn[i]()
		if list.SeatsSum() != seatsSum {
			t.Errorf("Expected seatsSum: %d, but got: %d", seatsSum, list.SeatsSum())
		}
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
