/*
Copyright 2019 The Kubernetes Authors.

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

package fairqueuing

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/waitgroup"
)

// adapted from https://github.com/tadglines/wfq/blob/master/wfq_test.go

type flowDesc struct {
	// In
	ftotal float64 // Total units in flow
	imin   float64 // Min testRequest servicetime
	imax   float64 // Max testRequest servicetime

	// Out
	idealPercent  float64
	actualPercent float64
}

func genFlow(fq *queueSetImpl, desc *flowDesc, key int) {
	for i, t := 1, float64(0); t < desc.ftotal; i++ {
		it := new(Request)
		it.QueueIdx = key
		it.Queue = fq.queues[key]
		if desc.imin == desc.imax {
			it.servicetime = desc.imax
		} else {
			it.servicetime = desc.imin + rand.Float64()*(desc.imax-desc.imin)
		}
		if float64(t)+it.servicetime > desc.ftotal {
			it.servicetime = desc.ftotal - float64(t)
		}
		t += it.servicetime
		it.seq = i
		// new packet
		fq.Enqueue(it)
	}
}

func consumeQueue(t *testing.T, fq *queueSetImpl, descs []flowDesc) (float64, error) {
	active := make(map[int]bool)
	var total float64
	acnt := make(map[int]float64)
	cnt := make(map[int]float64)
	seqs := make(map[int]int)

	for it, ok := fq.Dequeue(); ok; it, ok = fq.Dequeue() {
		// callback to update virtualtime w/ correct service time for request
		fq.finishRequest(it)

		seq := seqs[it.QueueIdx]
		if seq+1 != it.seq {
			return 0, fmt.Errorf("testRequest for flow %d came out of queue out-of-order: expected %d, got %d", it.QueueIdx, seq+1, it.seq)
		}
		seqs[it.QueueIdx] = it.seq

		// set the flow this item is a part of to active
		active[it.QueueIdx] = true

		cnt[it.QueueIdx] += it.servicetime

		// if # of active flows is equal to the # of total flows, add to total
		if len(active) == len(descs) {
			acnt[it.QueueIdx] += it.servicetime
			total += it.servicetime
		}

		// if all items have been processed from the flow, remove it from active
		if cnt[it.QueueIdx] == descs[it.QueueIdx].ftotal {
			delete(active, it.QueueIdx)
		}
	}

	if total == 0 {
		t.Fatalf("expected 'total' to be nonzero")
	}

	var variance float64
	for key := 0; key < len(descs); key++ {
		// flows in this test have same expected # of requests
		// idealPercent = total-all-active/len(flows) / total-all-active
		// "how many bytes/requests you expect for this flow - all-active"
		descs[key].idealPercent = float64(100) / float64(len(descs))

		// actualPercent = requests-for-this-flow-all-active / total-reqs
		// "how many bytes/requests you got for this flow - all-active"
		descs[key].actualPercent = (acnt[key] / total) * 100

		x := descs[key].idealPercent - descs[key].actualPercent
		x *= x
		variance += x
	}
	variance /= float64(len(descs))

	stdDev := math.Sqrt(variance)
	return stdDev, nil
}

func TestSingleFlow(t *testing.T) {
	var flows = []flowDesc{
		{100, 1, 1, 0, 0},
	}
	flowStdDevTest(t, flows, 0)
}

func TestUniformMultiFlow(t *testing.T) {
	var flows = []flowDesc{
		{100, 1, 1, 0, 0},
		{100, 1, 1, 0, 0},
		{100, 1, 1, 0, 0},
		{100, 1, 1, 0, 0},
		{100, 1, 1, 0, 0},
		{100, 1, 1, 0, 0},
		{100, 1, 1, 0, 0},
		{100, 1, 1, 0, 0},
		{100, 1, 1, 0, 0},
		{100, 1, 1, 0, 0},
	}
	// .35 was expectedStdDev used in
	// https://github.com/tadglines/wfq/blob/master/wfq_test.go
	flowStdDevTest(t, flows, .041)
}

func TestOneBurstingFlow(t *testing.T) {

	var flows = []flowDesc{
		{1000, 1, 1, 0, 0},
		{100, 1, 1, 0, 0},
	}
	flowStdDevTest(t, flows, 0)
}

func flowStdDevTest(t *testing.T, flows []flowDesc, expectedStdDev float64) {
	runtime.GOMAXPROCS(runtime.NumCPU())
	// queues := initQueues(len(flows))

	// a fake clock that returns the current time is used for enqueing which
	// returns the same time (now)
	// this simulates all queued requests coming at the same time
	now := time.Now()
	fc := clock.NewFakeClock(now)

	// fqqueues := make([]*Queue, len(queues), len(queues))
	// for i := range queues {
	// 	fqqueues[i] = queues[i]
	// }
	fq := newQueueSetImpl("", 20000, len(flows), 20000, 5*time.Second, fc, waitgroup.NoWaitGroup())
	for n := 0; n < len(flows); n++ {
		genFlow(fq, &flows[n], n)
	}

	// prior to dequeing, we switch to an interval clock which will simulate
	// each dequeue happening at a fixed interval of time
	ic := &clock.IntervalClock{
		Time:     now,
		Duration: time.Millisecond,
	}
	fq.clk = ic

	stdDev, err := consumeQueue(t, fq, flows)

	if err != nil {
		t.Fatal(err.Error())
	}

	if stdDev > expectedStdDev {
		for k, d := range flows {
			t.Logf("For flow %d: Expected %v%%, got %v%%", k, d.idealPercent, d.actualPercent)
		}
		t.Fatalf("StdDev was expected to be < %f but got %v", expectedStdDev, stdDev)
	}
}
