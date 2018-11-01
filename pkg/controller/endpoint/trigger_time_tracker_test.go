/*
Copyright 2018 The Kubernetes Authors.

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


package endpoint

import (
	"reflect"
	"runtime"
	"testing"
	"time"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

var (
	t0 = time.Date(2018, 01, 01, 0, 0, 0, 0, time.UTC)
	t1 = t0.Add(time.Second)
	t2 = t1.Add(time.Second)
	t3 = t2.Add(time.Second)
	t4 = t3.Add(time.Second)
	t5 = t4.Add(time.Second)

	key = "my_endpoint"
)

func TestObserveBeforeStartListing(t *testing.T) {
	tester := newTester(t)

	tester.observe(key, t0) // P0 - Pod0. To make clear which object was observed.
	tester.observe(key, t1) // P1

	tester.startListing(key)     // P0, P1
	tester.whenListingReturned(key, t0, t1).expect(t0)

	tester.observe(key, t2) // P1
	tester.observe(key, t3) // P0

	tester.startListing(key)     // P0, P1
	tester.whenListingReturned(key, t3, t2).expect(t2)
}

func TestObserveBeforeStartListing_MultipleUpdatesOfTheSameObject(t *testing.T) {
	tester := newTester(t)

	tester.observe(key, t0) // P0
	tester.observe(key, t1) // P1
	tester.observe(key, t2) // P0

	tester.startListing(key)     // P0, P1
	tester.whenListingReturned(key, t2, t1).expect(t0)

	tester.observe(key, t3) // P1
	tester.observe(key, t4) // P0
	tester.observe(key, t5) // P1

	tester.startListing(key)     // P0, P1
	tester.whenListingReturned(key, t4, t5).expect(t3)
}

func TestEventsDuringListening(t *testing.T) {
	tester := newTester(t)

	tester.observe(key, t0) // P0

	tester.startListing(key)
	tester.observe(key, t1) // P1
	tester.observe(key, t2) // P0
	//                              P0, P1
	tester.whenListingReturned(key, t2, t1).expect(t0)

	tester.observe(key, t3) // P1
	tester.observe(key, t4) // P0

	tester.startListing(key)     // P0, P1
	tester.whenListingReturned(key, t4, t3).expect(t3)
}

func TestEventsDuringListening_FresherEvent(t *testing.T) {
	tester := newTester(t)

	tester.observe(key, t0) // P0
	tester.observe(key, t1) // P1

	tester.startListing(key)

	// This event arrived directly after listing ended, it's possible as the mutex is unlocked outside
	// of the list method - mutex.Lock(); List(); (event observed) mutex.Unlock();
	tester.observe(key, t2) // P2
	//                              P0, P1
	tester.whenListingReturned(key, t0, t1).expect(t0)

	tester.observe(key, t3) // P2

	tester.startListing(key)
	tester.observe(key, t4) // P2
	//                              P0, P1  P2
	tester.whenListingReturned(key, t0, t1, t4).expect(t2) // t2 wasn't lost, it was marked as dirty
	                                                       // and processed later.
}

func TestEventsDuringListening_MultipleEvents(t *testing.T) {
	tester := newTester(t)

	tester.observe(key, t0) // P0
	tester.observe(key, t1) // P1

	tester.startListing(key)

	tester.observe(key, t2) // P0
	tester.observe(key, t3) // P2
	//                              P0, P1
	tester.whenListingReturned(key, t2, t1).expect(t0)

	tester.observe(key, t4) // P2

	tester.startListing(key)     // P0, P1  P2
	tester.whenListingReturned(key, t0, t1, t4).expect(t3)
}

func TestEventsAfterListingInFirstBatch(t *testing.T) {
	tester := newTester(t)

	tester.startListing(key)
	// P0, P1 already changed and will be returned in list, but no events yet.
	//                              P0, P1
	tester.whenListingReturned(key, t0, t1).expect(t0)

	// Delayed events
	tester.observe(key, t0) // P0
	tester.observe(key, t1) // P1

	tester.observe(key, t2) // P1

	tester.startListing(key)     // P0, P1
	tester.whenListingReturned(key, t0, t2).expect(t2) // t0 and t1 change was already processed
}

func TestEventsAfterListingInBothBatches(t *testing.T) {
	tester := newTester(t)

	tester.startListing(key)
	// P0, P1 already changed and will be returned in list, but no events yet.
	//                              P0, P1
	tester.whenListingReturned(key, t0, t1).expect(t0)

	// Delayed events
	tester.observe(key, t0) // P0
	tester.observe(key, t1) // P1

	tester.startListing(key)     // P0, P1
	tester.whenListingReturned(key, t3, t2).expect(t2) // t0 and t1 change was already processed

	tester.observe(key, t2) // P1
	tester.observe(key, t3) // P0

	tester.startListing(key)     // P0, P1
	tester.whenListingReturned(key, t3, t2).expectNil() // t2 and t3 was already processed.
}

func TestEventsAfterListing_SameObjectUpdatedMultipleTimes(t *testing.T) {
	tester := newTester(t)

	tester.startListing(key)
	// P0, P1 already changed and will be returned in list, but no events yet.
	//                              P0, P1
	tester.whenListingReturned(key, t2, t1).expect(t1)

	tester.assertCounterValue(LastChangeTriggerTimeMiscalculated, 0)
	// Delayed events
	tester.observe(key, t0) // P0 - here we realize that we exported wrong time.
	// Error counter was incremented.
	tester.assertCounterValue(LastChangeTriggerTimeMiscalculated, 1)

	tester.observe(key, t1) // P1
	tester.observe(key, t2) // P0
	tester.observe(key, t3) // P1

	tester.startListing(key)     // P0, P1
	tester.whenListingReturned(key, t2, t3).expect(t3)
}

func TestMultipleKeysUpdatedSimultaneously(t *testing.T) {
	key2 := "my-endpoints-2"

	tester := newTester(t)

	tester.observe(key, t0) // E0_P0
	tester.observe(key, t1) // E0_P1

	tester.observe(key2, t1) // E1_P0

	tester.startListing(key)     // P0, P1
	tester.whenListingReturned(key, t0, t1).expect(t0)

	tester.startListing(key2)
	tester.observe(key2, t2) // E1_P1
	//                               P0  P1
	tester.whenListingReturned(key2, t1, t2).expect(t1)

	tester.observe(key, t5) // E0_P0

	tester.startListing(key2)     // P0, P1
	tester.whenListingReturned(key2, t4, t3).expect(t3)

	tester.startListing(key)     // P0, P1
	tester.whenListingReturned(key, t5, t1).expect(t5)
}

func TestNoEventsListReturnedNothing(t *testing.T) {
	tester := newTester(t)

	tester.startListing(key)
	// This shouldn't happen, but make sure it won't crash.
	tester.whenListingReturned(key).expectNil()
}

func TestListWithoutUpdate(t *testing.T) {
	tester := newTester(t)

	tester.observe(key, t0) // P0
	tester.observe(key, t1) // P1
	tester.observe(key, t2) // P1

	tester.startListing(key)
	tester.whenListingReturned(key, t0, t1, t2).expect(t0)

	// No observe events, but list again. It can happen e.g. when labels where updated.
	tester.startListing(key)
	tester.whenListingReturned(key, t0, t1, t2).expectNil() // Nil returned, nothing will be exported.
}



// ------- Test Utils -------

type tester struct {
	*triggerTimeTracker
	t *testing.T
}

func newTester(t *testing.T) *tester {
	return &tester { newTriggerTimeTracker(), t}
}

func (this *tester) whenListingReturned(key string, val ...time.Time) subject {
	return subject { this.stopListingAndReset(key, val), this.t }
}

type subject struct {
	got *time.Time
	t *testing.T
}


func (s subject) expect(val time.Time) {
	s.expectPointer(&val)
}

func (s subject) expectNil() {
	s.expectPointer(nil)
}

func (s subject) expectPointer(val *time.Time) {
	if !reflect.DeepEqual(s.got, val) {
		_, fn, line, _ := runtime.Caller(2)
		s.t.Errorf("Wrong trigger time in %s:%d expected %s, got %s", fn, line, val, s.got)
	}
}

func (this *tester) assertCounterValue(c prometheus.Counter, wanted float64) {
	pb := &dto.Metric{}
	c.Write(pb)
	got := pb.GetCounter().GetValue()
	if got != wanted {
		this.t.Errorf("Wrong counter (%s) value, expected %f, got %f", c.Desc(), wanted, got)
	}
}
