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

package promise

import (
	"context"
	"os"
	"sync"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/util/flowcontrol/counter"
	"k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/promise"
	testeventclock "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/testing/eventclock"
	"k8s.io/klog/v2"
)

func TestMain(m *testing.M) {
	klog.InitFlags(nil)
	os.Exit(m.Run())
}

func TestCountingWriteOnceSet(t *testing.T) {
	oldTime := time.Now()
	cval := &oldTime
	doneCh := make(chan struct{})
	now := time.Now()
	clock, counter := testeventclock.NewFake(now, 0, nil)
	var lock sync.Mutex
	wr := NewCountingWriteOnce(counter, &lock, nil, doneCh, cval)
	gots := make(chan interface{}, 1)
	goGetExpectNotYet(t, clock, counter, wr, gots, "Set")
	aval := &now
	func() {
		lock.Lock()
		defer lock.Unlock()
		if !wr.Set(aval) {
			t.Error("Set() returned false")
		}
	}()
	clock.Run(nil)
	expectGotValue(t, gots, aval)
	goGetAndExpect(t, clock, counter, wr, gots, aval)
	later := time.Now()
	bval := &later
	func() {
		lock.Lock()
		defer lock.Unlock()
		if wr.Set(bval) {
			t.Error("second Set() returned true")
		}
	}()
	goGetAndExpect(t, clock, counter, wr, gots, aval)
	counter.Add(1) // account for unblocking the receive on doneCh
	close(doneCh)
	time.Sleep(time.Second) // give it a chance to misbehave
	goGetAndExpect(t, clock, counter, wr, gots, aval)
}
func TestCountingWriteOnceCancel(t *testing.T) {
	oldTime := time.Now()
	cval := &oldTime
	clock, counter := testeventclock.NewFake(oldTime, 0, nil)
	ctx, cancel := context.WithCancel(context.Background())
	var lock sync.Mutex
	wr := NewCountingWriteOnce(counter, &lock, nil, ctx.Done(), cval)
	gots := make(chan interface{}, 1)
	goGetExpectNotYet(t, clock, counter, wr, gots, "cancel")
	counter.Add(1) // account for unblocking the receive on doneCh
	cancel()
	clock.Run(nil)
	expectGotValue(t, gots, cval)
	goGetAndExpect(t, clock, counter, wr, gots, cval)
	later := time.Now()
	bval := &later
	func() {
		lock.Lock()
		defer lock.Unlock()
		if wr.Set(bval) {
			t.Error("Set() after cancel returned true")
		}
	}()
	goGetAndExpect(t, clock, counter, wr, gots, cval)
}

func TestCountingWriteOnceInitial(t *testing.T) {
	oldTime := time.Now()
	cval := &oldTime
	clock, counter := testeventclock.NewFake(oldTime, 0, nil)
	ctx, cancel := context.WithCancel(context.Background())
	var lock sync.Mutex
	now := time.Now()
	aval := &now
	wr := NewCountingWriteOnce(counter, &lock, aval, ctx.Done(), cval)
	gots := make(chan interface{}, 1)
	goGetAndExpect(t, clock, counter, wr, gots, aval)
	goGetAndExpect(t, clock, counter, wr, gots, aval) // check that a set value stays set
	later := time.Now()
	bval := &later
	func() {
		lock.Lock()
		defer lock.Unlock()
		if wr.Set(bval) {
			t.Error("Set of initialized promise returned true")
		}
	}()
	goGetAndExpect(t, clock, counter, wr, gots, aval)
	counter.Add(1) // account for unblocking receive on doneCh
	cancel()
	time.Sleep(time.Second) // give it a chance to misbehave
	goGetAndExpect(t, clock, counter, wr, gots, aval)
}

func goGetExpectNotYet(t *testing.T, clk *testeventclock.Fake, grc counter.GoRoutineCounter, wr promise.WriteOnce, gots chan interface{}, trigger string) {
	grc.Add(1) // count the following goroutine
	go func() {
		defer grc.Add(-1) // count completion of this goroutine
		gots <- wr.Get()
	}()
	clk.Run(nil)
	select {
	case <-gots:
		t.Errorf("Get returned before %s", trigger)
	case <-time.After(time.Second):
		t.Log("Good: Get did not return yet")
	}

}

func goGetAndExpect(t *testing.T, clk *testeventclock.Fake, grc counter.GoRoutineCounter, wr promise.WriteOnce, gots chan interface{}, expected interface{}) {
	grc.Add(1)
	go func() {
		defer grc.Add(-1)
		gots <- wr.Get()
	}()
	clk.Run(nil)
	expectGotValue(t, gots, expected)
}

func expectGotValue(t *testing.T, gots <-chan interface{}, expected interface{}) {
	select {
	case gotVal := <-gots:
		t.Logf("Got %v", gotVal)
		if gotVal != expected {
			t.Errorf("Get returned %v, expected: %v", gotVal, expected)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Error("Get did not return")
	}
}
