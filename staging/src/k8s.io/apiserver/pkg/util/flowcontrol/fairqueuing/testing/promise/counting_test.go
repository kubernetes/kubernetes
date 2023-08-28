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
	doneCtx := NewQueueWaitTimeWithContext(context.Background(), time.Second)
	now := time.Now()
	clock, counter := testeventclock.NewFake(now, 0, nil)
	var lock sync.Mutex
	wr := NewCountingWriteOnce(clock, counter, &lock, nil, doneCtx, "canceled")

	beforeCanceled := now.Add(time.Second).Add(-time.Millisecond)
	clock.Run(&beforeCanceled)

	gots := make(chan interface{}, 1)
	counter.Add(1)
	go func() {
		defer counter.Add(-1)
		gots <- wr.Get()
	}()
	select {
	case <-gots:
		t.Errorf("Get returned before Set")
	case <-time.After(time.Second):
		t.Log("Good: Get did not return yet")
	}

	func() {
		lock.Lock()
		defer lock.Unlock()
		if !wr.Set("execute") {
			t.Error("Set() returned false")
		}
	}()
	clock.Run(nil)

	expectGotValue(t, gots, "execute")
	goGetAndExpect(t, clock, counter, wr, gots, "execute")
	func() {
		lock.Lock()
		defer lock.Unlock()
		if wr.Set("second time") {
			t.Error("second Set() returned true")
		}
	}()
	goGetAndExpect(t, clock, counter, wr, gots, "execute")
}

func TestCountingWriteOnceCancel(t *testing.T) {
	doneCtx := NewQueueWaitTimeWithContext(context.Background(), time.Second)
	now := time.Now()
	clock, counter := testeventclock.NewFake(now, 0, nil)
	var lock sync.Mutex
	wr := NewCountingWriteOnce(clock, counter, &lock, nil, doneCtx, "canceled")

	gots := make(chan interface{}, 1)
	counter.Add(1)
	go func() {
		defer counter.Add(-1)
		gots <- wr.Get()
	}()
	select {
	case <-gots:
		t.Errorf("Get returned before Set")
	default:
		t.Log("Good: Get did not return yet")
	}

	clock.Run(nil)

	expectGotValue(t, gots, "canceled")
	goGetAndExpect(t, clock, counter, wr, gots, "canceled")
	func() {
		lock.Lock()
		defer lock.Unlock()
		if wr.Set("should fail") {
			t.Error("Set() after cancel returned true")
		}
	}()
	goGetAndExpect(t, clock, counter, wr, gots, "canceled")
}

func TestCountingWriteOnceInitial(t *testing.T) {
	doneCtx := NewQueueWaitTimeWithContext(context.Background(), time.Second)
	now := time.Now()
	clock, counter := testeventclock.NewFake(now, 0, nil)
	var lock sync.Mutex
	wr := NewCountingWriteOnce(clock, counter, &lock, "execute", doneCtx, "canceled")

	if got := wr.Get(); got != "execute" {
		t.Errorf("Expected initial value to be set")
	}

	clock.Run(nil)

	gots := make(chan interface{}, 1)
	counter.Add(1)
	go func() {
		defer counter.Add(-1)
		gots <- wr.Get()
	}()
	expectGotValue(t, gots, "execute")

	goGetAndExpect(t, clock, counter, wr, gots, "execute")
	goGetAndExpect(t, clock, counter, wr, gots, "execute") // check that a set value stays set
	func() {
		lock.Lock()
		defer lock.Unlock()
		if wr.Set("should fail") {
			t.Error("Set of initialized promise returned true")
		}
	}()
	goGetAndExpect(t, clock, counter, wr, gots, "execute")
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
