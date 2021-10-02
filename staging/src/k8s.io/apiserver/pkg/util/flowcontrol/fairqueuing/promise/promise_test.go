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
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
)

func TestWriteOnceSet(t *testing.T) {
	oldTime := time.Now()
	cval := &oldTime
	ctx, cancel := context.WithCancel(context.Background())
	wr := NewWriteOnce(nil, ctx.Done(), cval)
	gots := make(chan interface{})
	goGetExpectNotYet(t, wr, gots, "Set")
	now := time.Now()
	aval := &now
	if !wr.Set(aval) {
		t.Error("Set() returned false")
	}
	expectGotValue(t, gots, aval)
	goGetAndExpect(t, wr, gots, aval)
	later := time.Now()
	bval := &later
	if wr.Set(bval) {
		t.Error("second Set() returned true")
	}
	goGetAndExpect(t, wr, gots, aval)
	cancel()
	time.Sleep(time.Second) // give it a chance to misbehave
	goGetAndExpect(t, wr, gots, aval)
}

func TestWriteOnceCancel(t *testing.T) {
	oldTime := time.Now()
	cval := &oldTime
	ctx, cancel := context.WithCancel(context.Background())
	wr := NewWriteOnce(nil, ctx.Done(), cval)
	gots := make(chan interface{})
	goGetExpectNotYet(t, wr, gots, "cancel")
	cancel()
	expectGotValue(t, gots, cval)
	goGetAndExpect(t, wr, gots, cval)
	later := time.Now()
	bval := &later
	if wr.Set(bval) {
		t.Error("Set() after cancel returned true")
	}
	goGetAndExpect(t, wr, gots, cval)
}

func TestWriteOnceInitial(t *testing.T) {
	oldTime := time.Now()
	cval := &oldTime
	ctx, cancel := context.WithCancel(context.Background())
	now := time.Now()
	aval := &now
	wr := NewWriteOnce(aval, ctx.Done(), cval)
	gots := make(chan interface{})
	goGetAndExpect(t, wr, gots, aval)
	later := time.Now()
	bval := &later
	if wr.Set(bval) {
		t.Error("Set of initialized promise returned true")
	}
	goGetAndExpect(t, wr, gots, aval)
	cancel()
	time.Sleep(time.Second) // give it a chance to misbehave
	goGetAndExpect(t, wr, gots, aval)
}

func goGetExpectNotYet(t *testing.T, wr WriteOnce, gots chan interface{}, trigger string) {
	go func() {
		gots <- wr.Get()
	}()
	select {
	case <-gots:
		t.Errorf("Get returned before %s", trigger)
	case <-time.After(time.Second):
		t.Log("Good: Get did not return yet")
	}
}

func goGetAndExpect(t *testing.T, wr WriteOnce, gots chan interface{}, expected interface{}) {
	go func() {
		gots <- wr.Get()
	}()
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
