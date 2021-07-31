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
	"sync/atomic"
	"testing"
	"time"
)

func TestWriteOnce(t *testing.T) {
	oldTime := time.Now()
	cval := &oldTime
	ctx := context.Background()
	ctx, cancel := context.WithCancel(ctx)
	wr := NewWriteOnce(nil, ctx.Done(), cval)
	var gots int32
	var got atomic.Value
	go func() {
		got.Store(wr.Get())
		atomic.AddInt32(&gots, 1)
	}()
	time.Sleep(5 * time.Second)
	if atomic.LoadInt32(&gots) != 0 {
		t.Error("Get returned before Set")
	}
	now := time.Now()
	aval := &now
	if !wr.Set(aval) {
		t.Error("Set() returned false")
	}
	time.Sleep(5 * time.Second)
	if atomic.LoadInt32(&gots) != 1 {
		t.Error("Get did not return after Set")
	}
	if got.Load() != aval {
		t.Error("Get did not return what was Set")
	}
	go func() {
		got.Store(wr.Get())
		atomic.AddInt32(&gots, 1)
	}()
	time.Sleep(5 * time.Second)
	if atomic.LoadInt32(&gots) != 2 {
		t.Error("Second Get did not return quickly")
	}
	if got.Load() != aval {
		t.Error("Second Get did not return what was Set")
	}
	later := time.Now()
	bval := &later
	if wr.Set(bval) {
		t.Error("second Set() returned true")
	}
	if wr.Get() != aval {
		t.Error("Get() after second Set returned wrong value")
	}
	cancel()
	time.Sleep(5 * time.Second)
	go func() {
		got.Store(wr.Get())
		atomic.AddInt32(&gots, 1)
	}()
	time.Sleep(5 * time.Second)
	if atomic.LoadInt32(&gots) != 3 {
		t.Error("Third Get did not return quickly")
	}
	if got.Load() != aval {
		t.Error("Third Get did not return what was Set")
	}
}

func TestWriteOnceCancel(t *testing.T) {
	oldTime := time.Now()
	cval := &oldTime
	ctx := context.Background()
	ctx, cancel := context.WithCancel(ctx)
	wr := NewWriteOnce(nil, ctx.Done(), cval)
	var gots int32
	var got atomic.Value
	go func() {
		got.Store(wr.Get())
		atomic.AddInt32(&gots, 1)
	}()
	time.Sleep(5 * time.Second)
	if atomic.LoadInt32(&gots) != 0 {
		t.Error("Get returned before Cancel")
	}
	cancel()
	time.Sleep(5 * time.Second)
	if atomic.LoadInt32(&gots) != 1 {
		t.Error("Get did not return after Cancel")
	}
	if got.Load() != cval {
		t.Error("Get after Cancel did not return cval")
	}
	go func() {
		got.Store(wr.Get())
		atomic.AddInt32(&gots, 1)
	}()
	time.Sleep(5 * time.Second)
	if atomic.LoadInt32(&gots) != 2 {
		t.Error("Second Get did not return quickly")
	}
	if got.Load() != cval {
		t.Error("Second Get did not return cval")
	}
	later := time.Now()
	bval := &later
	if wr.Set(bval) {
		t.Error("second Set() returned true")
	}
	if wr.Get() != cval {
		t.Error("Get() after Cancel then Set returned wrong value")
	}
}

func TestWriteOnceInitial(t *testing.T) {
	oldTime := time.Now()
	cval := &oldTime
	ctx := context.Background()
	ctx, cancel := context.WithCancel(ctx)
	now := time.Now()
	aval := &now
	wr := NewWriteOnce(aval, ctx.Done(), cval)
	if wr.Get() != aval {
		t.Error("First Get of initialized promise did not return initial value")
	}
	later := time.Now()
	bval := &later
	if wr.Set(bval) {
		t.Error("Set of initialized promise returned true")
	}
	if wr.Get() != aval {
		t.Error("Second Get of initialized promise did not return initial value")
	}
	cancel()
	time.Sleep(5 * time.Second)
	if wr.Get() != aval {
		t.Error("Get of initialized promise after cancel did not return initial value")
	}
}
