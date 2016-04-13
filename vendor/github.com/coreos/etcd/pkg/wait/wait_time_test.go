// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package wait

import (
	"testing"
	"time"
)

func TestWaitTime(t *testing.T) {
	wt := NewTimeList()
	ch1 := wt.Wait(time.Now())
	t1 := time.Now()
	wt.Trigger(t1)
	select {
	case <-ch1:
	case <-time.After(100 * time.Millisecond):
		t.Fatalf("cannot receive from ch as expected")
	}

	ch2 := wt.Wait(time.Now())
	t2 := time.Now()
	wt.Trigger(t1)
	select {
	case <-ch2:
		t.Fatalf("unexpected to receive from ch")
	case <-time.After(10 * time.Millisecond):
	}
	wt.Trigger(t2)
	select {
	case <-ch2:
	case <-time.After(10 * time.Millisecond):
		t.Fatalf("cannot receive from ch as expected")
	}
}

func TestWaitTestStress(t *testing.T) {
	chs := make([]<-chan struct{}, 0)
	wt := NewTimeList()
	for i := 0; i < 10000; i++ {
		chs = append(chs, wt.Wait(time.Now()))
		// sleep one nanosecond before waiting on the next event
		time.Sleep(time.Nanosecond)
	}
	wt.Trigger(time.Now())

	for _, ch := range chs {
		select {
		case <-ch:
		case <-time.After(time.Second):
			t.Fatalf("cannot receive from ch as expected")
		}
	}
}

func BenchmarkWaitTime(b *testing.B) {
	t := time.Now()
	wt := NewTimeList()
	for i := 0; i < b.N; i++ {
		wt.Wait(t)
	}
}

func BenchmarkTriggerAnd10KWaitTime(b *testing.B) {
	for i := 0; i < b.N; i++ {
		t := time.Now()
		wt := NewTimeList()
		for j := 0; j < 10000; j++ {
			wt.Wait(t)
		}
		wt.Trigger(time.Now())
	}
}
