// Copyright 2015 The etcd Authors
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
	ch1 := wt.Wait(1)
	wt.Trigger(2)
	select {
	case <-ch1:
	default:
		t.Fatalf("cannot receive from ch as expected")
	}

	ch2 := wt.Wait(4)
	wt.Trigger(3)
	select {
	case <-ch2:
		t.Fatalf("unexpected to receive from ch2")
	default:
	}
	wt.Trigger(4)
	select {
	case <-ch2:
	default:
		t.Fatalf("cannot receive from ch2 as expected")
	}

	select {
	// wait on a triggered deadline
	case <-wt.Wait(4):
	default:
		t.Fatalf("unexpected blocking when wait on triggered deadline")
	}
}

func TestWaitTestStress(t *testing.T) {
	chs := make([]<-chan struct{}, 0)
	wt := NewTimeList()
	for i := 0; i < 10000; i++ {
		chs = append(chs, wt.Wait(uint64(i)))
	}
	wt.Trigger(10000 + 1)

	for _, ch := range chs {
		select {
		case <-ch:
		case <-time.After(time.Second):
			t.Fatalf("cannot receive from ch as expected")
		}
	}
}

func BenchmarkWaitTime(b *testing.B) {
	wt := NewTimeList()
	for i := 0; i < b.N; i++ {
		wt.Wait(1)
	}
}

func BenchmarkTriggerAnd10KWaitTime(b *testing.B) {
	for i := 0; i < b.N; i++ {
		wt := NewTimeList()
		for j := 0; j < 10000; j++ {
			wt.Wait(uint64(j))
		}
		wt.Trigger(10000 + 1)
	}
}
