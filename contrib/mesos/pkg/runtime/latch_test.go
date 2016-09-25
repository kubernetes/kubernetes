/*
Copyright 2015 The Kubernetes Authors.

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

package runtime

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func Test_LatchAcquireBasic(t *testing.T) {
	var x Latch
	if !x.Acquire() {
		t.Fatalf("expected first acquire to succeed")
	}
	if x.Acquire() {
		t.Fatalf("expected second acquire to fail")
	}
	if x.Acquire() {
		t.Fatalf("expected third acquire to fail")
	}
}

func Test_LatchAcquireConcurrent(t *testing.T) {
	var x Latch
	const NUM = 10
	ch := make(chan struct{})
	var success int32
	var wg sync.WaitGroup
	wg.Add(NUM)
	for i := 0; i < NUM; i++ {
		go func() {
			defer wg.Done()
			<-ch
			if x.Acquire() {
				atomic.AddInt32(&success, 1)
			}
		}()
	}
	time.Sleep(200 * time.Millisecond)
	close(ch)
	wg.Wait()
	if success != 1 {
		t.Fatalf("expected single acquire to succeed instead of %d", success)
	}
}
