/*
Copyright 2014 Google Inc. All rights reserved.

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

package apiserver

import (
	"testing"
	"time"
)

func TestOperation(t *testing.T) {
	ops := NewOperations()

	c := make(chan interface{})
	op := ops.NewOperation(c)
	go func() {
		time.Sleep(500 * time.Millisecond)
		c <- "All done"
	}()

	if op.expired(time.Now().Add(-time.Minute)) {
		t.Errorf("Expired before finished: %#v", op)
	}
	ops.expire(time.Minute)
	if tmp := ops.Get(op.ID); tmp == nil {
		t.Errorf("expire incorrectly removed the operation %#v", ops)
	}

	op.WaitFor(10 * time.Millisecond)
	if _, completed := op.Describe(); completed {
		t.Errorf("Unexpectedly fast completion")
	}

	op.WaitFor(time.Second)
	if _, completed := op.Describe(); !completed {
		t.Errorf("Unexpectedly slow completion")
	}

	time.Sleep(100 * time.Millisecond)

	if op.expired(time.Now().Add(-time.Second)) {
		t.Errorf("Should not be expired: %#v", op)
	}
	if !op.expired(time.Now().Add(-80 * time.Millisecond)) {
		t.Errorf("Should be expired: %#v", op)
	}

	ops.expire(80 * time.Millisecond)
	if tmp := ops.Get(op.ID); tmp != nil {
		t.Errorf("expire failed to remove the operation %#v", ops)
	}

	if op.result.(string) != "All done" {
		t.Errorf("Got unexpected result: %#v", op.result)
	}
}
