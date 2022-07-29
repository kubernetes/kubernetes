/*
Copyright 2022 The Kubernetes Authors.

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

package cacher

import (
	"testing"
)

func Test_newReady(t *testing.T) {
	errCh := make(chan error, 10)
	ready := newReady()
	ready.set(false)
	// create 10 goroutines waiting for ready
	for i := 0; i < 10; i++ {
		go func() {
			errCh <- ready.wait()
		}()
	}
	ready.set(true)
	for i := 0; i < 10; i++ {
		if err := <-errCh; err != nil {
			t.Errorf("unexpected error on channel %d", i)
		}
	}
}

func Test_newReadyStop(t *testing.T) {
	errCh := make(chan error, 10)
	ready := newReady()
	ready.set(false)
	// create 10 goroutines waiting for ready and stop
	for i := 0; i < 10; i++ {
		go func() {
			errCh <- ready.wait()
		}()
	}
	ready.stop()
	for i := 0; i < 10; i++ {
		if err := <-errCh; err == nil {
			t.Errorf("unexpected success on channel %d", i)
		}
	}
}

func Test_newReadyCheck(t *testing.T) {
	ready := newReady()
	// it starts as false
	if ready.check() {
		t.Errorf("unexpected ready state %v", ready.check())
	}
	ready.set(true)
	if !ready.check() {
		t.Errorf("unexpected ready state %v", ready.check())
	}
	// stop sets ready to false
	ready.stop()
	if ready.check() {
		t.Errorf("unexpected ready state %v", ready.check())
	}
	// can not set to true if is stopped
	ready.set(true)
	if ready.check() {
		t.Errorf("unexpected ready state %v", ready.check())
	}
	err := ready.wait()
	if err == nil {
		t.Errorf("expected error waiting on a stopped state")
	}
}
