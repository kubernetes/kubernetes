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

package cloud

import (
	"context"
	"testing"
	"time"
)

type FakeAcceptor struct{ accept func() }

func (f *FakeAcceptor) Accept() {
	f.accept()
}

func TestAcceptRateLimiter(t *testing.T) {
	fa := &FakeAcceptor{accept: func() {}}
	arl := &AcceptRateLimiter{fa}
	err := arl.Accept(context.Background(), nil)
	if err != nil {
		t.Errorf("AcceptRateLimiter.Accept() = %v, want nil", err)
	}

	// Use context that has been cancelled and expect a context error returned.
	ctxCancelled, cancelled := context.WithCancel(context.Background())
	cancelled()
	// Verify context is cancelled by now.
	<-ctxCancelled.Done()

	fa.accept = func() { time.Sleep(1 * time.Second) }
	err = arl.Accept(ctxCancelled, nil)
	if err != ctxCancelled.Err() {
		t.Errorf("AcceptRateLimiter.Accept() = %v, want %v", err, ctxCancelled.Err())
	}
}

func TestMinimumRateLimiter(t *testing.T) {
	fa := &FakeAcceptor{accept: func() {}}
	arl := &AcceptRateLimiter{fa}
	var called bool
	fa.accept = func() { called = true }
	m := &MinimumRateLimiter{RateLimiter: arl, Minimum: 10 * time.Millisecond}

	err := m.Accept(context.Background(), nil)
	if err != nil {
		t.Errorf("MinimumRateLimiter.Accept = %v, want nil", err)
	}
	if !called {
		t.Errorf("`called` = false, want true")
	}

	// Use context that has been cancelled and expect a context error returned.
	ctxCancelled, cancelled := context.WithCancel(context.Background())
	cancelled()
	// Verify context is cancelled by now.
	<-ctxCancelled.Done()
	called = false
	err = m.Accept(ctxCancelled, nil)
	if err != ctxCancelled.Err() {
		t.Errorf("AcceptRateLimiter.Accept() = %v, want %v", err, ctxCancelled.Err())
	}
	if called {
		t.Errorf("`called` = true, want false")
	}
}
