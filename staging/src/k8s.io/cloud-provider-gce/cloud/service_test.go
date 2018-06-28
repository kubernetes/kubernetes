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
	"fmt"
	"testing"
)

func TestPollOperation(t *testing.T) {
	const totalAttempts = 10
	var attempts int
	fo := &fakeOperation{isDoneFunc: func(ctx context.Context) (bool, error) {
		attempts++
		if attempts < totalAttempts {
			return false, nil
		}
		return true, nil
	}}
	s := Service{RateLimiter: &NopRateLimiter{}}
	// Check that pollOperation will retry the operation multiple times.
	err := s.pollOperation(context.Background(), fo)
	if err != nil {
		t.Errorf("pollOperation() = %v, want nil", err)
	}
	if attempts != totalAttempts {
		t.Errorf("`attempts` = %d, want %d", attempts, totalAttempts)
	}

	// Check that the operation's error is returned.
	fo.err = fmt.Errorf("test operation failed")
	err = s.pollOperation(context.Background(), fo)
	if err != fo.err {
		t.Errorf("pollOperation() = %v, want %v", err, fo.err)
	}
	fo.err = nil

	fo.isDoneFunc = func(ctx context.Context) (bool, error) {
		return false, nil
	}
	// Use context that has been cancelled and expect a context error returned.
	ctxCancelled, cancelled := context.WithCancel(context.Background())
	cancelled()
	// Verify context is cancelled by now.
	<-ctxCancelled.Done()
	// Check that pollOperation returns because the context is cancelled.
	err = s.pollOperation(ctxCancelled, fo)
	if err == nil {
		t.Errorf("pollOperation() = nil, want: %v", ctxCancelled.Err())
	}
}

type fakeOperation struct {
	isDoneFunc func(ctx context.Context) (bool, error)
	err        error
	rateKey    *RateLimitKey
}

func (f *fakeOperation) isDone(ctx context.Context) (bool, error) {
	return f.isDoneFunc(ctx)
}

func (f *fakeOperation) error() error {
	return f.err
}

func (f *fakeOperation) rateLimitKey() *RateLimitKey {
	return f.rateKey
}
