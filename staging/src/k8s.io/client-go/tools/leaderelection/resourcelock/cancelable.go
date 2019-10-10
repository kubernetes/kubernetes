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

package resourcelock

import (
	"context"
)

// TODO(directxman12): get rid of this when we get generated clients that accept contexts --
// the underlying machinery is there in RESTClient, but it's not exposed.  The contents of
// this file are just a stopgap until then.

// StubbornInterface is like Interface, except that it does not support using a
// context for timeout/cancelation.
//
// Use MakeCancelable to convert a StubbornInterface into an Interface, but prefer
// to just write using cancelable code natively.
type StubbornInterface interface {
	// Get returns the LeaderElectionRecord
	Get() (*LeaderElectionRecord, []byte, error)

	// Create attempts to create a LeaderElectionRecord
	Create(ler LeaderElectionRecord) error

	// Update will update and existing LeaderElectionRecord
	Update(ler LeaderElectionRecord) error

	// RecordEvent is used to record events
	RecordEvent(string)

	// Identity will return the locks Identity
	Identity() string

	// Describe is used to convert details on current resource lock
	// into a string
	Describe() string
}

// MakeCancelable makes a stubborn (non-cancelable) StubbornInterface into a (cancelable) Interface
// using goroutines.  This won't magically time out the underlying HTTP requests, etc.
func MakeCancelable(raw StubbornInterface) Interface {
	return &cancelableLockWrapper{wrapped: raw}
}

// cancelableLockWrapper embeds a non-cancelable StubbornInterface
// and turns it into a cancelable interface using goroutines.
type cancelableLockWrapper struct {
	wrapped StubbornInterface
}

func (w *cancelableLockWrapper) Get(ctx context.Context) (*LeaderElectionRecord, []byte, error) {
	errCh := make(chan error, 1)
	var rec *LeaderElectionRecord
	var raw []byte
	go func() {
		defer close(errCh)
		var err error
		rec, raw, err = w.wrapped.Get()
		errCh <- err
	}()

	select {
	case err := <-errCh:
		return rec, raw, err
	case <-ctx.Done():
		// timeout
		return nil, nil, ctx.Err()
	}
}
func (w *cancelableLockWrapper) Create(ctx context.Context, ler LeaderElectionRecord) error {
	errCh := make(chan error, 1)
	go func() {
		defer close(errCh)
		errCh <- w.wrapped.Create(ler)
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		// timeout
		return ctx.Err()
	}
}
func (w *cancelableLockWrapper) Update(ctx context.Context, ler LeaderElectionRecord) error {
	errCh := make(chan error, 1)
	go func() {
		defer close(errCh)
		errCh <- w.wrapped.Update(ler)
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		// timeout
		return ctx.Err()
	}
}
func (w *cancelableLockWrapper) RecordEvent(evt string) {
	w.wrapped.RecordEvent(evt)
}
func (w *cancelableLockWrapper) Identity() string {
	return w.wrapped.Identity()
}
func (w *cancelableLockWrapper) Describe() string {
	return w.wrapped.Describe()
}
