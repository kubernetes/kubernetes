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

package proc

import (
	"fmt"
	"sync"
	"time"
)

type errorOnce struct {
	once  sync.Once
	err   chan error
	abort <-chan struct{}
}

// NewErrorOnce creates an ErrorOnce that aborts blocking func calls once
// the given abort chan has closed.
func NewErrorOnce(abort <-chan struct{}) ErrorOnce {
	return &errorOnce{
		err:   make(chan error, 1),
		abort: abort,
	}
}

func (b *errorOnce) Err() <-chan error {
	return b.err
}

func (b *errorOnce) Reportf(msg string, args ...interface{}) {
	b.Report(fmt.Errorf(msg, args...))
}

func (b *errorOnce) Report(err error) {
	b.once.Do(func() {
		select {
		case b.err <- err:
		default:
		}
	})
}

func (b *errorOnce) Send(errIn <-chan error) ErrorOnce {
	if errIn == nil {
		// don't execute this in a goroutine; save resources AND the caller
		// likely wants this executed ASAP because some of some operation
		// ordering semantics. forward() will not block here on a nil input
		// so this is safe to do.
		b.forward(nil)
	} else {
		go b.forward(errIn)
	}
	return b
}

func (b *errorOnce) forward(errIn <-chan error) {
	if errIn == nil {
		// important: nil never blocks; Report(nil) is guaranteed to be a
		// non-blocking operation.
		b.Report(nil)
		return
	}
	select {
	case err, _ := <-errIn:
		b.Report(err)
	case <-b.abort:
		// double-check that errIn was blocked: don't falsely return
		// errProcessTerminated if errIn was really ready
		select {
		case err, _ := <-errIn:
			b.Report(err)
		default:
			b.Report(errProcessTerminated)
		}
	}
}

func (b *errorOnce) WaitFor(d time.Duration) (error, bool) {
	t := time.NewTimer(d)
	select {
	case err, _ := <-b.err:
		t.Stop()
		return err, true
	case <-t.C:
		return nil, false
	}
}
