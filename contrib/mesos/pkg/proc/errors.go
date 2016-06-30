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
	"errors"
	"fmt"

	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
)

var (
	errProcessTerminated = errors.New("cannot execute action because process has terminated")
	errIllegalState      = errors.New("illegal state, cannot execute action")

	closedErrChan <-chan error // singleton chan that's always closed
)

func init() {
	ch := make(chan error)
	close(ch)
	closedErrChan = ch
}

func IsProcessTerminated(err error) bool {
	return err == errProcessTerminated
}

func IsIllegalState(err error) bool {
	return err == errIllegalState
}

// OnError spawns a goroutine that waits for an error. if a non-nil error is read from
// the channel then the handler func is invoked, otherwise (nil error or closed chan)
// the handler is skipped. if a nil handler is specified then it's not invoked.
// the signal chan that's returned closes once the error process logic (and handler,
// if any) has completed.
func OnError(ch <-chan error, f func(error), abort <-chan struct{}) <-chan struct{} {
	return runtime.After(func() {
		if ch == nil {
			return
		}
		select {
		case err, ok := <-ch:
			if ok && err != nil && f != nil {
				f(err)
			}
		case <-abort:
			if f != nil {
				f(errProcessTerminated)
			}
		}
	})
}

// ErrorChanf is a convenience func that returns a chan that yields an error
// generated from the given msg format and args.
func ErrorChanf(msg string, args ...interface{}) <-chan error {
	return ErrorChan(fmt.Errorf(msg, args...))
}

// ErrorChan is a convenience func that returns a chan that yields the given error.
// If err is nil then a closed chan is returned.
func ErrorChan(err error) <-chan error {
	if err == nil {
		return closedErrChan
	}
	ch := make(chan error, 1)
	ch <- err
	return ch
}
