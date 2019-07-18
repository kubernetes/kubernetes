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

package util

import "context"

// ErrorChannel supports non-blocking send and receive operation to capture error.
// A maximum of one error is kept in the channel and the rest of the errors sent
// are ignored, unless the existing error is received and the channel becomes empty
// again.
type ErrorChannel struct {
	errCh chan error
}

// SendError sends an error without blocking the sender.
func (e *ErrorChannel) SendError(err error) {
	select {
	case e.errCh <- err:
	default:
	}
}

// SendErrorWithCancel sends an error without blocking the sender and calls
// cancel function.
func (e *ErrorChannel) SendErrorWithCancel(err error, cancel context.CancelFunc) {
	e.SendError(err)
	cancel()
}

// ReceiveError receives an error from channel without blocking on the receiver.
func (e *ErrorChannel) ReceiveError() error {
	select {
	case err := <-e.errCh:
		return err
	default:
		return nil
	}
}

// NewErrorChannel returns a new ErrorChannel.
func NewErrorChannel() *ErrorChannel {
	return &ErrorChannel{
		errCh: make(chan error, 1),
	}
}
