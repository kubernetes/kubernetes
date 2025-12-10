/*
Copyright 2025 The Kubernetes Authors.

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

package app

import "context"

type options struct {
	// cancelMainContext is used to cancel upper level
	// context when a background error occurs.
	// It's called by HandleError if set.
	cancelMainContext context.CancelCauseFunc
}

// TestOption implements the functional options pattern
// dedicated for usage in testing code.
type TestOption func(o *options) error

// CancelMainContext sets a context cancellation function for
// the plugin. This function is called by HandleError
// when an error occurs in the background.
func CancelMainContext(cancel context.CancelCauseFunc) TestOption {
	return func(o *options) error {
		o.cancelMainContext = cancel
		return nil
	}
}
