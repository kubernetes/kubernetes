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

package generator

import (
	"io"
)

// ErrorTracker tracks errors to the underlying writer, so that you can ignore
// them until you're ready to return.
type ErrorTracker struct {
	io.Writer
	err error
}

// NewErrorTracker makes a new error tracker; note that it implements io.Writer.
func NewErrorTracker(w io.Writer) *ErrorTracker {
	return &ErrorTracker{Writer: w}
}

// Write intercepts calls to Write.
func (et *ErrorTracker) Write(p []byte) (n int, err error) {
	if et.err != nil {
		return 0, et.err
	}
	n, err = et.Writer.Write(p)
	if err != nil {
		et.err = err
	}
	return n, err
}

// Error returns nil if no error has occurred, otherwise it returns the error.
func (et *ErrorTracker) Error() error {
	return et.err
}
