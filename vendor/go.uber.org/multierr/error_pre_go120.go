// Copyright (c) 2017-2023 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

//go:build !go1.20
// +build !go1.20

package multierr

import "errors"

// Versions of Go before 1.20 did not support the Unwrap() []error method.
// This provides a similar behavior by implementing the Is(..) and As(..)
// methods.
// See the errors.Join proposal for details:
// https://github.com/golang/go/issues/53435

// As attempts to find the first error in the error list that matches the type
// of the value that target points to.
//
// This function allows errors.As to traverse the values stored on the
// multierr error.
func (merr *multiError) As(target interface{}) bool {
	for _, err := range merr.Errors() {
		if errors.As(err, target) {
			return true
		}
	}
	return false
}

// Is attempts to match the provided error against errors in the error list.
//
// This function allows errors.Is to traverse the values stored on the
// multierr error.
func (merr *multiError) Is(target error) bool {
	for _, err := range merr.Errors() {
		if errors.Is(err, target) {
			return true
		}
	}
	return false
}

func extractErrors(err error) []error {
	if err == nil {
		return nil
	}

	// Note that we're casting to multiError, not errorGroup. Our contract is
	// that returned errors MAY implement errorGroup. Errors, however, only
	// has special behavior for multierr-specific error objects.
	//
	// This behavior can be expanded in the future but I think it's prudent to
	// start with as little as possible in terms of contract and possibility
	// of misuse.
	eg, ok := err.(*multiError)
	if !ok {
		return []error{err}
	}

	return append(([]error)(nil), eg.Errors()...)
}
