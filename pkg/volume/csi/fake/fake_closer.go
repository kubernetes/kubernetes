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

package fake

import (
	"testing"
)

// NewCloser returns a fake io.Closer
func NewCloser(t *testing.T) *Closer {
	return &Closer{
		t: t,
	}
}

// Closer is a fake io.Closer used for testing
type Closer struct {
	wasCalled bool
	t         *testing.T
}

// Close implements the io.Closer interface
func (c *Closer) Close() error {
	c.wasCalled = true
	return nil
}

// Check is a test helper function that fails if Close
// was not called
func (c *Closer) Check() *Closer {
	c.t.Helper()

	if !c.wasCalled {
		c.t.Error("expected closer to have been called")
	}

	return c
}
