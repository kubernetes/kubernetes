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

package apitesting

import (
	"io"
	"testing"
)

// Close and fail the test if it returns an error.
func Close(t TestingT, c io.Closer) {
	t.Helper()
	assertNoError(t, c.Close())
}

// CloseNoOp does nothing. Use as a replacement for Close when you
// need to disable a defer.
func CloseNoOp(TestingT, io.Closer) {}

// TestingT simulates assert.TestingT and assert.tHelper without adding
// testify as a non-test dependency.
type TestingT interface {
	Errorf(format string, args ...interface{})
	Helper()
}

// Ensure that testing T & B satisfy the TestingT interface
var _ TestingT = &testing.T{}
var _ TestingT = &testing.B{}

// assertNoError simulates assert.NoError without adding testify as a
// non-test dependency.
//
// In test files, use github.com/stretchr/testify/assert instead.
func assertNoError(t TestingT, err error) {
	t.Helper()
	if err != nil {
		t.Errorf("Received unexpected error:\n%+v", err)
	}
}
