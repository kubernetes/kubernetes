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

package coverage

import (
	"io"
)

// This is an implementation of testing.testDeps. It doesn't need to do anything, because
// no tests are actually run. It does need a concrete implementation of at least ImportPath,
// which is called unconditionally when running tests.
type fakeTestDeps struct{}

func (fakeTestDeps) ImportPath() string {
	return ""
}

func (fakeTestDeps) MatchString(pat, str string) (bool, error) {
	return false, nil
}

func (fakeTestDeps) StartCPUProfile(io.Writer) error {
	return nil
}

func (fakeTestDeps) StopCPUProfile() {}

func (fakeTestDeps) StartTestLog(io.Writer) {}

func (fakeTestDeps) StopTestLog() error {
	return nil
}

func (fakeTestDeps) WriteHeapProfile(io.Writer) error {
	return nil
}

func (fakeTestDeps) WriteProfileTo(string, io.Writer, int) error {
	return nil
}
