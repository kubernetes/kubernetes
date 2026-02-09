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
	"reflect"
	"time"
)

// This is an implementation of testing.testDeps. It doesn't need to do anything, because
// no tests are actually run. It does need a concrete implementation of at least ImportPath,
// which is called unconditionally when running tests.
//
//nolint:unused // U1000 see comment above, we know it's unused normally.
type fakeTestDeps struct{}

// https://go.dev/src/testing/fuzz.go#L88
//
//nolint:unused // U1000 see comment above, we know it's unused normally.
type corpusEntry = struct {
	Parent     string
	Path       string
	Data       []byte
	Values     []any
	Generation int
	IsSeed     bool
}

//nolint:unused // U1000 see comment above, we know it's unused normally.
func (fakeTestDeps) ImportPath() string {
	return ""
}

//nolint:unused // U1000 see comment above, we know it's unused normally.
func (fakeTestDeps) MatchString(pat, str string) (bool, error) {
	return false, nil
}

//nolint:unused // U1000 see comment above, we know it's unused normally.
func (fakeTestDeps) SetPanicOnExit0(bool) {}

//nolint:unused // U1000 see comment above, we know it's unused normally.
func (fakeTestDeps) StartCPUProfile(io.Writer) error {
	return nil
}

//nolint:unused // U1000 see comment above, we know it's unused normally.
func (fakeTestDeps) StopCPUProfile() {}

//nolint:unused // U1000 see comment above, we know it's unused normally.
func (fakeTestDeps) StartTestLog(io.Writer) {}

//nolint:unused // U1000 see comment above, we know it's unused normally.
func (fakeTestDeps) StopTestLog() error {
	return nil
}

//nolint:unused // U1000 see comment above, we know it's unused normally.
func (fakeTestDeps) WriteHeapProfile(io.Writer) error {
	return nil
}

//nolint:unused // U1000 see comment above, we know it's unused normally.
func (fakeTestDeps) WriteProfileTo(string, io.Writer, int) error {
	return nil
}

//nolint:unused // U1000 see comment above, we know it's unused normally.
func (fakeTestDeps) CoordinateFuzzing(time.Duration, int64, time.Duration, int64, int, []corpusEntry, []reflect.Type, string, string) error {
	return nil
}

//nolint:unused // U1000 see comment above, we know it's unused normally.
func (fakeTestDeps) RunFuzzWorker(func(corpusEntry) error) error {
	return nil
}

//nolint:unused // U1000 see comment above, we know it's unused normally.
func (fakeTestDeps) ReadCorpus(string, []reflect.Type) ([]corpusEntry, error) {
	return nil, nil
}

//nolint:unused // U1000 see comment above, we know it's unused normally.
func (fakeTestDeps) CheckCorpus([]any, []reflect.Type) error {
	return nil
}

//nolint:unused // U1000 see comment above, we know it's unused normally.
func (fakeTestDeps) ResetCoverage() {}

//nolint:unused // U1000 see comment above, we know it's unused normally.
func (fakeTestDeps) SnapshotCoverage() {}

//nolint:unused // U1000 see comment above, we know it's unused normally.
func (fakeTestDeps) InitRuntimeCoverage() (mode string, tearDown func(string, string) (string, error), snapcov func() float64) {
	return "", nil, nil
}
