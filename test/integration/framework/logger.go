/*
Copyright 2023 The Kubernetes Authors.

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

package framework

import (
	"flag"
	"io"
	"testing"

	"k8s.io/klog/v2"
)

// RedirectKlog modifies the global klog logger so that it writes via the given
// writer. This only works when different tests run sequentially.
//
// The returned cleanup function restores the previous state. Beware that it is
// not thread-safe, all goroutines which call klog must have been stopped.
func RedirectKlog(tb testing.TB, output io.Writer) func() {
	expectNoError := func(err error) {
		if err != nil {
			tb.Fatalf("unexpected error: %v", err)
		}
	}

	state := klog.CaptureState()
	defer func() {
		if r := recover(); r != nil {
			state.Restore()
			panic(r)
		}
	}()
	var fs flag.FlagSet
	klog.InitFlags(&fs)
	expectNoError(fs.Set("log_file", "/dev/null"))
	expectNoError(fs.Set("logtostderr", "false"))
	expectNoError(fs.Set("alsologtostderr", "false"))
	expectNoError(fs.Set("stderrthreshold", "10"))
	expectNoError(fs.Set("one_output", "true"))
	klog.SetOutput(output)
	return state.Restore
}

// NewTBWriter creates an io.Writer which turns each write into a tb.Log call.
//
// Note that no attempts are made to determine the actual call site because
// our caller doesn't know about the TB instance and thus cannot mark itself
// as helper. Therefore the code here doesn't do it either and thus shows up
// as call site in the testing output. To avoid that, contextual logging
// and ktesting have to be used.
func NewTBWriter(tb testing.TB) io.Writer {
	return testingWriter{TB: tb}
}

type testingWriter struct {
	testing.TB
}

func (tw testingWriter) Write(data []byte) (int, error) {
	logLen := len(data)
	if logLen == 0 {
		return 0, nil
	}
	// Trim trailing line break? Log will add it.
	if data[logLen-1] == '\n' {
		logLen--
	}
	// We could call TB.Helper here, but that doesn't really help because
	// then our caller (code in klog) will be reported instead, which isn't
	// right either. klog would have to call TB.Helper itself, but doesn't
	// know about the TB instance.
	tw.Log(string(data[:logLen]))
	return len(data), nil
}

var _ io.Writer = testingWriter{}
