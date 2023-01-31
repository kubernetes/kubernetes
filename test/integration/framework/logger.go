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

// RedirectKlog modifies the global klog logger so that it writes via
// tb.Log. This only works when different tests run sequentially and causes
// this file to be logged by testing.TB, so this is inferior compared to
// per-test output with klog/ktesting. Converting code to use ktesting
// and contextual logging is preferred.
//
// The writer is returned. It can be used to redirect also other output, for
// example of commands.
func RedirectKlog(tb testing.TB) io.Writer {
	expectNoError := func(err error) {
		if err != nil {
			tb.Fatalf("unexpected error: %v", err)
		}
	}

	state := klog.CaptureState()
	tb.Cleanup(state.Restore)
	var fs flag.FlagSet
	klog.InitFlags(&fs)
	expectNoError(fs.Set("log_file", "/dev/null"))
	expectNoError(fs.Set("logtostderr", "false"))
	expectNoError(fs.Set("alsologtostderr", "false"))
	expectNoError(fs.Set("stderrthreshold", "10"))
	expectNoError(fs.Set("one_output", "true"))
	output := testingWriter{TB: tb}
	klog.SetOutput(output)
	return output
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
	// right either. klog itself would have to call TB.Helper, but doesn't
	// know about the TB instance and thus cannot do it.
	tw.Log(string(data[:logLen]))
	return len(data), nil
}

var _ io.Writer = testingWriter{}
