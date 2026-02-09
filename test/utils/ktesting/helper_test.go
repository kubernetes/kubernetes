/*
Copyright 2024 The Kubernetes Authors.

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

package ktesting

import (
	"fmt"
	"regexp"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/assert"
)

// testcase wraps a callback which is called with a TContext that intercepts
// errors and log output. Those get compared.
type testcase struct {
	cb             func(TContext)
	expectDuration time.Duration
	expectTrace    string
}

func (tc testcase) run(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		buffer := &mockTB{}
		tCtx := Init(buffer)
		start := time.Now()
		func() {
			defer func() {
				if r := recover(); r != nil && r != logBufferStop {
					panic(r)
				}
			}()
			tc.cb(tCtx)
		}()
		duration := time.Since(start)

		trace := buffer.log.String()
		t.Logf("Trace:\n%s\n", trace)
		assert.Equal(t, tc.expectDuration, duration, "callback invocation duration %s")
		assert.Equal(t, tc.expectTrace, normalize(trace))
	})
}

// normalize replaces parts of message texts which may vary with constant strings.
func normalize(msg string) string {
	// duration
	msg = regexp.MustCompile(`[[:digit:]]+\.[[:digit:]]+s`).ReplaceAllString(msg, "x.y s")
	// hex pointer value
	msg = regexp.MustCompile(`0x[[:xdigit:]]+`).ReplaceAllString(msg, "0xXXXX")
	// per-test klog header
	msg = regexp.MustCompile(`[EI][[:digit:]]{4} [[:digit:]]{2}:[[:digit:]]{2}:[[:digit:]]{2}\.[[:digit:]]{6}\]`).ReplaceAllString(msg, "<klog header>:")
	return msg
}

// mockTB records which calls were made (type and parameters)
//
// The final string looks similar to the output visible in `go test -v`,
// except that it is visible how the sausage was made (Fatal vs Log + FailNow).
// Log+FailNow and Fatal are equivalent with testing.T, but not
// with Ginkgo as underlying TB because it can only properly
// capture the failure message if Fatal is used.
type mockTB struct {
	log strings.Builder
}

func (m *mockTB) Attr(key, value string) {
	m.log.WriteString(fmt.Sprintf("(ATTR) %q %q\n", key, value))
}

func (m *mockTB) Chdir(dir string) {
	m.log.WriteString(fmt.Sprintf("(CHDIR) %q\n", dir))
}

func (m *mockTB) Cleanup(func()) {
	// Gets called by Init all the time, not logged because it's distracting.
	// m.log.WriteString("(CLEANUP)\n")
}

func (m *mockTB) Error(args ...any) {
	m.log.WriteString(fmt.Sprintln(append([]any{"(ERROR)"}, args...)...))
}

func (m *mockTB) Errorf(format string, args ...any) {
	m.log.WriteString(fmt.Sprintf("(ERRORF) "+format+"\n", args))
}

func (m *mockTB) Fail() {
	m.log.WriteString("(FAIL)\n")
}

func (m *mockTB) FailNow() {
	m.log.WriteString("(FAILNOW)\n")
	panic(logBufferStop)
}

func (m *mockTB) Failed() bool {
	m.log.WriteString("(FAILED)\n")
	return false
}

func (m *mockTB) Fatal(args ...any) {
	m.log.WriteString(fmt.Sprintln(append([]any{"(FATAL)"}, args...)...))
	panic(logBufferStop)
}

func (m *mockTB) Fatalf(format string, args ...any) {
	m.log.WriteString(fmt.Sprintf("(FATALF) "+format+"\n", args))
	panic(logBufferStop)
}

func (m *mockTB) Helper() {
	// TODO: include stack unwinding to verify that Helper is called in the right places.
	// Merely logging it is not sufficient.
	// m.log.WriteString("HELPER\n")
}

func (m *mockTB) Log(args ...any) {
	m.log.WriteString(fmt.Sprintln(append([]any{"(LOG)"}, args...)...))
}

func (m *mockTB) Logf(format string, args ...any) {
	m.log.WriteString(fmt.Sprintf("(LOGF) "+format+"\n", args))
}

func (m *mockTB) Name() string {
	// Gets called by Init all the time, not logged because its distracting.
	// m.log.WriteString("(NAME)\n")
	return "logBufferT"
}

func (m *mockTB) Setenv(key, value string) {
	m.log.WriteString("(SETENV)\n")
}

func (m *mockTB) Skip(args ...any) {
	m.log.WriteString(fmt.Sprintln(append([]any{"(SKIP)"}, args...)...))
	panic(logBufferStop)
}

func (m *mockTB) SkipNow() {
	m.log.WriteString("(SKIPNOW)\n")
	panic(logBufferStop)
}

func (m *mockTB) Skipf(format string, args ...any) {
	m.log.WriteString(fmt.Sprintf("(SKIPF) "+format+"\n", args...))
	panic(logBufferStop)
}

func (m *mockTB) Skipped() bool {
	m.log.WriteString("(SKIPPED)\n")
	return false
}

func (m *mockTB) TempDir() string {
	m.log.WriteString("(TEMPDIR)\n")
	return "/no-such-dir"
}

var (
	logBufferStop = "STOP"
)
