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
	"time"

	"github.com/stretchr/testify/assert"
)

// testcase wraps a callback which is called with a TContext that intercepts
// errors and log output. Those get compared.
type testcase struct {
	cb             func(TContext)
	expectNoFail   bool
	expectError    string
	expectDuration time.Duration
	expectLog      string
}

func (tc testcase) run(t *testing.T) {
	bufferT := &logBufferT{T: t}
	tCtx := Init(bufferT)
	var err error
	tCtx, finalize := WithError(tCtx, &err)
	start := time.Now()
	func() {
		defer finalize()
		tc.cb(tCtx)
	}()

	log := bufferT.log.String()
	t.Logf("Log output:\n%s\n", log)
	if tc.expectLog != "" {
		assert.Equal(t, tc.expectLog, normalize(log))
	} else if log != "" {
		t.Error("Expected no log output.")
	}

	duration := time.Since(start)
	assert.InDelta(t, tc.expectDuration.Seconds(), duration.Seconds(), 0.1, "callback invocation duration %s", duration)
	assert.Equal(t, !tc.expectNoFail, tCtx.Failed(), "Failed()")
	if tc.expectError == "" {
		assert.NoError(t, err)
	} else if assert.Error(t, err) {
		t.Logf("Result:\n%s", err.Error())
		assert.Equal(t, tc.expectError, normalize(err.Error()))
	}
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

type logBufferT struct {
	*testing.T
	log strings.Builder
}

func (l *logBufferT) Log(args ...any) {
	l.log.WriteString(fmt.Sprintln(args...))
}

func (l *logBufferT) Logf(format string, args ...any) {
	l.log.WriteString(fmt.Sprintf(format, args...))
	l.log.WriteRune('\n')
}
