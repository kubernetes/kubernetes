/*
Copyright 2021 The logr Authors.

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

package logr

import (
	"context"
	"testing"
)

// testLogger is a Logger just for testing that does nothing.
type testLogger struct{}

func (l *testLogger) Enabled() bool {
	return false
}

func (l *testLogger) Info(msg string, keysAndValues ...interface{}) {
}

func (l *testLogger) Error(err error, msg string, keysAndValues ...interface{}) {
}

func (l *testLogger) V(level int) Logger {
	return l
}

func (l *testLogger) WithValues(keysAndValues ...interface{}) Logger {
	return l
}

func (l *testLogger) WithName(name string) Logger {
	return l
}

// Verify that it actually implements the interface
var _ Logger = &testLogger{}

func TestContext(t *testing.T) {
	ctx := context.TODO()

	if out := FromContext(ctx); out != nil {
		t.Errorf("expected nil logger, got %#v", out)
	}
	if out := FromContextOrDiscard(ctx); out == nil {
		t.Errorf("expected non-nil logger")
	} else if _, ok := out.(DiscardLogger); !ok {
		t.Errorf("expected a DiscardLogger, got %#v", out)
	}

	logger := &testLogger{}
	lctx := NewContext(ctx, logger)
	if out := FromContext(lctx); out == nil {
		t.Errorf("expected non-nil logger")
	} else if out.(*testLogger) != logger {
		t.Errorf("expected output to be the same as input: got in=%p, out=%p", logger, out)
	}
	if out := FromContextOrDiscard(lctx); out == nil {
		t.Errorf("expected non-nil logger")
	} else if out.(*testLogger) != logger {
		t.Errorf("expected output to be the same as input: got in=%p, out=%p", logger, out)
	}
}

// testCallDepthLogger is a Logger just for testing that does nothing.
type testCallDepthLogger struct {
	*testLogger
	depth int
}

func (l *testCallDepthLogger) WithCallDepth(depth int) Logger {
	return &testCallDepthLogger{l.testLogger, l.depth + depth}
}

// Verify that it actually implements the interface
var _ CallDepthLogger = &testCallDepthLogger{}

func TestWithCallDepth(t *testing.T) {
	// Test an impl that does not support it.
	t.Run("not supported", func(t *testing.T) {
		in := &testLogger{}
		out := WithCallDepth(in, 42)
		if out.(*testLogger) != in {
			t.Errorf("expected output to be the same as input: got in=%p, out=%p", in, out)
		}
	})

	// Test an impl that does support it.
	t.Run("supported", func(t *testing.T) {
		in := &testCallDepthLogger{&testLogger{}, 0}
		out := WithCallDepth(in, 42)
		if out.(*testCallDepthLogger) == in {
			t.Errorf("expected output to be different than input: got in=out=%p", in)
		}
		if cdl := out.(*testCallDepthLogger); cdl.depth != 42 {
			t.Errorf("expected depth=42, got %d", cdl.depth)
		}
	})
}
