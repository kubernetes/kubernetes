// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package delay

import (
	"bytes"
	"encoding/gob"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"

	"google.golang.org/appengine/internal"
	"google.golang.org/appengine/taskqueue"
)

type CustomType struct {
	N int
}

type CustomInterface interface {
	N() int
}

type CustomImpl int

func (c CustomImpl) N() int { return int(c) }

// CustomImpl needs to be registered with gob.
func init() {
	gob.Register(CustomImpl(0))
}

var (
	invalidFunc = Func("invalid", func() {})

	regFuncRuns = 0
	regFuncMsg  = ""
	regFunc     = Func("reg", func(c context.Context, arg string) {
		regFuncRuns++
		regFuncMsg = arg
	})

	custFuncTally = 0
	custFunc      = Func("cust", func(c context.Context, ct *CustomType, ci CustomInterface) {
		a, b := 2, 3
		if ct != nil {
			a = ct.N
		}
		if ci != nil {
			b = ci.N()
		}
		custFuncTally += a + b
	})

	anotherCustFunc = Func("cust2", func(c context.Context, n int, ct *CustomType, ci CustomInterface) {
	})

	varFuncMsg = ""
	varFunc    = Func("variadic", func(c context.Context, format string, args ...int) {
		// convert []int to []interface{} for fmt.Sprintf.
		as := make([]interface{}, len(args))
		for i, a := range args {
			as[i] = a
		}
		varFuncMsg = fmt.Sprintf(format, as...)
	})

	errFuncRuns = 0
	errFuncErr  = errors.New("error!")
	errFunc     = Func("err", func(c context.Context) error {
		errFuncRuns++
		if errFuncRuns == 1 {
			return nil
		}
		return errFuncErr
	})
)

type fakeContext struct {
	ctx     context.Context
	logging [][]interface{}
}

func newFakeContext() *fakeContext {
	f := new(fakeContext)
	f.ctx = internal.WithCallOverride(context.Background(), f.call)
	f.ctx = internal.WithLogOverride(f.ctx, f.logf)
	return f
}

func (f *fakeContext) call(ctx context.Context, service, method string, in, out proto.Message) error {
	panic("should never be called")
}

var logLevels = map[int64]string{1: "INFO", 3: "ERROR"}

func (f *fakeContext) logf(level int64, format string, args ...interface{}) {
	f.logging = append(f.logging, append([]interface{}{logLevels[level], format}, args...))
}

func TestInvalidFunction(t *testing.T) {
	c := newFakeContext()

	if got, want := invalidFunc.Call(c.ctx), fmt.Errorf("delay: func is invalid: %s", errFirstArg); got.Error() != want.Error() {
		t.Errorf("Incorrect error: got %q, want %q", got, want)
	}
}

func TestVariadicFunctionArguments(t *testing.T) {
	// Check the argument type validation for variadic functions.

	c := newFakeContext()

	calls := 0
	taskqueueAdder = func(c context.Context, t *taskqueue.Task, _ string) (*taskqueue.Task, error) {
		calls++
		return t, nil
	}

	varFunc.Call(c.ctx, "hi")
	varFunc.Call(c.ctx, "%d", 12)
	varFunc.Call(c.ctx, "%d %d %d", 3, 1, 4)
	if calls != 3 {
		t.Errorf("Got %d calls to taskqueueAdder, want 3", calls)
	}

	if got, want := varFunc.Call(c.ctx, "%d %s", 12, "a string is bad"), errors.New("delay: argument 3 has wrong type: string is not assignable to int"); got.Error() != want.Error() {
		t.Errorf("Incorrect error: got %q, want %q", got, want)
	}
}

func TestBadArguments(t *testing.T) {
	// Try running regFunc with different sets of inappropriate arguments.

	c := newFakeContext()

	tests := []struct {
		args    []interface{} // all except context
		wantErr string
	}{
		{
			args:    nil,
			wantErr: "delay: too few arguments to func: 1 < 2",
		},
		{
			args:    []interface{}{"lala", 53},
			wantErr: "delay: too many arguments to func: 3 > 2",
		},
		{
			args:    []interface{}{53},
			wantErr: "delay: argument 1 has wrong type: int is not assignable to string",
		},
	}
	for i, tc := range tests {
		got := regFunc.Call(c.ctx, tc.args...)
		if got.Error() != tc.wantErr {
			t.Errorf("Call %v: got %q, want %q", i, got, tc.wantErr)
		}
	}
}

func TestRunningFunction(t *testing.T) {
	c := newFakeContext()

	// Fake out the adding of a task.
	var task *taskqueue.Task
	taskqueueAdder = func(_ context.Context, tk *taskqueue.Task, queue string) (*taskqueue.Task, error) {
		if queue != "" {
			t.Errorf(`Got queue %q, expected ""`, queue)
		}
		task = tk
		return tk, nil
	}

	regFuncRuns, regFuncMsg = 0, "" // reset state
	const msg = "Why, hello!"
	regFunc.Call(c.ctx, msg)

	// Simulate the Task Queue service.
	req, err := http.NewRequest("POST", path, bytes.NewBuffer(task.Payload))
	if err != nil {
		t.Fatalf("Failed making http.Request: %v", err)
	}
	rw := httptest.NewRecorder()
	runFunc(c.ctx, rw, req)

	if regFuncRuns != 1 {
		t.Errorf("regFuncRuns: got %d, want 1", regFuncRuns)
	}
	if regFuncMsg != msg {
		t.Errorf("regFuncMsg: got %q, want %q", regFuncMsg, msg)
	}
}

func TestCustomType(t *testing.T) {
	c := newFakeContext()

	// Fake out the adding of a task.
	var task *taskqueue.Task
	taskqueueAdder = func(_ context.Context, tk *taskqueue.Task, queue string) (*taskqueue.Task, error) {
		if queue != "" {
			t.Errorf(`Got queue %q, expected ""`, queue)
		}
		task = tk
		return tk, nil
	}

	custFuncTally = 0 // reset state
	custFunc.Call(c.ctx, &CustomType{N: 11}, CustomImpl(13))

	// Simulate the Task Queue service.
	req, err := http.NewRequest("POST", path, bytes.NewBuffer(task.Payload))
	if err != nil {
		t.Fatalf("Failed making http.Request: %v", err)
	}
	rw := httptest.NewRecorder()
	runFunc(c.ctx, rw, req)

	if custFuncTally != 24 {
		t.Errorf("custFuncTally = %d, want 24", custFuncTally)
	}

	// Try the same, but with nil values; one is a nil pointer (and thus a non-nil interface value),
	// and the other is a nil interface value.
	custFuncTally = 0 // reset state
	custFunc.Call(c.ctx, (*CustomType)(nil), nil)

	// Simulate the Task Queue service.
	req, err = http.NewRequest("POST", path, bytes.NewBuffer(task.Payload))
	if err != nil {
		t.Fatalf("Failed making http.Request: %v", err)
	}
	rw = httptest.NewRecorder()
	runFunc(c.ctx, rw, req)

	if custFuncTally != 5 {
		t.Errorf("custFuncTally = %d, want 5", custFuncTally)
	}
}

func TestRunningVariadic(t *testing.T) {
	c := newFakeContext()

	// Fake out the adding of a task.
	var task *taskqueue.Task
	taskqueueAdder = func(_ context.Context, tk *taskqueue.Task, queue string) (*taskqueue.Task, error) {
		if queue != "" {
			t.Errorf(`Got queue %q, expected ""`, queue)
		}
		task = tk
		return tk, nil
	}

	varFuncMsg = "" // reset state
	varFunc.Call(c.ctx, "Amiga %d has %d KB RAM", 500, 512)

	// Simulate the Task Queue service.
	req, err := http.NewRequest("POST", path, bytes.NewBuffer(task.Payload))
	if err != nil {
		t.Fatalf("Failed making http.Request: %v", err)
	}
	rw := httptest.NewRecorder()
	runFunc(c.ctx, rw, req)

	const expected = "Amiga 500 has 512 KB RAM"
	if varFuncMsg != expected {
		t.Errorf("varFuncMsg = %q, want %q", varFuncMsg, expected)
	}
}

func TestErrorFunction(t *testing.T) {
	c := newFakeContext()

	// Fake out the adding of a task.
	var task *taskqueue.Task
	taskqueueAdder = func(_ context.Context, tk *taskqueue.Task, queue string) (*taskqueue.Task, error) {
		if queue != "" {
			t.Errorf(`Got queue %q, expected ""`, queue)
		}
		task = tk
		return tk, nil
	}

	errFunc.Call(c.ctx)

	// Simulate the Task Queue service.
	// The first call should succeed; the second call should fail.
	{
		req, err := http.NewRequest("POST", path, bytes.NewBuffer(task.Payload))
		if err != nil {
			t.Fatalf("Failed making http.Request: %v", err)
		}
		rw := httptest.NewRecorder()
		runFunc(c.ctx, rw, req)
	}
	{
		req, err := http.NewRequest("POST", path, bytes.NewBuffer(task.Payload))
		if err != nil {
			t.Fatalf("Failed making http.Request: %v", err)
		}
		rw := httptest.NewRecorder()
		runFunc(c.ctx, rw, req)
		if rw.Code != http.StatusInternalServerError {
			t.Errorf("Got status code %d, want %d", rw.Code, http.StatusInternalServerError)
		}

		wantLogging := [][]interface{}{
			{"ERROR", "delay: func failed (will retry): %v", errFuncErr},
		}
		if !reflect.DeepEqual(c.logging, wantLogging) {
			t.Errorf("Incorrect logging: got %+v, want %+v", c.logging, wantLogging)
		}
	}
}
