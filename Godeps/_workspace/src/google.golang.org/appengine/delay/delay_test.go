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

	"google.golang.org/appengine"
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
	regFunc     = Func("reg", func(c appengine.Context, arg string) {
		regFuncRuns++
		regFuncMsg = arg
	})

	custFuncTally = 0
	custFunc      = Func("cust", func(c appengine.Context, ct *CustomType, ci CustomInterface) {
		a, b := 2, 3
		if ct != nil {
			a = ct.N
		}
		if ci != nil {
			b = ci.N()
		}
		custFuncTally += a + b
	})

	anotherCustFunc = Func("cust2", func(c appengine.Context, n int, ct *CustomType, ci CustomInterface) {
	})

	varFuncMsg = ""
	varFunc    = Func("variadic", func(c appengine.Context, format string, args ...int) {
		// convert []int to []interface{} for fmt.Sprintf.
		as := make([]interface{}, len(args))
		for i, a := range args {
			as[i] = a
		}
		varFuncMsg = fmt.Sprintf(format, as...)
	})

	errFuncRuns = 0
	errFuncErr  = errors.New("error!")
	errFunc     = Func("err", func(c appengine.Context) error {
		errFuncRuns++
		if errFuncRuns == 1 {
			return nil
		}
		return errFuncErr
	})
)

type fakeContext struct {
	appengine.Context
	logging [][]interface{}
}

func (f *fakeContext) log(level, format string, args ...interface{}) {
	f.logging = append(f.logging, append([]interface{}{level, format}, args...))
}

func (f *fakeContext) Infof(format string, args ...interface{})  { f.log("INFO", format, args...) }
func (f *fakeContext) Errorf(format string, args ...interface{}) { f.log("ERROR", format, args...) }

func TestInvalidFunction(t *testing.T) {
	c := &fakeContext{}

	invalidFunc.Call(c)

	wantLogging := [][]interface{}{
		{"ERROR", "%v", fmt.Errorf("delay: func is invalid: %s", errFirstArg)},
	}
	if !reflect.DeepEqual(c.logging, wantLogging) {
		t.Errorf("Incorrect logging: got %+v, want %+v", c.logging, wantLogging)
	}
}

func TestVariadicFunctionArguments(t *testing.T) {
	// Check the argument type validation for variadic functions.

	c := &fakeContext{}

	calls := 0
	taskqueueAdder = func(c appengine.Context, t *taskqueue.Task, _ string) (*taskqueue.Task, error) {
		calls++
		return t, nil
	}

	varFunc.Call(c, "hi")
	varFunc.Call(c, "%d", 12)
	varFunc.Call(c, "%d %d %d", 3, 1, 4)
	if calls != 3 {
		t.Errorf("Got %d calls to taskqueueAdder, want 3", calls)
	}

	varFunc.Call(c, "%d %s", 12, "a string is bad")
	wantLogging := [][]interface{}{
		{"ERROR", "%v", errors.New("delay: argument 3 has wrong type: string is not assignable to int")},
	}
	if !reflect.DeepEqual(c.logging, wantLogging) {
		t.Errorf("Incorrect logging: got %+v, want %+v", c.logging, wantLogging)
	}
}

func TestBadArguments(t *testing.T) {
	// Try running regFunc with different sets of inappropriate arguments.

	c := &fakeContext{}

	regFunc.Call(c)
	regFunc.Call(c, "lala", 53)
	regFunc.Call(c, 53)

	wantLogging := [][]interface{}{
		{"ERROR", "%v", errors.New("delay: too few arguments to func: 1 < 2")},
		{"ERROR", "%v", errors.New("delay: too many arguments to func: 3 > 2")},
		{"ERROR", "%v", errors.New("delay: argument 1 has wrong type: int is not assignable to string")},
	}
	if !reflect.DeepEqual(c.logging, wantLogging) {
		t.Errorf("Incorrect logging: got %+v, want %+v", c.logging, wantLogging)
	}
}

func TestRunningFunction(t *testing.T) {
	c := &fakeContext{}

	// Fake out the adding of a task.
	var task *taskqueue.Task
	taskqueueAdder = func(_ appengine.Context, tk *taskqueue.Task, queue string) (*taskqueue.Task, error) {
		if queue != "" {
			t.Errorf(`Got queue %q, expected ""`, queue)
		}
		task = tk
		return tk, nil
	}

	regFuncRuns, regFuncMsg = 0, "" // reset state
	const msg = "Why, hello!"
	regFunc.Call(c, msg)

	// Simulate the Task Queue service.
	req, err := http.NewRequest("POST", path, bytes.NewBuffer(task.Payload))
	if err != nil {
		t.Fatalf("Failed making http.Request: %v", err)
	}
	rw := httptest.NewRecorder()
	runFunc(c, rw, req)

	if regFuncRuns != 1 {
		t.Errorf("regFuncRuns: got %d, want 1", regFuncRuns)
	}
	if regFuncMsg != msg {
		t.Errorf("regFuncMsg: got %q, want %q", regFuncMsg, msg)
	}
}

func TestCustomType(t *testing.T) {
	c := &fakeContext{}

	// Fake out the adding of a task.
	var task *taskqueue.Task
	taskqueueAdder = func(_ appengine.Context, tk *taskqueue.Task, queue string) (*taskqueue.Task, error) {
		if queue != "" {
			t.Errorf(`Got queue %q, expected ""`, queue)
		}
		task = tk
		return tk, nil
	}

	custFuncTally = 0 // reset state
	custFunc.Call(c, &CustomType{N: 11}, CustomImpl(13))

	// Simulate the Task Queue service.
	req, err := http.NewRequest("POST", path, bytes.NewBuffer(task.Payload))
	if err != nil {
		t.Fatalf("Failed making http.Request: %v", err)
	}
	rw := httptest.NewRecorder()
	runFunc(c, rw, req)

	if custFuncTally != 24 {
		t.Errorf("custFuncTally = %d, want 24", custFuncTally)
	}

	// Try the same, but with nil values; one is a nil pointer (and thus a non-nil interface value),
	// and the other is a nil interface value.
	custFuncTally = 0 // reset state
	custFunc.Call(c, (*CustomType)(nil), nil)

	// Simulate the Task Queue service.
	req, err = http.NewRequest("POST", path, bytes.NewBuffer(task.Payload))
	if err != nil {
		t.Fatalf("Failed making http.Request: %v", err)
	}
	rw = httptest.NewRecorder()
	runFunc(c, rw, req)

	if custFuncTally != 5 {
		t.Errorf("custFuncTally = %d, want 5", custFuncTally)
	}
}

func TestRunningVariadic(t *testing.T) {
	c := &fakeContext{}

	// Fake out the adding of a task.
	var task *taskqueue.Task
	taskqueueAdder = func(_ appengine.Context, tk *taskqueue.Task, queue string) (*taskqueue.Task, error) {
		if queue != "" {
			t.Errorf(`Got queue %q, expected ""`, queue)
		}
		task = tk
		return tk, nil
	}

	varFuncMsg = "" // reset state
	varFunc.Call(c, "Amiga %d has %d KB RAM", 500, 512)

	// Simulate the Task Queue service.
	req, err := http.NewRequest("POST", path, bytes.NewBuffer(task.Payload))
	if err != nil {
		t.Fatalf("Failed making http.Request: %v", err)
	}
	rw := httptest.NewRecorder()
	runFunc(c, rw, req)

	const expected = "Amiga 500 has 512 KB RAM"
	if varFuncMsg != expected {
		t.Errorf("varFuncMsg = %q, want %q", varFuncMsg, expected)
	}
}

func TestErrorFunction(t *testing.T) {
	c := &fakeContext{}

	// Fake out the adding of a task.
	var task *taskqueue.Task
	taskqueueAdder = func(_ appengine.Context, tk *taskqueue.Task, queue string) (*taskqueue.Task, error) {
		if queue != "" {
			t.Errorf(`Got queue %q, expected ""`, queue)
		}
		task = tk
		return tk, nil
	}

	errFunc.Call(c)

	// Simulate the Task Queue service.
	// The first call should succeed; the second call should fail.
	{
		req, err := http.NewRequest("POST", path, bytes.NewBuffer(task.Payload))
		if err != nil {
			t.Fatalf("Failed making http.Request: %v", err)
		}
		rw := httptest.NewRecorder()
		runFunc(c, rw, req)
	}
	{
		req, err := http.NewRequest("POST", path, bytes.NewBuffer(task.Payload))
		if err != nil {
			t.Fatalf("Failed making http.Request: %v", err)
		}
		rw := httptest.NewRecorder()
		runFunc(c, rw, req)
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
