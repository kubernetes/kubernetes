// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

/*
Package delay provides a way to execute code outside the scope of a
user request by using the taskqueue API.

To declare a function that may be executed later, call Func
in a top-level assignment context, passing it an arbitrary string key
and a function whose first argument is of type context.Context.
	var laterFunc = delay.Func("key", myFunc)
It is also possible to use a function literal.
	var laterFunc = delay.Func("key", func(c context.Context, x string) {
		// ...
	})

To call a function, invoke its Call method.
	laterFunc.Call(c, "something")
A function may be called any number of times. If the function has any
return arguments, and the last one is of type error, the function may
return a non-nil error to signal that the function should be retried.

The arguments to functions may be of any type that is encodable by the gob
package. If an argument is of interface type, it is the client's responsibility
to register with the gob package whatever concrete type may be passed for that
argument; see http://golang.org/pkg/gob/#Register for details.

Any errors during initialization or execution of a function will be
logged to the application logs. Error logs that occur during initialization will
be associated with the request that invoked the Call method.

The state of a function invocation that has not yet successfully
executed is preserved by combining the file name in which it is declared
with the string key that was passed to the Func function. Updating an app
with pending function invocations is safe as long as the relevant
functions have the (filename, key) combination preserved.

The delay package uses the Task Queue API to create tasks that call the
reserved application path "/_ah/queue/go/delay".
This path must not be marked as "login: required" in app.yaml;
it must be marked as "login: admin" or have no access restriction.
*/
package delay // import "google.golang.org/appengine/delay"

import (
	"bytes"
	"encoding/gob"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"runtime"

	"golang.org/x/net/context"

	"google.golang.org/appengine"
	"google.golang.org/appengine/log"
	"google.golang.org/appengine/taskqueue"
)

// Function represents a function that may have a delayed invocation.
type Function struct {
	fv  reflect.Value // Kind() == reflect.Func
	key string
	err error // any error during initialization
}

const (
	// The HTTP path for invocations.
	path = "/_ah/queue/go/delay"
	// Use the default queue.
	queue = ""
)

var (
	// registry of all delayed functions
	funcs = make(map[string]*Function)

	// precomputed types
	contextType = reflect.TypeOf((*context.Context)(nil)).Elem()
	errorType   = reflect.TypeOf((*error)(nil)).Elem()

	// errors
	errFirstArg = errors.New("first argument must be context.Context")
)

// Func declares a new Function. The second argument must be a function with a
// first argument of type context.Context.
// This function must be called at program initialization time. That means it
// must be called in a global variable declaration or from an init function.
// This restriction is necessary because the instance that delays a function
// call may not be the one that executes it. Only the code executed at program
// initialization time is guaranteed to have been run by an instance before it
// receives a request.
func Func(key string, i interface{}) *Function {
	f := &Function{fv: reflect.ValueOf(i)}

	// Derive unique, somewhat stable key for this func.
	_, file, _, _ := runtime.Caller(1)
	f.key = file + ":" + key

	t := f.fv.Type()
	if t.Kind() != reflect.Func {
		f.err = errors.New("not a function")
		return f
	}
	if t.NumIn() == 0 || t.In(0) != contextType {
		f.err = errFirstArg
		return f
	}

	// Register the function's arguments with the gob package.
	// This is required because they are marshaled inside a []interface{}.
	// gob.Register only expects to be called during initialization;
	// that's fine because this function expects the same.
	for i := 0; i < t.NumIn(); i++ {
		// Only concrete types may be registered. If the argument has
		// interface type, the client is resposible for registering the
		// concrete types it will hold.
		if t.In(i).Kind() == reflect.Interface {
			continue
		}
		gob.Register(reflect.Zero(t.In(i)).Interface())
	}

	funcs[f.key] = f
	return f
}

type invocation struct {
	Key  string
	Args []interface{}
}

// Call invokes a delayed function.
//   err := f.Call(c, ...)
// is equivalent to
//   t, _ := f.Task(...)
//   err := taskqueue.Add(c, t, "")
func (f *Function) Call(c context.Context, args ...interface{}) error {
	t, err := f.Task(args...)
	if err != nil {
		return err
	}
	_, err = taskqueueAdder(c, t, queue)
	return err
}

// Task creates a Task that will invoke the function.
// Its parameters may be tweaked before adding it to a queue.
// Users should not modify the Path or Payload fields of the returned Task.
func (f *Function) Task(args ...interface{}) (*taskqueue.Task, error) {
	if f.err != nil {
		return nil, fmt.Errorf("delay: func is invalid: %v", f.err)
	}

	nArgs := len(args) + 1 // +1 for the context.Context
	ft := f.fv.Type()
	minArgs := ft.NumIn()
	if ft.IsVariadic() {
		minArgs--
	}
	if nArgs < minArgs {
		return nil, fmt.Errorf("delay: too few arguments to func: %d < %d", nArgs, minArgs)
	}
	if !ft.IsVariadic() && nArgs > minArgs {
		return nil, fmt.Errorf("delay: too many arguments to func: %d > %d", nArgs, minArgs)
	}

	// Check arg types.
	for i := 1; i < nArgs; i++ {
		at := reflect.TypeOf(args[i-1])
		var dt reflect.Type
		if i < minArgs {
			// not a variadic arg
			dt = ft.In(i)
		} else {
			// a variadic arg
			dt = ft.In(minArgs).Elem()
		}
		// nil arguments won't have a type, so they need special handling.
		if at == nil {
			// nil interface
			switch dt.Kind() {
			case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
				continue // may be nil
			}
			return nil, fmt.Errorf("delay: argument %d has wrong type: %v is not nilable", i, dt)
		}
		switch at.Kind() {
		case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
			av := reflect.ValueOf(args[i-1])
			if av.IsNil() {
				// nil value in interface; not supported by gob, so we replace it
				// with a nil interface value
				args[i-1] = nil
			}
		}
		if !at.AssignableTo(dt) {
			return nil, fmt.Errorf("delay: argument %d has wrong type: %v is not assignable to %v", i, at, dt)
		}
	}

	inv := invocation{
		Key:  f.key,
		Args: args,
	}

	buf := new(bytes.Buffer)
	if err := gob.NewEncoder(buf).Encode(inv); err != nil {
		return nil, fmt.Errorf("delay: gob encoding failed: %v", err)
	}

	return &taskqueue.Task{
		Path:    path,
		Payload: buf.Bytes(),
	}, nil
}

var taskqueueAdder = taskqueue.Add // for testing

func init() {
	http.HandleFunc(path, func(w http.ResponseWriter, req *http.Request) {
		runFunc(appengine.NewContext(req), w, req)
	})
}

func runFunc(c context.Context, w http.ResponseWriter, req *http.Request) {
	defer req.Body.Close()

	var inv invocation
	if err := gob.NewDecoder(req.Body).Decode(&inv); err != nil {
		log.Errorf(c, "delay: failed decoding task payload: %v", err)
		log.Warningf(c, "delay: dropping task")
		return
	}

	f := funcs[inv.Key]
	if f == nil {
		log.Errorf(c, "delay: no func with key %q found", inv.Key)
		log.Warningf(c, "delay: dropping task")
		return
	}

	ft := f.fv.Type()
	in := []reflect.Value{reflect.ValueOf(c)}
	for _, arg := range inv.Args {
		var v reflect.Value
		if arg != nil {
			v = reflect.ValueOf(arg)
		} else {
			// Task was passed a nil argument, so we must construct
			// the zero value for the argument here.
			n := len(in) // we're constructing the nth argument
			var at reflect.Type
			if !ft.IsVariadic() || n < ft.NumIn()-1 {
				at = ft.In(n)
			} else {
				at = ft.In(ft.NumIn() - 1).Elem()
			}
			v = reflect.Zero(at)
		}
		in = append(in, v)
	}
	out := f.fv.Call(in)

	if n := ft.NumOut(); n > 0 && ft.Out(n-1) == errorType {
		if errv := out[n-1]; !errv.IsNil() {
			log.Errorf(c, "delay: func failed (will retry): %v", errv.Interface())
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
	}
}
