/*
Copyright 2014 The Kubernetes Authors.

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

package runtime

import (
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/golang/glog"
)

var (
	// ReallyCrash controls the behavior of HandleCrash and now defaults
	// true. It's still exposed so components can optionally set to false
	// to restore prior behavior.
	ReallyCrash = true
)

// PanicHandlers is a list of functions which will be invoked when a panic happens.
var PanicHandlers = []func(interface{}){logPanic}

// HandleCrash simply catches a crash and logs an error. Meant to be called via
// defer.  Additional context-specific handlers can be provided, and will be
// called in case of panic.  HandleCrash actually crashes, after calling the
// handlers and logging the panic message.
//
// TODO: remove this function. We are switching to a world where it's safe for
// apiserver to panic, since it will be restarted by kubelet. At the beginning
// of the Kubernetes project, nothing was going to restart apiserver and so
// catching panics was important. But it's actually much simpler for montoring
// software if we just exit when an unexpected panic happens.
func HandleCrash(additionalHandlers ...func(interface{})) {
	if r := recover(); r != nil {
		for _, fn := range PanicHandlers {
			fn(r)
		}
		for _, fn := range additionalHandlers {
			fn(r)
		}
		if ReallyCrash {
			// Actually proceed to panic.
			panic(r)
		}
	}
}

// logPanic logs the caller tree when a panic occurs.
func logPanic(r interface{}) {
	callers := getCallers(r)
	glog.Errorf("Observed a panic: %#v (%v)\n%v", r, r, callers)
}

func getCallers(r interface{}) string {
	callers := ""
	for i := 0; true; i++ {
		_, file, line, ok := runtime.Caller(i)
		if !ok {
			break
		}
		callers = callers + fmt.Sprintf("%v:%v\n", file, line)
	}

	return callers
}

// ErrorHandlers is a list of functions which will be invoked when an unreturnable
// error occurs.
// TODO(lavalamp): for testability, this and the below HandleError function
// should be packaged up into a testable and reusable object.
var ErrorHandlers = []func(error){
	logError,
	(&rudimentaryErrorBackoff{
		lastErrorTime: time.Now(),
		// 1ms was the number folks were able to stomach as a global rate limit.
		// If you need to log errors more than 1000 times a second you
		// should probably consider fixing your code instead. :)
		minPeriod: time.Millisecond,
	}).OnError,
}

// HandlerError is a method to invoke when a non-user facing piece of code cannot
// return an error and needs to indicate it has been ignored. Invoking this method
// is preferable to logging the error - the default behavior is to log but the
// errors may be sent to a remote server for analysis.
func HandleError(err error) {
	// this is sometimes called with a nil error.  We probably shouldn't fail and should do nothing instead
	if err == nil {
		return
	}

	for _, fn := range ErrorHandlers {
		fn(err)
	}
}

// logError prints an error with the call stack of the location it was reported
func logError(err error) {
	glog.ErrorDepth(2, err)
}

type rudimentaryErrorBackoff struct {
	minPeriod time.Duration // immutable
	// TODO(lavalamp): use the clock for testability. Need to move that
	// package for that to be accessible here.
	lastErrorTimeLock sync.Mutex
	lastErrorTime     time.Time
}

// OnError will block if it is called more often than the embedded period time.
// This will prevent overly tight hot error loops.
func (r *rudimentaryErrorBackoff) OnError(error) {
	r.lastErrorTimeLock.Lock()
	defer r.lastErrorTimeLock.Unlock()
	d := time.Since(r.lastErrorTime)
	if d < r.minPeriod {
		time.Sleep(r.minPeriod - d)
	}
	r.lastErrorTime = time.Now()
}

// GetCaller returns the caller of the function that calls it.
func GetCaller() string {
	var pc [1]uintptr
	runtime.Callers(3, pc[:])
	f := runtime.FuncForPC(pc[0])
	if f == nil {
		return fmt.Sprintf("Unable to find caller")
	}
	return f.Name()
}

// RecoverFromPanic replaces the specified error with an error containing the
// original error, and  the call tree when a panic occurs. This enables error
// handlers to handle errors and panics the same way.
func RecoverFromPanic(err *error) {
	if r := recover(); r != nil {
		callers := getCallers(r)

		*err = fmt.Errorf(
			"recovered from panic %q. (err=%v) Call stack:\n%v",
			r,
			*err,
			callers)
	}
}
