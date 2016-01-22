/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package util

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/util/intstr"

	"github.com/golang/glog"
)

// For testing, bypass HandleCrash.
var ReallyCrash bool

// For any test of the style:
//   ...
//   <- time.After(timeout):
//      t.Errorf("Timed out")
// The value for timeout should effectively be "forever." Obviously we don't want our tests to truly lock up forever, but 30s
// is long enough that it is effectively forever for the things that can slow down a run on a heavily contended machine
// (GC, seeks, etc), but not so long as to make a developer ctrl-c a test run if they do happen to break that test.
var ForeverTestTimeout = time.Second * 30

// PanicHandlers is a list of functions which will be invoked when a panic happens.
var PanicHandlers = []func(interface{}){logPanic}

// HandleCrash simply catches a crash and logs an error. Meant to be called via defer.
// Additional context-specific handlers can be provided, and will be called in case of panic
func HandleCrash(additionalHandlers ...func(interface{})) {
	if ReallyCrash {
		return
	}
	if r := recover(); r != nil {
		for _, fn := range PanicHandlers {
			fn(r)
		}
		for _, fn := range additionalHandlers {
			fn(r)
		}
	}
}

// logPanic logs the caller tree when a panic occurs.
func logPanic(r interface{}) {
	callers := ""
	for i := 0; true; i++ {
		_, file, line, ok := runtime.Caller(i)
		if !ok {
			break
		}
		callers = callers + fmt.Sprintf("%v:%v\n", file, line)
	}
	glog.Errorf("Recovered from panic: %#v (%v)\n%v", r, r, callers)
}

// ErrorHandlers is a list of functions which will be invoked when an unreturnable
// error occurs.
var ErrorHandlers = []func(error){logError}

// HandlerError is a method to invoke when a non-user facing piece of code cannot
// return an error and needs to indicate it has been ignored. Invoking this method
// is preferable to logging the error - the default behavior is to log but the
// errors may be sent to a remote server for analysis.
func HandleError(err error) {
	for _, fn := range ErrorHandlers {
		fn(err)
	}
}

// logError prints an error with the call stack of the location it was reported
func logError(err error) {
	glog.ErrorDepth(2, err)
}

// NeverStop may be passed to Until to make it never stop.
var NeverStop <-chan struct{} = make(chan struct{})

// Forever is syntactic sugar on top of Until
func Forever(f func(), period time.Duration) {
	Until(f, period, NeverStop)
}

// Until loops until stop channel is closed, running f every period.
// Catches any panics, and keeps going. f may not be invoked if
// stop channel is already closed. Pass NeverStop to Until if you
// don't want it stop.
func Until(f func(), period time.Duration, stopCh <-chan struct{}) {
	select {
	case <-stopCh:
		return
	default:
	}

	for {
		func() {
			defer HandleCrash()
			f()
		}()
		select {
		case <-stopCh:
			return
		case <-time.After(period):
		}
	}
}

func GetIntOrPercentValue(intOrStr *intstr.IntOrString) (int, bool, error) {
	switch intOrStr.Type {
	case intstr.Int:
		return intOrStr.IntValue(), false, nil
	case intstr.String:
		s := strings.Replace(intOrStr.StrVal, "%", "", -1)
		v, err := strconv.Atoi(s)
		if err != nil {
			return 0, false, fmt.Errorf("invalid value %q: %v", intOrStr.StrVal, err)
		}
		return int(v), true, nil
	}
	return 0, false, fmt.Errorf("invalid value: neither int nor percentage")
}

func GetValueFromPercent(percent int, value int) int {
	return int(math.Ceil(float64(percent) * (float64(value)) / 100))
}

// Takes a list of strings and compiles them into a list of regular expressions
func CompileRegexps(regexpStrings []string) ([]*regexp.Regexp, error) {
	regexps := []*regexp.Regexp{}
	for _, regexpStr := range regexpStrings {
		r, err := regexp.Compile(regexpStr)
		if err != nil {
			return []*regexp.Regexp{}, err
		}
		regexps = append(regexps, r)
	}
	return regexps, nil
}

// Detects if using systemd as the init system
// Please note that simply reading /proc/1/cmdline can be misleading because
// some installation of various init programs can automatically make /sbin/init
// a symlink or even a renamed version of their main program.
// TODO(dchen1107): realiably detects the init system using on the system:
// systemd, upstart, initd, etc.
func UsingSystemdInitSystem() bool {
	if _, err := os.Stat("/run/systemd/system"); err == nil {
		return true
	}

	return false
}

// Tests whether all pointer fields in a struct are nil.  This is useful when,
// for example, an API struct is handled by plugins which need to distinguish
// "no plugin accepted this spec" from "this spec is empty".
//
// This function is only valid for structs and pointers to structs.  Any other
// type will cause a panic.  Passing a typed nil pointer will return true.
func AllPtrFieldsNil(obj interface{}) bool {
	v := reflect.ValueOf(obj)
	if !v.IsValid() {
		panic(fmt.Sprintf("reflect.ValueOf() produced a non-valid Value for %#v", obj))
	}
	if v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return true
		}
		v = v.Elem()
	}
	for i := 0; i < v.NumField(); i++ {
		if v.Field(i).Kind() == reflect.Ptr && !v.Field(i).IsNil() {
			return false
		}
	}
	return true
}

func FileExists(filename string) (bool, error) {
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		return false, nil
	} else if err != nil {
		return false, err
	}
	return true, nil
}

// borrowed from ioutil.ReadDir
// ReadDir reads the directory named by dirname and returns
// a list of directory entries, minus those with lstat errors
func ReadDirNoExit(dirname string) ([]os.FileInfo, []error, error) {
	if dirname == "" {
		dirname = "."
	}

	f, err := os.Open(dirname)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	names, err := f.Readdirnames(-1)
	list := make([]os.FileInfo, 0, len(names))
	errs := make([]error, 0, len(names))
	for _, filename := range names {
		fip, lerr := os.Lstat(dirname + "/" + filename)
		if os.IsNotExist(lerr) {
			// File disappeared between readdir + stat.
			// Just treat it as if it didn't exist.
			continue
		}

		list = append(list, fip)
		errs = append(errs, lerr)
	}

	return list, errs, nil
}
