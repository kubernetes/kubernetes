/*
Copyright 2014 The Go4 Authors

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

// Package fault handles fault injection for testing.
package fault // import "go4.org/fault"

import (
	"errors"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

var fakeErr = errors.New("fake injected error for testing")

// An Injector reports whether fake errors should be returned.
type Injector struct {
	failPercent int
}

// NewInjector returns a new fault injector with the given name.  The
// environment variable "FAULT_" + capital(name) + "_FAIL_PERCENT"
// controls the percentage of requests that fail. If undefined or
// zero, no requests fail.
func NewInjector(name string) *Injector {
	var failPercent, _ = strconv.Atoi(os.Getenv("FAULT_" + strings.ToUpper(name) + "_FAIL_PERCENT"))
	return &Injector{
		failPercent: failPercent,
	}
}

// ShouldFail reports whether a fake error should be returned.
func (in *Injector) ShouldFail() bool {
	return in.failPercent > 0 && in.failPercent > rand.Intn(100)
}

// FailErr checks ShouldFail and, if true, assigns a fake error to err
// and returns true.
func (in *Injector) FailErr(err *error) bool {
	if !in.ShouldFail() {
		return false
	}
	*err = fakeErr
	return true
}
