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

package openstack

import (
	"testing"
)

func TestCaller(t *testing.T) {
	called := false
	myFunc := func() { called = true }

	c := NewCaller()
	c.Call(myFunc)

	if !called {
		t.Errorf("Caller failed to call function in default case")
	}

	c.Disarm()
	called = false
	c.Call(myFunc)

	if called {
		t.Error("Caller still called function when disarmed")
	}

	// Confirm the "usual" deferred Caller pattern works as expected

	called = false
	success_case := func() {
		c := NewCaller()
		defer c.Call(func() { called = true })
		c.Disarm()
	}
	if success_case(); called {
		t.Error("Deferred success case still invoked unwind")
	}

	called = false
	failure_case := func() {
		c := NewCaller()
		defer c.Call(func() { called = true })
	}
	if failure_case(); !called {
		t.Error("Deferred failure case failed to invoke unwind")
	}
}
