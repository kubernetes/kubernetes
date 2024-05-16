/*
Copyright 2017 The Kubernetes Authors.

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

package admission

import (
	"testing"
	"time"
)

func TestWaitForReady(t *testing.T) {
	handler := newFakeHandler()

	// 1. test no readyFunc
	if !handler.WaitForReady() {
		t.Errorf("Expect ready for no readyFunc provided.")
	}

	// 2. readyFunc return ready immediately
	readyFunc := func() bool {
		return true
	}
	handler.SetReadyFunc(readyFunc)
	if !handler.WaitForReady() {
		t.Errorf("Expect ready for readyFunc returns ready immediately.")
	}

	// 3. readyFunc always return not ready. WaitForReady timeout
	readyFunc = func() bool {
		return false
	}
	startTime := time.Now()
	handler.SetReadyFunc(readyFunc)
	if handler.WaitForReady() {
		t.Errorf("Expect not ready for readyFunc returns not ready immediately.")
	}
	if time.Since(startTime) < timeToWaitForReady {
		t.Errorf("Expect WaitForReady timeout.")
	}
}

func newFakeHandler() *Handler {
	return NewHandler(Create, Update)
}
