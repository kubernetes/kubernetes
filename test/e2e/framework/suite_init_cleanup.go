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

package framework

import (
	"sync"
)

type SuiteInitActionHandle *int

var suiteInitActionsLock sync.Mutex
var suiteInitActions = map[SuiteInitActionHandle]func(){}

// AddSuiteInitAction installs a function that will be called when the test suite is being setup.
// This allows any frameworks extending the core framework to hook into SynchronizedBeforeSuite().
func AddSuiteInitAction(fn func()) {
	p := SuiteInitActionHandle(new(int))
	suiteInitActionsLock.Lock()
	defer suiteInitActionsLock.Unlock()
	suiteInitActions[p] = fn
}

// RunSuiteInitActions runs all functions installed by AddSuiteInitAction during test suite setup.
func RunSuiteInitActions() {
	list := []func(){}
	func() {
		suiteInitActionsLock.Lock()
		defer suiteInitActionsLock.Unlock()
		for _, fn := range suiteInitActions {
			list = append(list, fn)
		}
	}()

	for _, fn := range list {
		fn()
	}
}

type SuiteCleanupActionHandle *int

var suiteCleanupActionsLock sync.Mutex
var suiteCleanupActions = map[SuiteCleanupActionHandle]func(){}

// AddSuiteCleanupAction installs a function that will be called when the whole test suite being terminated.
// This allows any frameworks extending the core framework to hook into SynchronizedAfterSuite().
func AddSuiteCleanupAction(fn func()) {
	p := SuiteCleanupActionHandle(new(int))
	suiteCleanupActionsLock.Lock()
	defer suiteCleanupActionsLock.Unlock()
	suiteCleanupActions[p] = fn
}

// RunSuiteCleanupActions runs all functions installed by AddSuiteCleanupAction during test suite termination.
func RunSuiteCleanupActions() {
	list := []func(){}
	func() {
		suiteCleanupActionsLock.Lock()
		defer suiteCleanupActionsLock.Unlock()
		for _, fn := range suiteCleanupActions {
			list = append(list, fn)
		}
	}()

	for _, fn := range list {
		fn()
	}
}
