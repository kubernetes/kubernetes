/*
Copyright 2019 The Kubernetes Authors.

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

package promise

// WriteOnce represents a variable that is initially not set and can
// be set once and is readable.  This is the common meaning for
// "promise".
type WriteOnce interface {
	// WaitAndGet blocks until the variable gets a value and then
	// returns it.
	WaitAndGet(<-chan struct{}) (interface{}, error)

	// IsSet returns immediately with an indication of whether this
	// variable has been set.
	IsSet() bool

	// Set normally writes a value into this variable, unblocks every
	// goroutine waiting for this variable to have a value, and
	// returns true.  In the unhappy case that this variable is
	// already set, this method returns false without modifying the
	// variable's value.
	Set(interface{}) bool
}
