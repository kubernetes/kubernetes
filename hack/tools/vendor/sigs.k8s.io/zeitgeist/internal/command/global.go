/*
Copyright 2021 The Kubernetes Authors.

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

package command

import (
	"sync/atomic"
)

// atomicInt is the global variable for storing the globally set verbosity
// level. It should never be used directly to avoid data races.
var atomicInt int32

// SetGlobalVerbose sets the global command verbosity to the specified value
func SetGlobalVerbose(to bool) {
	var i int32 = 0
	if to {
		i = 1
	}
	atomic.StoreInt32(&atomicInt, i)
}

// GetGlobalVerbose returns the globally set command verbosity
func GetGlobalVerbose() bool {
	return atomic.LoadInt32(&atomicInt) != 0
}
