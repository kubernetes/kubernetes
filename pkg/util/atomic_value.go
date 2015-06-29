/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"sync"
	"sync/atomic"
)

// TODO(ArtfulCoder)
// sync/atomic/Value was added in golang 1.4
// Once support is dropped for go 1.3, this type must be deprecated in favor of sync/atomic/Value.
// The functions are named Load/Store to match sync/atomic/Value function names.
type AtomicValue struct {
	value      interface{}
	valueMutex sync.RWMutex
}

func (at *AtomicValue) Store(val interface{}) {
	at.valueMutex.Lock()
	defer at.valueMutex.Unlock()
	at.value = val
}

func (at *AtomicValue) Load() interface{} {
	at.valueMutex.RLock()
	defer at.valueMutex.RUnlock()
	return at.value
}

// HighWaterMark is a thread-safe object for tracking the maximum value seen
// for some quantity.
type HighWaterMark int64

// Check returns true iff 'current' is the highest value ever seen.
func (hwm *HighWaterMark) Check(current int64) bool {
	for {
		old := atomic.LoadInt64((*int64)(hwm))
		if current <= old {
			return false
		}
		if atomic.CompareAndSwapInt64((*int64)(hwm), old, current) {
			return true
		}
	}
}
