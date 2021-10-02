// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package metric // import "go.opentelemetry.io/otel/sdk/metric"

import (
	"sync/atomic"
)

// refcountMapped atomically counts the number of references (usages) of an entry
// while also keeping a state of mapped/unmapped into a different data structure
// (an external map or list for example).
//
// refcountMapped uses an atomic value where the least significant bit is used to
// keep the state of mapping ('1' is used for unmapped and '0' is for mapped) and
// the rest of the bits are used for refcounting.
type refcountMapped struct {
	// refcount has to be aligned for 64-bit atomic operations.
	value int64
}

// ref returns true if the entry is still mapped and increases the
// reference usages, if unmapped returns false.
func (rm *refcountMapped) ref() bool {
	// Check if this entry was marked as unmapped between the moment
	// we got a reference to it (or will be removed very soon) and here.
	return atomic.AddInt64(&rm.value, 2)&1 == 0
}

func (rm *refcountMapped) unref() {
	atomic.AddInt64(&rm.value, -2)
}

// tryUnmap flips the mapped bit to "unmapped" state and returns true if both of the
// following conditions are true upon entry to this function:
//  * There are no active references;
//  * The mapped bit is in "mapped" state.
// Otherwise no changes are done to mapped bit and false is returned.
func (rm *refcountMapped) tryUnmap() bool {
	if atomic.LoadInt64(&rm.value) != 0 {
		return false
	}
	return atomic.CompareAndSwapInt64(
		&rm.value,
		0,
		1,
	)
}
