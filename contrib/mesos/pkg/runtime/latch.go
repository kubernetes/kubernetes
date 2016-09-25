/*
Copyright 2015 The Kubernetes Authors.

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
	"sync/atomic"
)

type Latch struct {
	int32
}

// return true if this latch was successfully acquired. concurrency safe. will only return true
// upon the first invocation, all subsequent invocations will return false. always returns false
// when self is nil.
func (self *Latch) Acquire() bool {
	if self == nil {
		return false
	}
	return atomic.CompareAndSwapInt32(&self.int32, 0, 1)
}
