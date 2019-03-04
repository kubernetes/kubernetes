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

package testing

import (
	"bytes"
	"sync"
)

// ThreadsafeBuffer is a goroutine safe bytes.Buffer
type ThreadsafeBuffer struct {
	buffer bytes.Buffer
	mutex  sync.Mutex
}

// Write implements interface
func (tb *ThreadsafeBuffer) Write(data []byte) (n int, err error) {
	tb.mutex.Lock()
	defer tb.mutex.Unlock()
	return tb.buffer.Write(data)
}

// String implements interface
func (tb *ThreadsafeBuffer) String() string {
	tb.mutex.Lock()
	defer tb.mutex.Unlock()
	return tb.buffer.String()
}
