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

package pool

import (
	"testing"
)

func Test_MemoryPoolAllocator_Allocate(t *testing.T) {
	driver := &TestPoolDriver{Items: []string{"a", "b", "c"}}
	pa := &MemoryPoolAllocator{}
	pa.Init(driver)

	TestPoolAllocatorAllocate(t, pa)
}

func Test_MemoryPoolAllocator_AllocateNext(t *testing.T) {
	driver := &TestPoolDriver{Items: []string{"a", "b", "c"}}
	pa := &MemoryPoolAllocator{}
	pa.Init(driver)

	// Turn off random allocation attempts, so we just allocate in sequence
	pa.DisableRandomAllocation()

	TestPoolAllocatorAllocateNext(t, pa)
}

func Test_MemoryPoolAllocator_Release(t *testing.T) {
	driver := &TestPoolDriver{Items: []string{"a", "b", "c"}}
	pa := &MemoryPoolAllocator{}
	pa.Init(driver)

	// Turn off random allocation attempts, so we just allocate in sequence
	pa.DisableRandomAllocation()

	TestPoolAllocatorRelease(t, pa)
}
