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

package allocator

// Interface manages the allocation of items out of a range. Interface
// should be threadsafe.
type Interface interface {
	Allocate(int) (bool, error)
	AllocateNext() (int, bool, error)
	Release(int) error
	ForEach(func(int))
	Has(int) bool
	Free() int

	// Destroy shuts down all internal structures.
	// Destroy needs to be implemented in thread-safe way and be prepared for being
	// called more than once.
	Destroy()
}

// Snapshottable is an Interface that can be snapshotted and restored. Snapshottable
// should be threadsafe.
type Snapshottable interface {
	Interface
	Snapshot() (string, []byte)
	Restore(string, []byte) error
}

type AllocatorFactory func(max int, rangeSpec string) (Interface, error)

type AllocatorWithOffsetFactory func(max int, rangeSpec string, offset int) (Interface, error)
