/*
Copyright 2020 The Kubernetes Authors.

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

package value

// Allocator provides a value object allocation strategy.
// Value objects can be allocated by passing an allocator to the "Using"
// receiver functions on the value interfaces, e.g. Map.ZipUsing(allocator, ...).
// Value objects returned from "Using" functions should be given back to the allocator
// once longer needed by calling Allocator.Free(Value).
type Allocator interface {
	// Free gives the allocator back any value objects returned by the "Using"
	// receiver functions on the value interfaces.
	// interface{} may be any of: Value, Map, List or Range.
	Free(interface{})

	// The unexported functions are for "Using" receiver functions of the value types
	// to request what they need from the allocator.
	allocValueUnstructured() *valueUnstructured
	allocListUnstructuredRange() *listUnstructuredRange
	allocValueReflect() *valueReflect
	allocMapReflect() *mapReflect
	allocStructReflect() *structReflect
	allocListReflect() *listReflect
	allocListReflectRange() *listReflectRange
}

// HeapAllocator simply allocates objects to the heap. It is the default
// allocator used receiver functions on the value interfaces that do not accept
// an allocator and should be used whenever allocating objects that will not
// be given back to an allocator by calling Allocator.Free(Value).
var HeapAllocator = &heapAllocator{}

type heapAllocator struct{}

func (p *heapAllocator) allocValueUnstructured() *valueUnstructured {
	return &valueUnstructured{}
}

func (p *heapAllocator) allocListUnstructuredRange() *listUnstructuredRange {
	return &listUnstructuredRange{vv: &valueUnstructured{}}
}

func (p *heapAllocator) allocValueReflect() *valueReflect {
	return &valueReflect{}
}

func (p *heapAllocator) allocStructReflect() *structReflect {
	return &structReflect{}
}

func (p *heapAllocator) allocMapReflect() *mapReflect {
	return &mapReflect{}
}

func (p *heapAllocator) allocListReflect() *listReflect {
	return &listReflect{}
}

func (p *heapAllocator) allocListReflectRange() *listReflectRange {
	return &listReflectRange{vr: &valueReflect{}}
}

func (p *heapAllocator) Free(_ interface{}) {}

// NewFreelistAllocator creates freelist based allocator.
// This allocator provides fast allocation and freeing of short lived value objects.
//
// The freelists are bounded in size by freelistMaxSize. If more than this amount of value objects is
// allocated at once, the excess will be returned to the heap for garbage collection when freed.
//
// This allocator is unsafe and must not be accessed concurrently by goroutines.
//
// This allocator works well for traversal of value data trees. Typical usage is to acquire
// a freelist at the beginning of the traversal and use it through out
// for all temporary value access.
func NewFreelistAllocator() Allocator {
	return &freelistAllocator{
		valueUnstructured: &freelist{new: func() interface{} {
			return &valueUnstructured{}
		}},
		listUnstructuredRange: &freelist{new: func() interface{} {
			return &listUnstructuredRange{vv: &valueUnstructured{}}
		}},
		valueReflect: &freelist{new: func() interface{} {
			return &valueReflect{}
		}},
		mapReflect: &freelist{new: func() interface{} {
			return &mapReflect{}
		}},
		structReflect: &freelist{new: func() interface{} {
			return &structReflect{}
		}},
		listReflect: &freelist{new: func() interface{} {
			return &listReflect{}
		}},
		listReflectRange: &freelist{new: func() interface{} {
			return &listReflectRange{vr: &valueReflect{}}
		}},
	}
}

// Bound memory usage of freelists. This prevents the processing of very large lists from leaking memory.
// This limit is large enough for endpoints objects containing 1000 IP address entries. Freed objects
// that don't fit into the freelist are orphaned on the heap to be garbage collected.
const freelistMaxSize = 1000

type freelistAllocator struct {
	valueUnstructured     *freelist
	listUnstructuredRange *freelist
	valueReflect          *freelist
	mapReflect            *freelist
	structReflect         *freelist
	listReflect           *freelist
	listReflectRange      *freelist
}

type freelist struct {
	list []interface{}
	new  func() interface{}
}

func (f *freelist) allocate() interface{} {
	var w2 interface{}
	if n := len(f.list); n > 0 {
		w2, f.list = f.list[n-1], f.list[:n-1]
	} else {
		w2 = f.new()
	}
	return w2
}

func (f *freelist) free(v interface{}) {
	if len(f.list) < freelistMaxSize {
		f.list = append(f.list, v)
	}
}

func (w *freelistAllocator) Free(value interface{}) {
	switch v := value.(type) {
	case *valueUnstructured:
		v.Value = nil // don't hold references to unstructured objects
		w.valueUnstructured.free(v)
	case *listUnstructuredRange:
		v.vv.Value = nil // don't hold references to unstructured objects
		w.listUnstructuredRange.free(v)
	case *valueReflect:
		v.ParentMapKey = nil
		v.ParentMap = nil
		w.valueReflect.free(v)
	case *mapReflect:
		w.mapReflect.free(v)
	case *structReflect:
		w.structReflect.free(v)
	case *listReflect:
		w.listReflect.free(v)
	case *listReflectRange:
		v.vr.ParentMapKey = nil
		v.vr.ParentMap = nil
		w.listReflectRange.free(v)
	}
}

func (w *freelistAllocator) allocValueUnstructured() *valueUnstructured {
	return w.valueUnstructured.allocate().(*valueUnstructured)
}

func (w *freelistAllocator) allocListUnstructuredRange() *listUnstructuredRange {
	return w.listUnstructuredRange.allocate().(*listUnstructuredRange)
}

func (w *freelistAllocator) allocValueReflect() *valueReflect {
	return w.valueReflect.allocate().(*valueReflect)
}

func (w *freelistAllocator) allocStructReflect() *structReflect {
	return w.structReflect.allocate().(*structReflect)
}

func (w *freelistAllocator) allocMapReflect() *mapReflect {
	return w.mapReflect.allocate().(*mapReflect)
}

func (w *freelistAllocator) allocListReflect() *listReflect {
	return w.listReflect.allocate().(*listReflect)
}

func (w *freelistAllocator) allocListReflectRange() *listReflectRange {
	return w.listReflectRange.allocate().(*listReflectRange)
}
