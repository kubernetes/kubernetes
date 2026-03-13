/*
Copyright 2024 The Kubernetes Authors.

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

package cbor

import (
	"fmt"
	goruntime "runtime"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/cbor"
)

var (
	scheme      = runtime.NewScheme()
	serializers = []cbor.Serializer{
		cbor.NewSerializer(scheme, scheme),
		cbor.NewSerializer(scheme, scheme, cbor.Strict(true)),
	}
)

// FuzzDecodeAllocations is a go-fuzz target that panics on inputs that cause an unreasonably large
// number of bytes to be allocated at decode time.
func FuzzDecodeAllocations(data []byte) (result int) {
	const (
		MaxInputBytes     = 128
		MaxAllocatedBytes = 16 * 1024
	)

	if len(data) > MaxInputBytes {
		// Longer inputs can require more allocations by unmarshaling to larger
		// objects. Focus on identifying short inputs that allocate an unreasonable number
		// of bytes to identify pathological cases.
		return -1
	}

	decode := func(serializer cbor.Serializer, data []byte) int {
		var u unstructured.Unstructured
		o, gvk, err := serializer.Decode(data, &schema.GroupVersionKind{}, &u)
		if err != nil {
			if o != nil {
				panic("returned non-nil error and non-nil runtime.Object")
			}

			return 0
		}

		if o == nil || gvk == nil {
			panic("returned nil error and nil runtime.Object or nil schema.GroupVersionKind")
		}

		return 1
	}

	for _, serializer := range serializers {
		// The first pass pre-warms anything that is lazily initialized. Doing things like
		// logging for the first time in a process can account for allocations on the order
		// of tens of kB.
		decode(serializer, data)

		var nBytesAllocated uint64
		for trial := 1; trial <= 10; trial++ {
			func() {
				defer goruntime.GOMAXPROCS(goruntime.GOMAXPROCS(1))
				var mem goruntime.MemStats
				goruntime.ReadMemStats(&mem)

				result |= decode(serializer, data)

				nBytesAllocated = mem.TotalAlloc
				goruntime.ReadMemStats(&mem)
				nBytesAllocated = mem.TotalAlloc - nBytesAllocated

			}()

			// The exact number of bytes allocated may vary due to allocations in
			// concurrently-executing goroutines or implementation details of the
			// runtime. Only panic on inputs that consistently exceed the allocation
			// threshold to reduce the false positive rate.
			if nBytesAllocated <= MaxAllocatedBytes {
				break
			}
		}

		if nBytesAllocated > MaxAllocatedBytes {
			panic(fmt.Sprintf("%d bytes allocated to decode input of length %d exceeds maximum of %d", nBytesAllocated, len(data), MaxAllocatedBytes))
		}
	}

	return result
}
