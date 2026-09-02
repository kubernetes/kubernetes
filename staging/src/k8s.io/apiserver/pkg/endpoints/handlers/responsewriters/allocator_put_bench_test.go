/*
Copyright The Kubernetes Authors.

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

package responsewriters

import (
	"fmt"
	"io"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/protobuf"
)

// BenchmarkProtobufEncodeAllocatorPutStrategies quantifies the cost of
// bounding the encode buffer pool on the non-streaming protobuf encode path
// (the one SerializeObject uses for single-object GET responses and
// non-streamed LISTs, where the allocator buffer holds the entire serialized
// response). It compares, on the same binary:
//
//   - Bound=Unbounded: the pre-bound behavior (and the behavior with the
//     AllocatorPoolBufferCap gate disabled), where a pooled buffer grows to
//     the largest response and is reused verbatim.
//   - Bound=<size>: PutAllocator with the pool bound set to that size (the
//     values a cluster operator would choose with
//     --max-pooled-encode-buffer-size), which drops buffers larger than the
//     bound, so every encode above it pays a fresh buffer allocation. Sizes
//     at or above ~1MiB are omitted: with streaming list encoding a buffer
//     holds at most one object, and the etcd object limit is 1.5MiB, so such
//     bounds behave like Unbounded.
//   - Pool=None: no pooling at all (SimpleAllocator), as a floor reference.
//
// The Unbounded→Bound delta on payloads above the bound is the CPU/alloc
// price of the bounded pool; payloads below the bound must show no
// difference.
func BenchmarkProtobufEncodeAllocatorPutStrategies(b *testing.B) {
	// Non-streaming serializer: one Allocate for the whole response.
	serializer := protobuf.NewSerializerWithOptions(nil, nil, protobuf.SerializerOptions{})

	previousBound := runtime.MaxPooledBufferCapacity()
	b.Cleanup(func() { runtime.SetMaxPooledBufferCapacity(previousBound) })

	type strategy struct {
		name  string
		bound int // -1: no pooling; 0: unbounded pool; >0: bounded pool
	}
	strategies := []strategy{
		{name: "Bound=Unbounded", bound: 0},
		{name: "Bound=128KiB", bound: 128 * 1024},
		{name: "Bound=256KiB", bound: 256 * 1024},
		{name: "Bound=384KiB", bound: 384 * 1024},
		{name: "Bound=512KiB", bound: 512 * 1024},
		{name: "Pool=None", bound: -1},
	}

	for _, count := range []int{50, 500, 5_000} {
		podList := benchmarkItems(b, count)
		payloadSize := podList.Size()
		b.Run(fmt.Sprintf("Pods=%d/PayloadKiB=%d", count, payloadSize/1024), func(b *testing.B) {
			for _, s := range strategies {
				b.Run(s.name, func(b *testing.B) {
					b.ReportAllocs()
					b.SetBytes(int64(payloadSize))
					if s.bound >= 0 {
						runtime.SetMaxPooledBufferCapacity(s.bound)
					}
					for b.Loop() {
						var err error
						if s.bound < 0 {
							err = serializer.EncodeWithAllocator(podList, io.Discard, &runtime.SimpleAllocator{})
						} else {
							allocator := runtime.AllocatorPool.Get().(*runtime.Allocator)
							err = serializer.EncodeWithAllocator(podList, io.Discard, allocator)
							runtime.PutAllocator(allocator)
						}
						if err != nil {
							b.Fatalf("unexpected encode error: %v", err)
						}
					}
				})
			}
		})
	}
}
