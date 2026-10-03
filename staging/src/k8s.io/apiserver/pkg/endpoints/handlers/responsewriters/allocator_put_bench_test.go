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
// PutAllocator's buffer-capacity cap on the non-streaming protobuf encode
// path (the one SerializeObject uses for single-object GET responses and
// non-streamed LISTs, where the allocator buffer holds the entire serialized
// response). It compares, on the same binary:
//
//   - Put=Uncapped: the pre-cap behavior (AllocatorPool.Put directly), where
//     a pooled buffer grows to the largest response and is reused verbatim.
//   - Put=Capped: the current behavior (PutAllocator), which drops buffers
//     larger than the pool's capacity bound, so every over-cap encode pays a
//     fresh buffer allocation.
//   - Put=None: no pooling at all (SimpleAllocator), as a floor reference.
//
// The Uncapped→Capped delta on payloads above the cap is the CPU/alloc price
// of the bounded pool; payloads below the cap must show no difference.
func BenchmarkProtobufEncodeAllocatorPutStrategies(b *testing.B) {
	// Non-streaming serializer: one Allocate for the whole response.
	serializer := protobuf.NewSerializerWithOptions(nil, nil, protobuf.SerializerOptions{})

	strategies := []struct {
		name string
		put  func(runtime.MemoryAllocator)
	}{
		{name: "Put=Uncapped", put: func(a runtime.MemoryAllocator) { runtime.AllocatorPool.Put(a) }},
		{name: "Put=Capped", put: runtime.PutAllocator},
		{name: "Put=None", put: nil},
	}

	for _, count := range []int{50, 500, 5_000} {
		podList := benchmarkItems(b, count)
		payloadSize := podList.Size()
		b.Run(fmt.Sprintf("Pods=%d/PayloadKiB=%d", count, payloadSize/1024), func(b *testing.B) {
			for _, strategy := range strategies {
				b.Run(strategy.name, func(b *testing.B) {
					b.ReportAllocs()
					b.SetBytes(int64(payloadSize))
					for b.Loop() {
						var err error
						if strategy.put == nil {
							err = serializer.EncodeWithAllocator(podList, io.Discard, &runtime.SimpleAllocator{})
						} else {
							allocator := runtime.AllocatorPool.Get().(*runtime.Allocator)
							err = serializer.EncodeWithAllocator(podList, io.Discard, allocator)
							strategy.put(allocator)
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
