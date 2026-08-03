/*
Copyright 2026 The Kubernetes Authors.

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

package apiserver

import (
	"runtime/debug"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// maxPooledBufferCapacity mirrors the unexported bound in
// k8s.io/apimachinery/pkg/runtime/allocator.go. If the bound there changes,
// update this constant.
const maxPooledBufferCapacity = 512 * 1024

// TestAllocatorPoolBufferCapacityBounded verifies end to end that serving
// protobuf responses larger than maxPooledBufferCapacity does not leave
// oversized encode buffers parked in runtime.AllocatorPool. Before the cap was
// introduced, every allocator that ever encoded a large object kept its grown
// buffer for the life of the process; on a GKE control plane this pinned
// ~49MB of heap.
//
// The test server runs in-process, so the pool under test is shared with the
// serving path. The test drives GETs of a ConfigMap well above the cap and a
// ConfigMap well below it through the real protobuf serialization stack
// (SerializeObject only uses AllocatorPool for encoders that implement
// EncoderWithAllocator, which protobuf does and JSON does not), then drains
// the pool and inspects the retained buffer capacities:
//
//   - No drained allocator may retain more than maxPooledBufferCapacity.
//   - At least one drained allocator must retain a buffer at least as large
//     as the small ConfigMap's payload, proving the pool is still actually
//     reusing buffers for sub-cap responses (i.e. the invariant above is not
//     passing vacuously because pooling broke entirely).
func TestAllocatorPoolBufferCapacityBounded(t *testing.T) {
	tCtx := ktesting.Init(t)
	client, kubeConfig, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{})
	defer tearDownFn()

	const (
		bigPayload   = 900 * 1024 // well above maxPooledBufferCapacity, under the 1MiB ConfigMap limit
		smallPayload = 64 * 1024  // well below maxPooledBufferCapacity
	)

	for name, size := range map[string]int{"big": bigPayload, "small": smallPayload} {
		cm := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: name},
			Data:       map[string]string{"blob": strings.Repeat("x", size)},
		}
		if _, err := client.CoreV1().ConfigMaps("default").Create(tCtx, cm, metav1.CreateOptions{}); err != nil {
			t.Fatalf("failed to create ConfigMap %q: %v", name, err)
		}
	}

	pbConfig := restclient.CopyConfig(kubeConfig)
	pbConfig.ContentType = "application/vnd.kubernetes.protobuf"
	pbClient, err := clientset.NewForConfig(pbConfig)
	if err != nil {
		t.Fatal(err)
	}

	// sync.Pool contents are cleared by GC. Disable GC so allocators returned
	// to the pool by the serving path reliably survive until we drain and
	// inspect them.
	defer debug.SetGCPercent(debug.SetGCPercent(-1))

	sawPooledSmallBuffer := false
	for round := 0; round < 10 && !sawPooledSmallBuffer; round++ {
		for i := 0; i < 5; i++ {
			for _, name := range []string{"big", "small"} {
				if _, err := pbClient.CoreV1().ConfigMaps("default").Get(tCtx, name, metav1.GetOptions{}); err != nil {
					t.Fatalf("failed to get ConfigMap %q: %v", name, err)
				}
			}
		}

		// Drain the pool. Allocators the serving path returned are eligible to
		// be handed back to us; Get also returns fresh zero-capacity allocators
		// once the pool is empty, which are harmless to inspect.
		drained := make([]*k8sruntime.Allocator, 0, 64)
		for i := 0; i < 64; i++ {
			drained = append(drained, k8sruntime.AllocatorPool.Get().(*k8sruntime.Allocator))
		}
		for _, a := range drained {
			// Allocate(0) exposes the retained buffer without growing it.
			retained := cap(a.Allocate(0))
			if retained > maxPooledBufferCapacity {
				t.Fatalf("AllocatorPool retained a buffer of %d bytes, want at most %d: encode buffers above the cap must be dropped, not pooled", retained, maxPooledBufferCapacity)
			}
			if retained >= smallPayload {
				sawPooledSmallBuffer = true
			}
		}
		for _, a := range drained {
			k8sruntime.PutAllocator(a)
		}
	}

	if !sawPooledSmallBuffer {
		t.Fatalf("never drained an allocator retaining at least %d bytes; the pool does not appear to be reusing buffers for sub-cap responses, so this test has lost its signal", smallPayload)
	}
}
