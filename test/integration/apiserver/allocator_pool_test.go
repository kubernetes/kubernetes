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

package apiserver

import (
	"runtime/debug"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/server"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// TestAllocatorPoolBufferCapacityBounded verifies end to end that the
// AllocatorPoolBufferCap feature gate and the --max-pooled-encode-buffer-size
// flag bound the encode buffers parked in runtime.AllocatorPool. Before the
// bound was introduced, every allocator that ever encoded a large object kept
// its grown buffer for the life of the process; on a GKE control plane this
// pinned ~49MB of heap.
//
// The test server runs in-process, so the pool under test is shared with the
// serving path. Each case drives GETs of ConfigMaps above and below the
// configured bound through the real protobuf serialization stack
// (SerializeObject only uses AllocatorPool for encoders that implement
// EncoderWithAllocator, which protobuf does and JSON does not), then drains
// the pool and inspects the retained buffer capacities:
//
//   - With the bound in effect, no drained allocator may retain more than the
//     bound, and at least one must retain a buffer at least as large as the
//     small payload, proving the pool still reuses buffers for sub-bound
//     responses (i.e. the invariant is not passing vacuously because pooling
//     broke entirely).
//   - With the gate disabled the pool is unbounded, so at least one drained
//     allocator must retain the big payload's buffer.
func TestAllocatorPoolBufferCapacityBounded(t *testing.T) {
	const (
		bigPayload   = 900 * 1024 // above every bound below, under the 1MiB ConfigMap limit
		smallPayload = 64 * 1024  // below every bound below
	)

	cases := []struct {
		name        string
		gateEnabled bool
		setup       framework.TestServerSetup
		// bound is the expected effective pool bound; 0 means unbounded.
		bound int
	}{
		{
			name:        "gate enabled with default size",
			gateEnabled: true,
			bound:       server.DefaultMaxPooledEncodeBufferSize,
		},
		{
			name:        "gate enabled with explicit size",
			gateEnabled: true,
			setup: framework.TestServerSetup{
				ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
					opts.GenericServerRunOptions.MaxPooledEncodeBufferSize = 128 * 1024
				},
			},
			bound: 128 * 1024,
		},
		{
			name:        "gate disabled",
			gateEnabled: false,
			bound:       0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AllocatorPoolBufferCap, tc.gateEnabled)
			tCtx := ktesting.Init(t)
			client, kubeConfig, tearDownFn := framework.StartTestServer(tCtx, t, tc.setup)
			defer tearDownFn()

			if got := k8sruntime.MaxPooledBufferCapacity(); got != tc.bound {
				t.Fatalf("effective pool bound is %d, want %d", got, tc.bound)
			}

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

			// sync.Pool contents are cleared by GC. Disable GC so allocators
			// returned to the pool by the serving path reliably survive until
			// we drain and inspect them.
			defer debug.SetGCPercent(debug.SetGCPercent(-1))

			// The pool must keep reusing buffers for responses below the bound
			// (or for everything, when unbounded); the big buffer must be
			// retained only when unbounded.
			wantRetainedAtLeast := smallPayload
			if tc.bound == 0 {
				wantRetainedAtLeast = bigPayload
			}
			sawExpectedBuffer := false
			for round := 0; round < 10 && !sawExpectedBuffer; round++ {
				for range 5 {
					for _, name := range []string{"big", "small"} {
						if _, err := pbClient.CoreV1().ConfigMaps("default").Get(tCtx, name, metav1.GetOptions{}); err != nil {
							t.Fatalf("failed to get ConfigMap %q: %v", name, err)
						}
					}
				}

				// Drain the pool. Allocators the serving path returned are
				// eligible to be handed back to us; Get also returns fresh
				// zero-capacity allocators once the pool is empty, which are
				// harmless to inspect.
				drained := make([]*k8sruntime.Allocator, 0, 64)
				for range 64 {
					drained = append(drained, k8sruntime.AllocatorPool.Get().(*k8sruntime.Allocator))
				}
				for _, a := range drained {
					// Allocate(0) exposes the retained buffer without growing it.
					retained := cap(a.Allocate(0))
					if tc.bound > 0 && retained > tc.bound {
						t.Fatalf("AllocatorPool retained a buffer of %d bytes, want at most %d: encode buffers above the bound must be dropped, not pooled", retained, tc.bound)
					}
					if retained >= wantRetainedAtLeast {
						sawExpectedBuffer = true
					}
				}
				for _, a := range drained {
					k8sruntime.PutAllocator(a)
				}
			}

			if !sawExpectedBuffer {
				t.Fatalf("never drained an allocator retaining at least %d bytes; the pool does not appear to be reusing buffers as expected, so this test has lost its signal", wantRetainedAtLeast)
			}
		})
	}
}
