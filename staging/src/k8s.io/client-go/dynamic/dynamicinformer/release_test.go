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

package dynamicinformer

import (
	"context"
	"sync"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/tools/cache"
)

var releaseGVR = schema.GroupVersionResource{Group: "example.com", Version: "v1", Resource: "widgets"}

func waitStopped(t *testing.T, informer cache.SharedIndexInformer) bool {
	t.Helper()
	err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
		return informer.IsStopped(), nil
	})
	return err == nil
}

// TestReleaseStopsInformerAfterLastHolder covers the contract of Release on a factory shared by
// several callers: the informer keeps running until every ForResource caller released it, a
// later ForResource builds a fresh informer, and releasing what is not held is harmless.
func TestReleaseStopsInformerAfterLastHolder(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	factory := NewDynamicSharedInformerFactory(fake.NewSimpleDynamicClientWithCustomListKinds(runtime.NewScheme(), map[schema.GroupVersionResource]string{releaseGVR: "WidgetList"}), 0)

	// two callers, as the garbage collector and the quota controller in kube-controller-manager
	first := factory.ForResource(releaseGVR).Informer()
	second := factory.ForResource(releaseGVR).Informer()
	if first != second {
		t.Fatalf("expected both callers to share one informer")
	}
	factory.Start(ctx.Done())
	if !cache.WaitForCacheSync(ctx.Done(), first.HasSynced) {
		t.Fatalf("informer did not sync")
	}

	if stopped := factory.Release(releaseGVR); stopped {
		t.Fatalf("first Release stopped the informer while another caller still held it")
	}
	time.Sleep(50 * time.Millisecond)
	if first.IsStopped() {
		t.Fatalf("informer stopped while another caller still held it")
	}

	if stopped := factory.Release(releaseGVR); !stopped {
		t.Fatalf("last Release did not report stopping the informer")
	}
	if !waitStopped(t, first) {
		t.Fatalf("informer still running after the last caller released it")
	}

	// a later ForResource builds a fresh informer and Start runs it
	fresh := factory.ForResource(releaseGVR).Informer()
	if fresh == first {
		t.Fatalf("ForResource returned the stopped informer")
	}
	factory.Start(ctx.Done())
	if !cache.WaitForCacheSync(ctx.Done(), fresh.HasSynced) {
		t.Fatalf("fresh informer did not sync")
	}
	if fresh.IsStopped() {
		t.Fatalf("fresh informer is stopped")
	}
	factory.Release(releaseGVR)

	// releasing what is not held is a no-op
	if factory.Release(releaseGVR) {
		t.Fatalf("Release of an already released resource reported a stop")
	}
	if factory.Release(schema.GroupVersionResource{Group: "example.com", Version: "v1", Resource: "unknown"}) {
		t.Fatalf("Release of an unknown resource reported a stop")
	}

	cancel()
	factory.Shutdown()
}

// TestReleaseConcurrent runs ForResource, Start and Release for the same resource concurrently;
// the factory must neither race nor deadlock, and Shutdown must return once the context is done.
func TestReleaseConcurrent(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	factory := NewDynamicSharedInformerFactory(fake.NewSimpleDynamicClientWithCustomListKinds(runtime.NewScheme(), map[schema.GroupVersionResource]string{releaseGVR: "WidgetList"}), 0)
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(3)
		go func() { defer wg.Done(); _ = factory.ForResource(releaseGVR).Informer() }()
		go func() { defer wg.Done(); factory.Start(ctx.Done()) }()
		go func() { defer wg.Done(); _ = factory.Release(releaseGVR) }()
	}
	wg.Wait()
	cancel()
	done := make(chan struct{})
	go func() { factory.Shutdown(); close(done) }()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatalf("Shutdown did not return after the context was cancelled")
	}
}
