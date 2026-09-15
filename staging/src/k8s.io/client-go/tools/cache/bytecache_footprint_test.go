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

package cache_test

// Memory footprint measurement for informer caches — run with:
//
//	go test -run TestInformerCacheMemoryFootprint -v ./tools/cache/
//
// Measures heap attributable to a real SharedIndexInformer at rest, with the
// default threadSafeMap and with the experimental bytecache store
// (KUBE_BYTECACHE=1). There is no other in-tree measurement of informer
// memory footprint today.

import (
	"context"
	"fmt"
	"runtime"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	fcache "k8s.io/client-go/tools/cache/testing"
)

func footprintPod(i int) *corev1.Pod {
	name := fmt.Sprintf("workload-%d", i)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: fmt.Sprintf("team-%d", i%50),
			UID:       "11111111-2222-3333-4444-555555555555",
			Labels: map[string]string{
				"app":                         name,
				"pod-template-hash":           "5f7b8c9d",
				"app.kubernetes.io/name":      name,
				"topology.kubernetes.io/zone": "us-east-1a",
			},
			Annotations: map[string]string{
				"kubectl.kubernetes.io/last-applied-configuration": `{"apiVersion":"v1","kind":"Pod","metadata":{"labels":{"app":"x"}},"spec":{"containers":[{"image":"nginx","name":"nginx"}]}}`,
				"prometheus.io/scrape":                             "true",
			},
		},
		Spec: corev1.PodSpec{
			NodeName:           fmt.Sprintf("node-%d", i%500),
			ServiceAccountName: "default",
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			PodIP: fmt.Sprintf("10.%d.%d.%d", i%250, (i/250)%250, i%250),
		},
	}
	for c := range 2 {
		container := corev1.Container{
			Name:  fmt.Sprintf("container-%d", c),
			Image: fmt.Sprintf("registry.example.com/app/%s:v1.%d", name, c),
			Resources: corev1.ResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceCPU:    resource.MustParse("100m"),
					corev1.ResourceMemory: resource.MustParse("128Mi"),
				},
			},
		}
		for e := range 8 {
			container.Env = append(container.Env, corev1.EnvVar{
				Name:  fmt.Sprintf("APP_SETTING_%d", e),
				Value: fmt.Sprintf("value-%d-%d", c, e),
			})
		}
		pod.Spec.Containers = append(pod.Spec.Containers, container)
	}
	return pod
}

func TestInformerCacheMemoryFootprint(t *testing.T) {
	if testing.Short() {
		t.Skip("footprint measurement, skipped in short mode")
	}
	const n = 20000
	for _, mode := range []string{"0", "1", "proto", "gob"} {
		name := "default"
		switch mode {
		case "1":
			name = "bytecache-reloc"
		case "proto", "gob":
			name = "bytecache-" + mode
		}
		t.Run(name, func(t *testing.T) {
			t.Setenv("KUBE_BYTECACHE", mode)

			source := fcache.NewFakeControllerSource()
			defer source.Shutdown()
			for i := range n {
				source.Add(footprintPod(i))
			}
			runtime.GC()
			runtime.GC()
			var base runtime.MemStats
			runtime.ReadMemStats(&base)

			informer := cache.NewSharedIndexInformer(source, &corev1.Pod{}, 0,
				cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
			stop := make(chan struct{})
			defer close(stop)
			go informer.Run(stop)
			if !cache.WaitForCacheSync(stop, informer.HasSynced) {
				t.Fatal("sync failed")
			}
			if err := wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 30*time.Second, true,
				func(context.Context) (bool, error) {
					return len(informer.GetStore().ListKeys()) == n, nil
				}); err != nil {
				t.Fatalf("never saw %d keys: %v", n, err)
			}

			// Exercise the read path a consumer would.
			obj, ok, err := informer.GetStore().GetByKey("team-7/workload-7")
			if err != nil || !ok {
				t.Fatalf("get: %v %v", ok, err)
			}
			pod := obj.(*corev1.Pod)
			if pod.Labels["app"] != "workload-7" || len(pod.Spec.Containers[0].Env) != 8 {
				t.Fatal("bad pod content")
			}
			byNS, err := informer.GetIndexer().ByIndex(cache.NamespaceIndex, "team-7")
			if err != nil || len(byNS) != n/50 {
				t.Fatalf("ByIndex: %d, %v", len(byNS), err)
			}

			runtime.GC()
			runtime.GC()
			var after runtime.MemStats
			runtime.ReadMemStats(&after)
			start := time.Now()
			for range 3 {
				runtime.GC()
			}
			gcTime := time.Since(start) / 3
			t.Logf("%s: informer-attributable heap %+.1f MB, heapObjects %+d, full GC %v",
				name, float64(after.HeapAlloc-base.HeapAlloc)/1e6,
				int64(after.HeapObjects)-int64(base.HeapObjects), gcTime)
		})
	}
}
