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

package bytecache

// Informer cache memory footprint against a REAL apiserver: pods created via
// the API carry defaulting, managedFields, and status as production objects
// do. One informer per KUBE_BYTECACHE mode syncs the same 10k pods; we
// measure the heap attributable to each.
//
//	go test -v -timeout 30m ./test/integration/bytecache/

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	nPods   = 10000
	nsName  = "footprint"
	workers = 64
)

func apiPod(i int) *corev1.Pod {
	name := fmt.Sprintf("workload-%d", i)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
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
		Spec: corev1.PodSpec{NodeName: fmt.Sprintf("node-%d", i%500)},
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
				Name: fmt.Sprintf("APP_SETTING_%d", e), Value: fmt.Sprintf("value-%d-%d", c, e),
			})
		}
		pod.Spec.Containers = append(pod.Spec.Containers, container)
	}
	return pod
}

func TestInformerFootprintAgainstAPIServer(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	config := restclient.CopyConfig(server.ClientConfig)
	config.QPS = 2000
	config.Burst = 4000
	client := kubernetes.NewForConfigOrDie(config)
	ctx := context.Background()

	if _, err := client.CoreV1().Namespaces().Create(ctx,
		&corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: nsName}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	t.Logf("creating %d pods via the API...", nPods)
	start := time.Now()
	var wg sync.WaitGroup
	errCh := make(chan error, workers)
	for w := range workers {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := w; i < nPods; i += workers {
				if _, err := client.CoreV1().Pods(nsName).Create(ctx, apiPod(i), metav1.CreateOptions{}); err != nil {
					select {
					case errCh <- err:
					default:
					}
					return
				}
			}
		}(w)
	}
	wg.Wait()
	select {
	case err := <-errCh:
		t.Fatal(err)
	default:
	}
	t.Logf("created %d pods in %v", nPods, time.Since(start))

	for _, mode := range []string{"0", "1", "proto"} {
		name := map[string]string{"0": "default", "1": "reloc", "proto": "proto"}[mode]
		t.Run(name, func(t *testing.T) {
			t.Setenv("KUBE_BYTECACHE", mode)
			runtime.GC()
			runtime.GC()
			var base runtime.MemStats
			runtime.ReadMemStats(&base)

			lw := cache.NewListWatchFromClient(client.CoreV1().RESTClient(), "pods", nsName, fields.Everything())
			informer := cache.NewSharedIndexInformer(lw, &corev1.Pod{}, 0,
				cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
			stop := make(chan struct{})
			go informer.Run(stop)
			if !cache.WaitForCacheSync(stop, informer.HasSynced) {
				close(stop)
				t.Fatal("sync failed")
			}
			if err := wait.PollUntilContextTimeout(ctx, 200*time.Millisecond, time.Minute, true,
				func(context.Context) (bool, error) {
					return len(informer.GetStore().ListKeys()) == nPods, nil
				}); err != nil {
				close(stop)
				t.Fatalf("never saw %d keys: %v", nPods, err)
			}

			obj, ok, err := informer.GetStore().GetByKey(nsName + "/workload-7")
			if err != nil || !ok {
				close(stop)
				t.Fatalf("get: %v %v", ok, err)
			}
			pod := obj.(*corev1.Pod)
			if pod.Labels["app"] != "workload-7" || len(pod.ManagedFields) == 0 {
				close(stop)
				t.Fatalf("unexpected pod content (managedFields=%d)", len(pod.ManagedFields))
			}

			runtime.GC()
			runtime.GC()
			var after runtime.MemStats
			runtime.ReadMemStats(&after)
			gcStart := time.Now()
			for range 3 {
				runtime.GC()
			}
			gcTime := time.Since(gcStart) / 3
			t.Logf("%-8s heap %+7.1f MB, heapObjects %+9d, full GC %v (managedFields present: %d entries on sample pod)",
				name, float64(after.HeapAlloc-base.HeapAlloc)/1e6,
				int64(after.HeapObjects)-int64(base.HeapObjects), gcTime, len(pod.ManagedFields))

			close(stop)
		})
	}
}
