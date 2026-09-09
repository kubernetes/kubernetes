/*
Copyright 2025 The Kubernetes Authors.

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

package cache

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/watch"
)

var benchmarkNamespace = "default"

func BenchmarkSharedIndexInformer(b *testing.B) {
	for _, podCount := range []int{10_000, 20_000, 40_000, 80_000, 160_000} {
		b.Run(fmt.Sprintf("podCount=%d", podCount), func(b *testing.B) {
			pods := createPods(podCount, benchmarkNamespace)
			for _, readers := range []int{0, 1, 10, 20, 40, 80} {
				b.Run(fmt.Sprintf("readers=%d", readers), func(b *testing.B) {
					watcher := watch.NewFakeWithChanSize(1, false)
					informer, stop := setupSharedIndexInformer(watcher, pods)
					defer stop()
					queuedEvents := 10
					benchmarkSharedIndexInformer(b, readers, watcher, informer, pods, queuedEvents)
				})
			}
		})
	}
}

func createPods(count int, namespace string) []corev1.Pod {
	pods := []corev1.Pod{}
	for i := 0; i < count; i++ {
		pod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      rand.String(20),
				Namespace: namespace,
			},
		}
		pods = append(pods, *pod)
	}
	return pods
}

func setupSharedIndexInformer(watcher watch.Interface, pods []corev1.Pod) (SharedIndexInformer, func()) {
	podInformer := NewSharedIndexInformerWithOptions(
		&ListWatch{
			ListWithContextFunc: func(ctx context.Context, options metav1.ListOptions) (runtime.Object, error) {
				return &corev1.PodList{
					Items: pods,
				}, nil
			},
			WatchFuncWithContext: func(ctx context.Context, options metav1.ListOptions) (watch.Interface, error) {
				return watcher, nil
			},
		},
		&corev1.Pod{},
		SharedIndexInformerOptions{
			ResyncPeriod: time.Second * 60,
			Indexers:     Indexers{NamespaceIndex: MetaNamespaceIndexFunc},
		},
	)

	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		podInformer.Run(stop)
	}()
	for !podInformer.HasSynced() {
		time.Sleep(time.Millisecond)
	}
	return podInformer, func() {
		close(stop)
		wg.Wait()
	}
}

func benchmarkSharedIndexInformer(b *testing.B, readers int, watcher *watch.FakeWatcher, podInformer SharedIndexInformer, pods []corev1.Pod, queuedEvents int) {
	var writes atomic.Int64
	got := make(chan struct{}, queuedEvents)

	_, err := podInformer.AddEventHandler(ResourceEventHandlerDetailedFuncs{
		UpdateFunc: func(oldObj, newObj interface{}) {
			writes.Add(1)
			select {
			case <-got:
			default:
			}
		},
	})
	if err != nil {
		b.Fatal(err)
	}
	pod := pods[rand.Intn(len(pods))]
	watcher.Modify(&pod)
	// Wait for modify to arrive to confirm initial pods where handled.
	for writes.Load() == 0 {
		time.Sleep(time.Millisecond)
	}

	var wg sync.WaitGroup
	stop := make(chan struct{})
	var reads atomic.Int64
	for i := 0; i < readers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				err := ListAllByNamespace(podInformer.GetIndexer(), benchmarkNamespace, labels.Everything(), func(obj interface{}) {
				})
				if err != nil {
					panic(err)
				}
				reads.Add(1)
			}
		}()
	}

	writes.Store(0)
	reads.Store(0)
	for b.Loop() {
		pod := pods[rand.Intn(len(pods))]
		watcher.Modify(&pod)
		got <- struct{}{}
	}
	b.ReportMetric(float64(reads.Load())/b.Elapsed().Seconds(), "reads/s")
	b.ReportMetric(float64(writes.Load())/b.Elapsed().Seconds(), "writes/s")
	close(stop)
	wg.Wait()
}
