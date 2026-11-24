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

package cacher

import (
	"context"
	"fmt"
	"os"
	"testing"

	"golang.org/x/sync/errgroup"
	"sigs.k8s.io/yaml"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

func BenchmarkCacherInit(b *testing.B) {
	const pods = 150_000

	ctx := context.Background()

	server, etcdStorage := benchmarkEtcdTestStorage(b)
	b.Cleanup(func() { server.Terminate(b) })

	seedCorev1PodsParallel(b, ctx, etcdStorage, 1, pods, 32)

	config := benchmarkConfig(etcdStorage)

	for _, rangeStream := range []bool{false, true} {
		for _, concurrentDecode := range []bool{false, true} {
			name := fmt.Sprintf("RangeStream=%v/ConcurrentDecode=%v", rangeStream, concurrentDecode)
			b.Run(name, func(b *testing.B) {
				featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, features.EtcdRangeStream, rangeStream)
				featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, features.ConcurrentWatchObjectDecode, concurrentDecode)

				b.ResetTimer()
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					cacher, err := NewCacherFromConfig(config)
					if err != nil {
						b.Fatal(err)
					}
					if err := cacher.Wait(ctx); err != nil {
						b.Fatal(err)
					}
					b.StopTimer()
					cacher.Stop()
					etcd3.TestOnlyResetResourceSizeEstimator(etcdStorage)
					b.StartTimer()
				}
				b.ReportMetric(float64(pods), "pods/cache")
			})
		}
	}
}

func loadExemplarPod(b *testing.B) *corev1.Pod {
	const path = "testdata/exemplar_pod.yaml"
	data, err := os.ReadFile(path)
	if err != nil {
		b.Fatalf("read %q: %v", path, err)
	}
	var pod corev1.Pod
	if err := yaml.UnmarshalStrict(data, &pod); err != nil {
		b.Fatalf("decode %q: %v", path, err)
	}
	return &pod
}

func randomizeCorev1Pod(pod *corev1.Pod, ns string) {
	pod.Namespace = ns
	pod.Name = pod.GenerateName + utilrand.String(10)
	pod.UID = uuid.NewUUID()
	pod.ResourceVersion = ""
	pod.Spec.NodeName = "some-node-prefix-" + utilrand.String(6)
}

func seedCorev1PodsParallel(b *testing.B, ctx context.Context, etcdStorage storage.Interface, namespaces, podsPerNS, workers int) {
	exemplar := loadExemplarPod(b)
	g, gctx := errgroup.WithContext(ctx)
	pods := make(chan *corev1.Pod, workers*4)

	for range workers {
		g.Go(func() error {
			var out corev1.Pod
			for pod := range pods {
				key := fmt.Sprintf("/pods/%s/%s", pod.Namespace, pod.Name)
				if err := etcdStorage.Create(gctx, key, pod, &out, 0); err != nil {
					return err
				}
			}
			return nil
		})
	}

	g.Go(func() error {
		defer close(pods)
		for range namespaces {
			ns := utilrand.String(10)
			for range podsPerNS {
				p := exemplar.DeepCopy()
				randomizeCorev1Pod(p, ns)
				select {
				case pods <- p:
				case <-gctx.Done():
					return gctx.Err()
				}
			}
		}
		return nil
	})

	if err := g.Wait(); err != nil {
		b.Fatalf("seed: %v", err)
	}
}

func benchmarkSetup(b *testing.B) (context.Context, *CacheDelegator) {
	server, etcdStorage := benchmarkEtcdTestStorage(b)
	config := benchmarkConfig(etcdStorage)
	cacher, err := NewCacherFromConfig(config)
	if err != nil {
		b.Fatalf("Failed to initialize cacher: %v", err)
	}
	ctx := context.Background()
	if err := cacher.Wait(ctx); err != nil {
		b.Fatal(err)
	}
	delegator := NewCacheDelegator(cacher, etcdStorage)
	b.Cleanup(func() {
		delegator.Stop()
		cacher.Stop()
		server.Terminate(b)
	})
	return ctx, delegator
}

func benchmarkConfig(etcdStorage storage.Interface) Config {
	config := storagetesting.StoreConfigForBenchmarks()
	return Config{
		Storage:             etcdStorage,
		Versioner:           config.Versioner,
		GroupResource:       config.GroupResource,
		EventsHistoryWindow: DefaultEventFreshDuration,
		ResourcePrefix:      config.ResourcePrefix,
		KeyFunc:             config.KeyFunc,
		GetAttrsFunc:        config.GetAttrsFunc,
		NewFunc:             config.NewFunc,
		NewListFunc:         config.NewListFunc,
		IndexerFuncs: map[string]storage.IndexerFunc{
			"spec.nodeName": func(obj runtime.Object) string {
				pod, ok := obj.(*corev1.Pod)
				if !ok {
					return ""
				}
				return pod.Spec.NodeName
			},
		},
		Indexers: &cache.Indexers{
			"f:spec.nodeName": func(obj interface{}) ([]string, error) {
				pod, ok := obj.(*corev1.Pod)
				if !ok {
					return nil, fmt.Errorf("not a pod")
				}
				return []string{pod.Spec.NodeName}, nil
			},
		},
		Codec: config.Codec,
		Clock: clock.RealClock{},
	}
}

func BenchmarkStoreWriteThroughput(b *testing.B) {
	klog.SetLogger(logr.Discard())
	dimensions := []struct {
		namespaceCount       int
		podPerNamespaceCount int
		nodeCount            int
	}{
		{
			namespaceCount:       50,
			podPerNamespaceCount: 3_000,
			nodeCount:            5_000,
		},
	}
	for _, dims := range dimensions {
		b.Run(fmt.Sprintf("Namespaces=%d/Pods=%d/Nodes=%d", dims.namespaceCount, dims.namespaceCount*dims.podPerNamespaceCount, dims.nodeCount), func(b *testing.B) {
			ctx, cacher := benchmarkSetup(b)
			data := storagetesting.PrepareBenchmarkData(dims.namespaceCount, dims.podPerNamespaceCount, dims.nodeCount)
			tracker := storagetesting.NewWatchLatencyTracker(clock.RealClock{})
			originalHandler := cacher.cacher.watchCache.config.eventHandler
			cacher.cacher.watchCache.config.eventHandler = func(event *watchCacheEvent) {
				if originalHandler != nil {
					originalHandler(event)
				}
				tracker.HandleEvent(event.Object)
			}
			b.Cleanup(func() {
				cacher.cacher.watchCache.config.eventHandler = originalHandler
			})
			storagetesting.RunBenchmarkWriteThroughput(ctx, b, cacher, data, true, tracker)
		})
	}
}

func BenchmarkStoreList(b *testing.B) {
	klog.SetLogger(logr.Discard())
	// Based on https://github.com/kubernetes/community/blob/master/sig-scalability/configs-and-limits/thresholds.md
	dimensions := []struct {
		namespaceCount       int
		podPerNamespaceCount int
		nodeCount            int
	}{
		{
			namespaceCount:       10_000,
			podPerNamespaceCount: 15,
			nodeCount:            5_000,
		},
		{
			namespaceCount:       50,
			podPerNamespaceCount: 3_000,
			nodeCount:            5_000,
		},
		{
			namespaceCount:       100,
			podPerNamespaceCount: 1_100,
			nodeCount:            1000,
		},
	}
	for _, dims := range dimensions {
		b.Run(fmt.Sprintf("Namespaces=%d/Pods=%d/Nodes=%d", dims.namespaceCount, dims.namespaceCount*dims.podPerNamespaceCount, dims.nodeCount), func(b *testing.B) {
			data := storagetesting.PrepareBenchmarkData(dims.namespaceCount, dims.podPerNamespaceCount, dims.nodeCount)
			ctx, cacher := benchmarkSetup(b)
			require.NoError(b, storagetesting.PrecreateBenchmarkPods(ctx, cacher, data))
			for _, useIndex := range []bool{true, false} {
				b.Run(fmt.Sprintf("Indexed=%v", useIndex), func(b *testing.B) {
					storagetesting.RunBenchmarkStoreList(ctx, b, cacher, data, useIndex)
				})
			}
		})
	}
}

func BenchmarkStoreStats(b *testing.B) {
	klog.SetLogger(logr.Discard())
	data := storagetesting.PrepareBenchmarkData(50, 3_000, 5_000)
	ctx, cacher := benchmarkSetup(b)
	require.NoError(b, storagetesting.PrecreateBenchmarkPods(ctx, cacher, data))
	storagetesting.RunBenchmarkStoreStats(ctx, b, cacher)
}
