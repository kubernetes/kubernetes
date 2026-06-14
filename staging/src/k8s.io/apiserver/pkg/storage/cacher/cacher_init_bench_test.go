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
	goruntime "runtime"
	"strings"
	"syscall"
	"testing"
	"time"

	"golang.org/x/sync/errgroup"
	"sigs.k8s.io/yaml"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storage/etcd3/testserver"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/utils/clock"
)

func BenchmarkCacherInit(b *testing.B) {
	const pods = 150_000

	ctx := context.Background()

	server, etcdStorage := newCorev1EtcdTestStorage(b)
	b.Cleanup(func() { server.Terminate(b) })

	seedCorev1PodsParallel(b, ctx, etcdStorage, 1, pods, 32)

	config := Config{
		Storage:             etcdStorage,
		Versioner:           storage.APIObjectVersioner{},
		GroupResource:       schema.GroupResource{Resource: "pods"},
		EventsHistoryWindow: DefaultEventFreshDuration,
		ResourcePrefix:      "/pods/",
		KeyFunc: func(obj runtime.Object) (string, error) {
			return storage.NamespaceKeyFunc("/pods/", obj)
		},
		GetAttrsFunc: getCorev1PodAttrs,
		NewFunc:      func() runtime.Object { return &corev1.Pod{} },
		NewListFunc:  func() runtime.Object { return &corev1.PodList{} },
		Codec:        corev1ProtoCodec,
		Clock:        clock.RealClock{},
	}

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
	if err := yaml.Unmarshal(data, &pod); err != nil {
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

const (
	crGroup    = "example.com"
	crVersion  = "v1"
	crKind     = "Widget"
	crResource = "widgets"

	crAPIVersion = crGroup + "/" + crVersion
	crKeyPrefix  = "/" + crResource + "/"
)

func rusageSeconds(ru syscall.Rusage) float64 {
	cpu := func(t syscall.Timeval) float64 { return float64(t.Sec) + float64(t.Usec)/1e6 }
	return cpu(ru.Utime) + cpu(ru.Stime)
}

// makeCR builds a generic unstructured custom resource of groups x fields padded
// entries. The default weights mirror the kueue Workload that motivated the gate
// (kubernetes#136950).
func makeCR(idx, groups, fields int) *unstructured.Unstructured {
	items := make([]any, 0, groups)
	for g := range groups {
		entries := make([]any, 0, fields)
		for f := range fields {
			entries = append(entries, map[string]any{
				"name":  fmt.Sprintf("k%d_%d", g, f),
				"value": strings.Repeat("x", 100),
			})
		}
		items = append(items, map[string]any{"name": fmt.Sprintf("g%d", g), "entries": entries})
	}
	return &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": crAPIVersion,
		"kind":       crKind,
		"metadata":   map[string]any{"name": fmt.Sprintf("cr-%d", idx), "namespace": "default"},
		"spec":       map[string]any{"groups": items},
	}}
}

func seedCRsParallel(b *testing.B, ctx context.Context, etcdStorage storage.Interface, count, groups, fields, workers int) {
	eg, gctx := errgroup.WithContext(ctx)
	idxCh := make(chan int, workers*4)

	for range workers {
		eg.Go(func() error {
			out := &unstructured.Unstructured{}
			for idx := range idxCh {
				w := makeCR(idx, groups, fields)
				key := crKeyPrefix + "default/" + w.GetName()
				if err := etcdStorage.Create(gctx, key, w, out, 0); err != nil {
					return err
				}
			}
			return nil
		})
	}

	eg.Go(func() error {
		defer close(idxCh)
		for i := range count {
			select {
			case idxCh <- i:
			case <-gctx.Done():
				return gctx.Err()
			}
		}
		return nil
	})

	if err := eg.Wait(); err != nil {
		b.Fatalf("seed: %v", err)
	}
}

// BenchmarkCacherInitConcurrentDecode measures cold watch-cache init for a
// JSON/unstructured custom resource with ConcurrentWatchObjectDecode off vs on.
// JSON decode is heavy enough (unlike protobuf Pods) to show the gate's effect.
func BenchmarkCacherInitConcurrentDecode(b *testing.B) {
	ctx := context.Background()
	codec := unstructured.UnstructuredJSONScheme
	versioner := storage.APIObjectVersioner{}
	gr := schema.GroupResource{Group: crGroup, Resource: crResource}
	newFunc := func() runtime.Object {
		u := &unstructured.Unstructured{}
		u.SetAPIVersion(crAPIVersion)
		u.SetKind(crKind)
		return u
	}
	newListFunc := func() runtime.Object {
		u := &unstructured.UnstructuredList{}
		u.SetAPIVersion(crAPIVersion)
		u.SetKind(crKind + "List")
		return u
	}

	cases := []struct {
		name                  string
		count, groups, fields int
	}{
		{"weight=medium/n=10k", 10_000, 10, 10}, // ~14 KB each
		{"weight=heavy/n=10k", 10_000, 60, 20},  // ~160 KB each (kueue Workload weight)
		{"weight=medium/n=50k", 50_000, 10, 10}, // more items
	}

	for _, tc := range cases {
		cfg := testserver.NewTestConfig(b)
		cfg.QuotaBackendBytes = 8 << 30 // headroom for heavy multi-GB seeds
		server := &etcd3testing.EtcdTestServer{V3Client: testserver.RunEtcd(b, cfg)}
		compactor := etcd3.NewCompactor(server.V3Client.Client, 0, clock.RealClock{}, nil)
		etcdStorage, err := etcd3.New(
			server.V3Client, compactor, codec, newFunc, newListFunc,
			etcd3testing.PathPrefix(), crKeyPrefix, gr,
			identity.NewEncryptCheckTransformer(), etcd3.NewDefaultLeaseManagerConfig(),
			etcd3.NewDefaultDecoder(codec, versioner), versioner)
		if err != nil {
			b.Fatal(err)
		}
		seedCRsParallel(b, ctx, etcdStorage, tc.count, tc.groups, tc.fields, 32)

		config := Config{
			Storage:             etcdStorage,
			Versioner:           versioner,
			GroupResource:       gr,
			EventsHistoryWindow: DefaultEventFreshDuration,
			ResourcePrefix:      crKeyPrefix,
			KeyFunc: func(obj runtime.Object) (string, error) {
				return storage.NamespaceKeyFunc(crKeyPrefix, obj)
			},
			GetAttrsFunc: storage.DefaultNamespaceScopedAttr,
			NewFunc:      newFunc,
			NewListFunc:  newListFunc,
			Codec:        codec,
			Clock:        clock.RealClock{},
		}

		for _, concurrent := range []bool{false, true} {
			b.Run(fmt.Sprintf("%s/Concurrent=%v", tc.name, concurrent), func(b *testing.B) {
				featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, features.ConcurrentWatchObjectDecode, concurrent)

				b.ResetTimer()
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					// Peak heap + CPU during init. ReadMemStats sampling adds fixed
					// overhead to sec/op equally for both arms; the delta is unaffected.
					var peak uint64
					stop, done := make(chan struct{}), make(chan struct{})
					go func() {
						defer close(done)
						var m goruntime.MemStats
						tk := time.NewTicker(50 * time.Millisecond)
						defer tk.Stop()
						for {
							select {
							case <-stop:
								return
							case <-tk.C:
								goruntime.ReadMemStats(&m)
								if m.HeapInuse > peak {
									peak = m.HeapInuse
								}
							}
						}
					}()
					var ru0, ru1 syscall.Rusage
					_ = syscall.Getrusage(syscall.RUSAGE_SELF, &ru0)
					wallStart := time.Now()

					cacher, err := NewCacherFromConfig(config)
					if err != nil {
						b.Fatal(err)
					}
					if err := cacher.Wait(ctx); err != nil {
						b.Fatal(err)
					}

					wall := time.Since(wallStart)
					_ = syscall.Getrusage(syscall.RUSAGE_SELF, &ru1)
					close(stop)
					<-done

					b.StopTimer()
					b.ReportMetric(float64(peak)/(1<<20), "peakHeapMB")
					b.ReportMetric((rusageSeconds(ru1)-rusageSeconds(ru0))/wall.Seconds(), "cpu-cores")
					cacher.Stop()
					etcd3.TestOnlyResetResourceSizeEstimator(etcdStorage)
					b.StartTimer()
				}
				b.ReportMetric(float64(tc.count), "objs/cache")
			})
		}

		etcdStorage.Close()
		compactor.Stop()
		server.Terminate(b)
	}
}
