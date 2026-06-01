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

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3"
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
