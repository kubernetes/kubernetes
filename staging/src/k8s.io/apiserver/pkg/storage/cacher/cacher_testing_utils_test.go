/*
Copyright 2023 The Kubernetes Authors.

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
	"testing"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/protobuf"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	example2v1 "k8s.io/apiserver/pkg/apis/example2/v1"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storage/etcd3/testserver"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
	"k8s.io/utils/clock"
)

var (
	scheme   = runtime.NewScheme()
	codecs   = serializer.NewCodecFactory(scheme)
	errDummy = fmt.Errorf("dummy error")
)

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	utilruntime.Must(example.AddToScheme(scheme))
	utilruntime.Must(examplev1.AddToScheme(scheme))
	utilruntime.Must(example2v1.AddToScheme(scheme))
	utilruntime.Must(corev1.AddToScheme(scheme))
	utilruntime.Must(metav1.AddMetaToScheme(scheme))
	scheme.AddUnversionedTypes(corev1.SchemeGroupVersion, &metav1.Status{})
	pb := protobuf.NewSerializer(scheme, scheme)
	corev1ProtoCodec = codecs.CodecForVersions(pb, pb, schema.GroupVersions{corev1.SchemeGroupVersion}, nil)
	examplev1ProtoCodec = codecs.CodecForVersions(pb, pb, schema.GroupVersions{examplev1.SchemeGroupVersion}, nil)
}

var (
	corev1ProtoCodec    runtime.Codec
	examplev1ProtoCodec runtime.Codec
)

func newPod() runtime.Object     { return &example.Pod{} }
func newPodList() runtime.Object { return &example.PodList{} }

func newEtcdTestStorage(t testing.TB, prefix string) (*etcd3testing.EtcdTestServer, storage.Interface) {
	return newEtcdTestStorageWithCodec(t, prefix, examplev1ProtoCodec)
}

func newEtcdTestStorageWithCodec(t testing.TB, prefix string, codec runtime.Codec) (*etcd3testing.EtcdTestServer, storage.Interface) {
	server, _ := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)
	versioner := storage.APIObjectVersioner{}
	compactor := etcd3.NewCompactor(server.V3Client.Client, 0, clock.RealClock{}, nil)
	t.Cleanup(compactor.Stop)
	storage, err := etcd3.New(
		server.V3Client,
		compactor,
		codec,
		newPod,
		newPodList,
		prefix,
		"/pods/",
		schema.GroupResource{Resource: "pods"},
		identity.NewEncryptCheckTransformer(),
		etcd3.NewDefaultLeaseManagerConfig(),
		etcd3.NewDefaultDecoder(codec, versioner),
		versioner)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(storage.Close)
	return server, storage
}

func computePodKey(obj *example.Pod) string {
	return fmt.Sprintf("/pods/%s/%s", obj.Namespace, obj.Name)
}

func newCorev1EtcdTestStorage(t testing.TB) (*etcd3testing.EtcdTestServer, storage.Interface) {
	cfg := testserver.NewTestConfig(t)
	cfg.QuotaBackendBytes = 4 << 30 // 4 GiB (default 2 GiB is too small for 150k pods)
	server := &etcd3testing.EtcdTestServer{V3Client: testserver.RunEtcd(t, cfg)}
	versioner := storage.APIObjectVersioner{}
	compactor := etcd3.NewCompactor(server.V3Client.Client, 0, clock.RealClock{}, nil)
	t.Cleanup(compactor.Stop)
	s, err := etcd3.New(
		server.V3Client,
		compactor,
		corev1ProtoCodec,
		func() runtime.Object { return &corev1.Pod{} },
		func() runtime.Object { return &corev1.PodList{} },
		etcd3testing.PathPrefix(),
		"/pods/",
		schema.GroupResource{Resource: "pods"},
		identity.NewEncryptCheckTransformer(),
		etcd3.NewDefaultLeaseManagerConfig(),
		etcd3.NewDefaultDecoder(corev1ProtoCodec, versioner),
		versioner)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	return server, s
}

func getCorev1PodAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		return nil, nil, fmt.Errorf("not a pod")
	}
	fs := fields.Set{
		"metadata.name":      pod.Name,
		"metadata.namespace": pod.Namespace,
		"spec.nodeName":      pod.Spec.NodeName,
		"spec.restartPolicy": string(pod.Spec.RestartPolicy),
		"status.phase":       string(pod.Status.Phase),
	}
	return labels.Set(pod.Labels), fs, nil
}

func compactWatch(c *CacheDelegator, client *clientv3.Client) storagetesting.Compaction {
	return func(ctx context.Context, t *testing.T, resourceVersion string) {
		versioner := storage.APIObjectVersioner{}
		rv, err := versioner.ParseResourceVersion(resourceVersion)
		if err != nil {
			t.Fatal(err)
		}

		err = c.cacher.watchCache.waitUntilFreshAndBlock(context.TODO(), rv)
		if err != nil {
			t.Fatalf("WatchCache didn't caught up to RV: %v", rv)
		}
		c.cacher.watchCache.RUnlock()

		c.cacher.watchCache.Lock()
		defer c.cacher.watchCache.Unlock()
		c.cacher.Lock()
		defer c.cacher.Unlock()

		if c.cacher.watchCache.resourceVersion < rv {
			t.Fatalf("Can't compact into a future version: %v", resourceVersion)
		}

		if len(c.cacher.watchers.allWatchers) > 0 || len(c.cacher.watchers.valueWatchers) > 0 {
			// We could consider terminating those watchers, but given
			// watchcache doesn't really support compaction and we don't
			// exercise it in tests, we just throw an error here.
			t.Error("Open watchers are not supported during compaction")
		}

		for c.cacher.watchCache.startIndex < c.cacher.watchCache.endIndex {
			index := c.cacher.watchCache.startIndex % c.cacher.watchCache.capacity
			if c.cacher.watchCache.cache[index].ResourceVersion > rv {
				break
			}

			c.cacher.watchCache.startIndex++
		}
		c.cacher.watchCache.listResourceVersion = rv
		if _, err := client.Compact(ctx, int64(rv)); err != nil {
			t.Fatalf("Could not compact: %v", err)
		}
	}
}

func compactStore(c *CacheDelegator, client *clientv3.Client) storagetesting.Compaction {
	return func(ctx context.Context, t *testing.T, resourceVersion string) {
		versioner := storage.APIObjectVersioner{}
		rv, err := versioner.ParseResourceVersion(resourceVersion)
		if err != nil {
			t.Fatal(err)
		}
		var currentVersion int64
		currentVersion, _, _, err = etcd3.Compact(ctx, client, currentVersion, int64(rv))
		if err != nil {
			_, _, _, err = etcd3.Compact(ctx, client, currentVersion, int64(rv))
		}
		if err != nil {
			t.Fatal(err)
		}
		// Wait for compaction to be observed.
		if c.cacher.compactor != nil {
			for {
				select {
				case <-ctx.Done():
					t.Fatal(ctx.Err())
				case <-time.After(100 * time.Millisecond):
				}
				compactedRev := c.storage.CompactRevision()
				if compactedRev == int64(rv) {
					break
				}
			}
			c.cacher.compactor.compactIfNeeded()
		}
	}
}

func increaseRVFunc(client *clientv3.Client) storagetesting.IncreaseRVFunc {
	return func(ctx context.Context, t *testing.T) int64 {
		resp, err := client.KV.Put(ctx, "increaseRV", "ok")
		if err != nil {
			t.Fatalf("Could not update increaseRV: %v", err)
		}
		return resp.Header.Revision
	}
}
