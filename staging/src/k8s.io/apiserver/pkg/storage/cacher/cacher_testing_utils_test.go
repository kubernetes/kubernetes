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
	"go.etcd.io/etcd/server/v3/embed"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/protobuf"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	example2v1 "k8s.io/apiserver/pkg/apis/example2/v1"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storage/etcd3/testserver"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/utils/clock"
)

const DefaultBookmarkFrequency = defaultBookmarkFrequency

var (
	scheme   = runtime.NewScheme()
	Codecs   = serializer.NewCodecFactory(scheme)
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
	Corev1ProtoCodec = Codecs.CodecForVersions(pb, pb, schema.GroupVersions{corev1.SchemeGroupVersion}, schema.GroupVersions{corev1.SchemeGroupVersion})
	Examplev1ProtoCodec = Codecs.CodecForVersions(pb, pb, schema.GroupVersions{examplev1.SchemeGroupVersion}, nil)
}

var (
	Corev1ProtoCodec    runtime.Codec
	Examplev1ProtoCodec runtime.Codec
)

func NewPod() runtime.Object     { return &example.Pod{} }
func NewPodList() runtime.Object { return &example.PodList{} }

func NewEtcdTestStorage(t testing.TB, prefix string) (*etcd3testing.EtcdTestServer, storage.Interface) {
	return NewEtcdTestStorageWithCodec(t, prefix, Examplev1ProtoCodec)
}

func NewEtcdTestStorageWithCodec(t testing.TB, prefix string, codec runtime.Codec) (*etcd3testing.EtcdTestServer, storage.Interface) {
	return NewEtcdTestStorageWithOptions(t, prefix, codec, identity.NewEncryptCheckTransformer(), storagetesting.PodReverseKeyFunc("/pods/", true))
}

func NewEtcdTestStorageWithOptions(t testing.TB, prefix string, codec runtime.Codec, transformer value.Transformer, reverseKeyFunc storage.ReverseKeyFunc) (*etcd3testing.EtcdTestServer, storage.Interface) {
	server, _ := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)
	versioner := storage.APIObjectVersioner{}
	compactor := etcd3.NewCompactor(server.V3Client.Client, 0, clock.RealClock{}, nil)
	t.Cleanup(compactor.Stop)
	storage, err := etcd3.New(
		server.V3Client,
		compactor,
		codec,
		NewPod,
		NewPodList,
		reverseKeyFunc,
		prefix,
		"/pods/",
		schema.GroupResource{Resource: "pods"},
		transformer,
		etcd3.NewDefaultLeaseManagerConfig(),
		etcd3.NewDefaultDecoder(codec, versioner),
		versioner)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(storage.Close)
	return server, storage
}

func ComputePodKey(obj metav1.Object) string {
	return fmt.Sprintf("/pods/%s/%s", obj.GetNamespace(), obj.GetName())
}

func benchmarkEtcdTestStorage(t testing.TB) (*etcd3testing.EtcdTestServer, storage.Interface) {
	config := storagetesting.StoreConfigForBenchmarks()
	server := &etcd3testing.EtcdTestServer{V3Client: testserver.RunEtcd(t, func(cfg *embed.Config) {
		cfg.QuotaBackendBytes = 4 << 30 // 4 GiB (default 2 GiB is too small for 150k pods)
	})}
	compactor := etcd3.NewCompactor(server.V3Client.Client, 0, clock.RealClock{}, nil)
	t.Cleanup(compactor.Stop)
	s, err := etcd3.New(
		server.V3Client,
		compactor,
		config.Codec,
		config.NewFunc,
		config.NewListFunc,
		storagetesting.PodReverseKeyFunc(config.ResourcePrefix, true),
		etcd3testing.PathPrefix(),
		config.ResourcePrefix,
		config.GroupResource,
		identity.NewEncryptCheckTransformer(),
		etcd3.NewDefaultLeaseManagerConfig(),
		etcd3.NewDefaultDecoder(config.Codec, config.Versioner),
		config.Versioner)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	return server, s
}

func CompactWatch(c *CacheDelegator, client *clientv3.Client) storagetesting.Compaction {
	return func(ctx context.Context, t *testing.T, resourceVersion string) {
		versioner := storage.APIObjectVersioner{}
		rv, err := versioner.ParseResourceVersion(resourceVersion)
		if err != nil {
			t.Fatal(err)
		}

		c.cacher.watchCache.RLock()
		err = c.cacher.watchCache.waitUntilFreshLocked(context.TODO(), rv)
		c.cacher.watchCache.RUnlock()
		if err != nil {
			t.Fatalf("WatchCache didn't caught up to RV: %v", rv)
		}
		c.cacher.watchCache.Lock()
		defer c.cacher.watchCache.Unlock()
		c.cacher.Lock()
		defer c.cacher.Unlock()

		if c.cacher.watchCache.resourceVersion < rv {
			t.Fatalf("Can't compact into a future version: %v", resourceVersion)
		}

		for c.cacher.watchCache.history.startIndex < c.cacher.watchCache.history.endIndex {
			index := c.cacher.watchCache.history.startIndex % c.cacher.watchCache.history.capacity
			if c.cacher.watchCache.history.cache[index].ResourceVersion > rv {
				break
			}

			c.cacher.watchCache.history.startIndex++
		}
		c.cacher.watchCache.history.listResourceVersion = rv
		if _, err := client.Compact(ctx, int64(rv)); err != nil {
			t.Fatalf("Could not compact: %v", err)
		}
	}
}

func CompactStore(c storage.Interface, client *clientv3.Client) storagetesting.Compaction {
	var currentVersion int64
	return func(ctx context.Context, t *testing.T, resourceVersion string) {
		versioner := storage.APIObjectVersioner{}
		rv, err := versioner.ParseResourceVersion(resourceVersion)
		if err != nil {
			t.Fatal(err)
		}
		var compactRev int64
		currentVersion, _, compactRev, err = etcd3.Compact(ctx, client, currentVersion, int64(rv))
		if err == nil && compactRev != int64(rv) {
			currentVersion, _, compactRev, err = etcd3.Compact(ctx, client, currentVersion, int64(rv))
		}
		if err != nil {
			t.Fatal(err)
		}
		if compactRev != int64(rv) {
			t.Fatalf("expected compact revision %d, got %d", rv, compactRev)
		}
		// Wait for compaction to be observed when snapshot compaction is enabled.
		if utilfeature.DefaultFeatureGate.Enabled(features.ListFromCacheSnapshot) {
			for c.CompactRevision() < int64(rv) {
				select {
				case <-ctx.Done():
					t.Fatal(ctx.Err())
				case <-time.After(5 * time.Millisecond):
				}
			}
		}
	}
}

func IncreaseRVFunc(client *clientv3.Client) storagetesting.IncreaseRVFunc {
	return func(ctx context.Context, t *testing.T) int64 {
		resp, err := client.KV.Put(ctx, "increaseRV", "ok")
		if err != nil {
			t.Fatalf("Could not update increaseRV: %v", err)
		}
		return resp.Header.Revision
	}
}
