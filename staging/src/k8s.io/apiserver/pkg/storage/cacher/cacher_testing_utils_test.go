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

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/server/v3/embed"

	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storage/etcd3/testserver"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
	"k8s.io/utils/clock"
)

const DefaultBookmarkFrequency = defaultBookmarkFrequency

var errDummy = fmt.Errorf("dummy error")

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
