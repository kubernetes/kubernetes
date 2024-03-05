/*
Copyright 2024 The Kubernetes Authors.

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
	"sync"

	clientv3 "go.etcd.io/etcd/client/v3"
	"google.golang.org/grpc/metadata"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

type TestCacher struct {
	*Cacher
}

func (c *TestCacher) NewFunc() runtime.Object {
	return c.newFunc()
}

func (c *TestCacher) WaitForEtcdBookmark(ctx context.Context) func() (uint64, error) {
	opts := storage.ListOptions{ResourceVersion: "", Predicate: storage.Everything, Recursive: true}
	opts.Predicate.AllowWatchBookmarks = true
	w, err := c.storage.Watch(ctx, "/pods/", opts)

	versioner := storage.APIObjectVersioner{}
	var rv uint64
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for event := range w.ResultChan() {
			if event.Type == watch.Bookmark {
				rv, err = versioner.ObjectResourceVersion(event.Object)
				break
			}
		}
	}()
	return func() (uint64, error) {
		defer w.Stop()
		wg.Wait()
		return rv, err
	}
}

func (c *TestCacher) WatchContextMetadata() metadata.MD {
	return c.watchCache.waitingUntilFresh.contextMetadata
}

func (c *TestCacher) WaitReady(ctx context.Context) error {
	return c.ready.wait(ctx)
}

func (c *TestCacher) ResourceVersion() uint64 {
	c.watchCache.RLock()
	defer c.watchCache.RUnlock()
	return c.watchCache.resourceVersion
}

func (c *TestCacher) Compact(ctx context.Context, client *clientv3.Client, resourceVersion string) error {
	versioner := storage.APIObjectVersioner{}
	rv, err := versioner.ParseResourceVersion(resourceVersion)
	if err != nil {
		return err
	}

	err = c.watchCache.waitUntilFreshAndBlock(context.TODO(), rv)
	if err != nil {
		return fmt.Errorf("WatchCache didn't caught up to RV: %v", rv)
	}
	c.watchCache.RUnlock()

	c.watchCache.Lock()
	defer c.watchCache.Unlock()
	c.Lock()
	defer c.Unlock()

	if c.watchCache.resourceVersion < rv {
		return fmt.Errorf("can't compact into a future version: %v", resourceVersion)
	}

	if len(c.watchers.allWatchers) > 0 || len(c.watchers.valueWatchers) > 0 {
		// We could consider terminating those watchers, but given
		// watchcache doesn't really support compaction and we don't
		// exercise it in tests, we just throw an error here.
		return fmt.Errorf("open watchers are not supported during compaction")
	}

	for c.watchCache.startIndex < c.watchCache.endIndex {
		index := c.watchCache.startIndex % c.watchCache.capacity
		if c.watchCache.cache[index].ResourceVersion > rv {
			break
		}

		c.watchCache.startIndex++
	}
	c.watchCache.listResourceVersion = rv

	if _, err = client.KV.Put(ctx, "compact_rev_key", resourceVersion); err != nil {
		return fmt.Errorf("could not update compact_rev_key: %w", err)
	}
	if _, err = client.Compact(ctx, int64(rv)); err != nil {
		return fmt.Errorf("could not compact: %w", err)
	}
	return nil
}
