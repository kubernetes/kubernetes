/*
Copyright 2015 The Kubernetes Authors.

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
	"hash"
	"hash/fnv"
	"os"
	"strconv"
	"sync"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/delegator"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/tracing"
	"k8s.io/klog/v2"
)

var (
	// ConsistencyCheckPeriod is the period of checking consistency between etcd and cache.
	// 5 minutes were proposed to match the default compaction period. It's magnitute higher than
	// List latency SLO (30 seconds) and timeout (1 minute).
	ConsistencyCheckPeriod = 5 * time.Minute
	// panicOnCacheInconsistency enables the consistency checking mechanism for cache.
	// Based on KUBE_WATCHCACHE_CONSISTENCY_CHECKER environment variable.
	panicOnCacheInconsistency = false
)

func init() {
	panicOnCacheInconsistency, _ = strconv.ParseBool(os.Getenv("KUBE_WATCHCACHE_CONSISTENCY_CHECKER"))
}

func NewCacheDelegator(cacher *Cacher, storage storage.Interface) *CacheDelegator {
	d := &CacheDelegator{
		cacher:  cacher,
		storage: storage,
		stopCh:  make(chan struct{}),
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.DetectCacheInconsistency) || panicOnCacheInconsistency {
		d.checker = newConsistencyChecker(cacher.resourcePrefix, cacher.groupResource, cacher.newListFunc, cacher, storage)
		d.wg.Add(1)
		go func() {
			defer d.wg.Done()
			d.checker.startChecking(d.stopCh)
		}()
	}
	return d
}

type CacheDelegator struct {
	cacher  *Cacher
	storage storage.Interface
	checker *consistencyChecker

	wg       sync.WaitGroup
	stopOnce sync.Once
	stopCh   chan struct{}
}

var _ storage.Interface = (*CacheDelegator)(nil)

func (c *CacheDelegator) Versioner() storage.Versioner {
	return c.storage.Versioner()
}

func (c *CacheDelegator) Create(ctx context.Context, key string, obj, out runtime.Object, ttl uint64) error {
	return c.storage.Create(ctx, key, obj, out, ttl)
}

func (c *CacheDelegator) GetCurrentResourceVersion(ctx context.Context) (uint64, error) {
	return c.storage.GetCurrentResourceVersion(ctx)
}

func (c *CacheDelegator) EnableResourceSizeEstimation(keys storage.KeysFunc) error {
	return c.storage.EnableResourceSizeEstimation(keys)
}

func (c *CacheDelegator) Delete(ctx context.Context, key string, out runtime.Object, preconditions *storage.Preconditions, validateDeletion storage.ValidateObjectFunc, cachedExistingObject runtime.Object, opts storage.DeleteOptions) error {
	// Ignore the suggestion and try to pass down the current version of the object
	// read from cache.
	if elem, exists, err := c.cacher.watchCache.GetByKey(key); err != nil {
		klog.Errorf("GetByKey returned error: %v", err)
	} else if exists {
		// DeepCopy the object since we modify resource version when serializing the
		// current object.
		currObj := elem.(*storeElement).Object.DeepCopyObject()
		return c.storage.Delete(ctx, key, out, preconditions, validateDeletion, currObj, opts)
	}
	// If we couldn't get the object, fallback to no-suggestion.
	return c.storage.Delete(ctx, key, out, preconditions, validateDeletion, nil, opts)
}

func (c *CacheDelegator) Watch(ctx context.Context, key string, opts storage.ListOptions) (watch.Interface, error) {
	// if the watch-list feature wasn't set and the resourceVersion is unset
	// ensure that the rv from which the watch is being served, is the latest
	// one. "latest" is ensured by serving the watch from
	// the underlying storage.
	//
	// it should never happen due to our validation but let's just be super-safe here
	// and disable sendingInitialEvents when the feature wasn't enabled
	if !utilfeature.DefaultFeatureGate.Enabled(features.WatchList) && opts.SendInitialEvents != nil {
		opts.SendInitialEvents = nil
	}
	// TODO: we should eventually get rid of this legacy case
	if utilfeature.DefaultFeatureGate.Enabled(features.WatchFromStorageWithoutResourceVersion) && opts.SendInitialEvents == nil && opts.ResourceVersion == "" {
		return c.storage.Watch(ctx, key, opts)
	}
	return c.cacher.Watch(ctx, key, opts)
}

func (c *CacheDelegator) Get(ctx context.Context, key string, opts storage.GetOptions, objPtr runtime.Object) error {
	ctx, span := tracing.Start(ctx, "cacher.Get",
		attribute.String("audit-id", audit.GetAuditIDTruncated(ctx)),
		attribute.String("key", key),
		attribute.String("resource-version", opts.ResourceVersion))
	defer span.End(500 * time.Millisecond)
	if opts.ResourceVersion == "" {
		// If resourceVersion is not specified, serve it from underlying
		// storage (for backward compatibility).
		span.AddEvent("About to Get from underlying storage")
		return c.storage.Get(ctx, key, opts, objPtr)
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.ResilientWatchCacheInitialization) {
		if !c.cacher.Ready() {
			// If Cache is not initialized, delegator Get requests to storage
			// as described in https://kep.k8s.io/4568
			span.AddEvent("About to Get from underlying storage - cache not initialized")
			return c.storage.Get(ctx, key, opts, objPtr)
		}
	}
	// If resourceVersion is specified, serve it from cache.
	// It's guaranteed that the returned value is at least that
	// fresh as the given resourceVersion.
	getRV, err := c.cacher.versioner.ParseResourceVersion(opts.ResourceVersion)
	if err != nil {
		return err
	}
	// Do not create a trace - it's not for free and there are tons
	// of Get requests. We can add it if it will be really needed.
	if !utilfeature.DefaultFeatureGate.Enabled(features.ResilientWatchCacheInitialization) {
		if getRV == 0 && !c.cacher.Ready() {
			// If Cacher is not yet initialized and we don't require any specific
			// minimal resource version, simply forward the request to storage.
			return c.storage.Get(ctx, key, opts, objPtr)
		}
		if err := c.cacher.ready.wait(ctx); err != nil {
			return errors.NewServiceUnavailable(err.Error())
		}
	}
	span.AddEvent("About to fetch object from cache")
	return c.cacher.Get(ctx, key, opts, objPtr)
}

func (c *CacheDelegator) GetList(ctx context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
	_, _, err := storage.ValidateListOptions(c.cacher.resourcePrefix, c.cacher.versioner, opts)
	if err != nil {
		return err
	}
	result, err := delegator.ShouldDelegateList(opts, c.cacher)
	if err != nil {
		return err
	}
	if result.ShouldDelegate {
		return c.storage.GetList(ctx, key, opts, listObj)
	}

	listRV, err := c.cacher.versioner.ParseResourceVersion(opts.ResourceVersion)
	if err != nil {
		return err
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.ResilientWatchCacheInitialization) {
		if !c.cacher.Ready() && shouldDelegateListOnNotReadyCache(opts) {
			// If Cacher is not initialized, delegator List requests to storage
			// as described in https://kep.k8s.io/4568
			return c.storage.GetList(ctx, key, opts, listObj)
		}
	} else {
		if listRV == 0 && !c.cacher.Ready() {
			// If Cacher is not yet initialized and we don't require any specific
			// minimal resource version, simply forward the request to storage.
			return c.storage.GetList(ctx, key, opts, listObj)
		}
	}
	err = c.cacher.GetList(ctx, key, opts, listObj)
	success := "true"
	fallback := "false"
	if err != nil {
		if errors.IsResourceExpired(err) && utilfeature.DefaultFeatureGate.Enabled(features.ListFromCacheSnapshot) {
			return c.storage.GetList(ctx, key, opts, listObj)
		}
		if result.ConsistentRead {
			// IsTooLargeResourceVersion occurs when the requested RV is higher than cache's current RV
			// and cache hasn't caught up within the timeout period. Fall back to etcd.
			if storage.IsTooLargeResourceVersion(err) {
				fallback = "true"
				err = c.storage.GetList(ctx, key, opts, listObj)
			}
			if err != nil {
				success = "false"
			}
			metrics.ConsistentReadTotal.WithLabelValues(c.cacher.groupResource.Group, c.cacher.groupResource.Resource, success, fallback).Add(1)
		}
		return err
	}
	if result.ConsistentRead {
		metrics.ConsistentReadTotal.WithLabelValues(c.cacher.groupResource.Group, c.cacher.groupResource.Resource, success, fallback).Add(1)
	}
	return nil
}

func shouldDelegateListOnNotReadyCache(opts storage.ListOptions) bool {
	pred := opts.Predicate
	noLabelSelector := pred.Label == nil || pred.Label.Empty()
	noFieldSelector := pred.Field == nil || pred.Field.Empty()
	hasLimit := pred.Limit > 0
	return noLabelSelector && noFieldSelector && hasLimit
}

func (c *CacheDelegator) GuaranteedUpdate(ctx context.Context, key string, destination runtime.Object, ignoreNotFound bool, preconditions *storage.Preconditions, tryUpdate storage.UpdateFunc, cachedExistingObject runtime.Object) error {
	// Ignore the suggestion and try to pass down the current version of the object
	// read from cache.
	if elem, exists, err := c.cacher.watchCache.GetByKey(key); err != nil {
		klog.Errorf("GetByKey returned error: %v", err)
	} else if exists {
		// DeepCopy the object since we modify resource version when serializing the
		// current object.
		currObj := elem.(*storeElement).Object.DeepCopyObject()
		return c.storage.GuaranteedUpdate(ctx, key, destination, ignoreNotFound, preconditions, tryUpdate, currObj)
	}
	// If we couldn't get the object, fallback to no-suggestion.
	return c.storage.GuaranteedUpdate(ctx, key, destination, ignoreNotFound, preconditions, tryUpdate, nil)
}

func (c *CacheDelegator) Stats(ctx context.Context) (storage.Stats, error) {
	return c.storage.Stats(ctx)
}

func (c *CacheDelegator) ReadinessCheck() error {
	if !c.cacher.Ready() {
		return storage.ErrStorageNotReady
	}
	return nil
}

func (c *CacheDelegator) RequestWatchProgress(ctx context.Context) error {
	return c.storage.RequestWatchProgress(ctx)
}

func (c *CacheDelegator) CompactRevision() int64 {
	if c.cacher.compactor == nil {
		return c.storage.CompactRevision()
	}
	return c.cacher.compactor.Revision()
}

func (c *CacheDelegator) Stop() {
	c.stopOnce.Do(func() {
		close(c.stopCh)
	})
	c.wg.Wait()
}

func newConsistencyChecker(resourcePrefix string, groupResource schema.GroupResource, newListFunc func() runtime.Object, cacher cacher, etcd getLister) *consistencyChecker {
	return &consistencyChecker{
		groupResource:  groupResource,
		resourcePrefix: resourcePrefix,
		newListFunc:    newListFunc,
		cacher:         cacher,
		etcd:           etcd,
	}
}

type consistencyChecker struct {
	resourcePrefix string
	groupResource  schema.GroupResource
	newListFunc    func() runtime.Object

	cacher cacher
	etcd   getLister
}

type cacher interface {
	getLister
	Ready() bool
	MarkConsistent(bool)
}

type getLister interface {
	GetList(ctx context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error
}

func (c consistencyChecker) startChecking(stopCh <-chan struct{}) {
	klog.V(3).InfoS("Cache consistency check start", "group", c.groupResource.Group, "resource", c.groupResource.Resource)
	jitter := 0.5 // Period between [interval, interval * (1.0 + jitter)]
	sliding := true
	// wait.JitterUntilWithContext starts work immediately, so wait first.
	select {
	case <-time.After(wait.Jitter(ConsistencyCheckPeriod, jitter)):
	case <-stopCh:
	}
	wait.JitterUntilWithContext(wait.ContextForChannel(stopCh), c.check, ConsistencyCheckPeriod, jitter, sliding)
}

func (c *consistencyChecker) check(ctx context.Context) {
	digests, err := c.calculateDigests(ctx)
	if err != nil {
		klog.ErrorS(err, "Cache consistency check error", "group", c.groupResource.Group, "resource", c.groupResource.Resource)
		metrics.StorageConsistencyCheckTotal.WithLabelValues(c.groupResource.Group, c.groupResource.Resource, "error").Inc()
		return
	}
	if digests.CacheDigest == digests.EtcdDigest {
		klog.V(3).InfoS("Cache consistency check passed", "group", c.groupResource.Group, "resource", c.groupResource.Resource, "resourceVersion", digests.ResourceVersion, "digest", digests.CacheDigest)
		metrics.StorageConsistencyCheckTotal.WithLabelValues(c.groupResource.Group, c.groupResource.Resource, "success").Inc()
		c.cacher.MarkConsistent(true)
		return
	}
	klog.ErrorS(nil, "Cache consistency check failed", "group", c.groupResource.Group, "resource", c.groupResource.Resource, "resourceVersion", digests.ResourceVersion, "etcdDigest", digests.EtcdDigest, "cacheDigest", digests.CacheDigest)
	metrics.StorageConsistencyCheckTotal.WithLabelValues(c.groupResource.Group, c.groupResource.Resource, "failure").Inc()
	if panicOnCacheInconsistency {
		panic(fmt.Sprintf("Cache consistency check failed, group: %q, resource: %q, resourceVersion: %q, etcdDigest: %q, cacheDigest: %q", c.groupResource.Group, c.groupResource.Resource, digests.ResourceVersion, digests.EtcdDigest, digests.CacheDigest))
	}
	c.cacher.MarkConsistent(false)
}

func (c *consistencyChecker) calculateDigests(ctx context.Context) (*storageDigest, error) {
	if !c.cacher.Ready() {
		return nil, fmt.Errorf("cache is not ready")
	}
	cacheDigest, cacheResourceVersion, err := c.calculateStoreDigest(ctx, c.cacher, "0", 0)
	if err != nil {
		return nil, fmt.Errorf("failed calculating cache digest: %w", err)
	}
	etcdDigest, etcdResourceVersion, err := c.calculateStoreDigest(ctx, c.etcd, cacheResourceVersion, storageWatchListPageSize)
	if err != nil {
		return nil, fmt.Errorf("failed calculating etcd digest: %w", err)
	}
	if cacheResourceVersion != etcdResourceVersion {
		return nil, fmt.Errorf("etcd returned different resource version then expected, cache: %q, etcd: %q", cacheResourceVersion, etcdResourceVersion)
	}
	return &storageDigest{
		ResourceVersion: cacheResourceVersion,
		CacheDigest:     cacheDigest,
		EtcdDigest:      etcdDigest,
	}, nil
}

type storageDigest struct {
	ResourceVersion string
	CacheDigest     string
	EtcdDigest      string
}

func (c *consistencyChecker) calculateStoreDigest(ctx context.Context, store getLister, resourceVersion string, limit int64) (digest, rv string, err error) {
	opts := storage.ListOptions{
		Recursive:       true,
		Predicate:       storage.Everything,
		ResourceVersion: resourceVersion,
	}
	opts.Predicate.Limit = limit
	if resourceVersion == "0" {
		opts.ResourceVersionMatch = metav1.ResourceVersionMatchNotOlderThan
	} else {
		opts.ResourceVersionMatch = metav1.ResourceVersionMatchExact
	}
	h := fnv.New64()
	for {
		resp := c.newListFunc()
		err = store.GetList(ctx, c.resourcePrefix, opts, resp)
		if err != nil {
			return "", "", err
		}
		err = addListToDigest(h, resp)
		if err != nil {
			return "", "", err
		}
		list, err := meta.ListAccessor(resp)
		if err != nil {
			return "", "", err
		}
		if resourceVersion == "0" {
			resourceVersion = list.GetResourceVersion()
		}
		continueToken := list.GetContinue()
		if continueToken == "" {
			return fmt.Sprintf("%x", h.Sum64()), resourceVersion, nil
		}
		opts.Predicate.Continue = continueToken
		opts.ResourceVersion = ""
		opts.ResourceVersionMatch = ""
	}
}

func addListToDigest(h hash.Hash64, list runtime.Object) error {
	return meta.EachListItem(list, func(obj runtime.Object) error {
		objectMeta, err := meta.Accessor(obj)
		if err != nil {
			return err
		}
		err = addObjectToDigest(h, objectMeta)
		if err != nil {
			return err
		}
		return nil
	})
}

func addObjectToDigest(h hash.Hash64, objectMeta metav1.Object) error {
	_, err := h.Write([]byte(objectMeta.GetNamespace()))
	if err != nil {
		return err
	}
	_, err = h.Write([]byte("/"))
	if err != nil {
		return err
	}
	_, err = h.Write([]byte(objectMeta.GetName()))
	if err != nil {
		return err
	}
	_, err = h.Write([]byte("/"))
	if err != nil {
		return err
	}
	_, err = h.Write([]byte(objectMeta.GetResourceVersion()))
	if err != nil {
		return err
	}
	return nil
}
