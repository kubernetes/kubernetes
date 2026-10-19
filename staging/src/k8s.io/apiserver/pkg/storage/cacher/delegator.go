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
	"sync"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/consistency"
	"k8s.io/apiserver/pkg/storage/cacher/delegator"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
)

func NewCacheDelegator(cacher *Cacher, storage storage.Interface) *CacheDelegator {
	d := &CacheDelegator{
		cacher:  cacher,
		storage: storage,
		stopCh:  make(chan struct{}),
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.DetectCacheInconsistency) || consistency.PanicOnCacheInconsistency {
		d.checker = consistency.NewChecker(cacher.resourcePrefix, cacher.groupResource, cacher.newListFunc, cacher, storage)
		d.wg.Add(1)
		go func() {
			defer d.wg.Done()
			d.checker.Run(d.stopCh)
		}()
	}
	return d
}

type CacheDelegator struct {
	cacher  *Cacher
	storage storage.Interface
	checker *consistency.Checker

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
	if !c.cacher.Ready() || opts.ExpectTransformOrDecodeError {
		return c.storage.Delete(ctx, key, out, preconditions, validateDeletion, cachedExistingObject, opts)
	}
	if delegator.ShouldDelegateGet(storage.GetOptions{}).ShouldDelegate {
		// Ignore the suggestion and try to pass down the current version of the object
		// read from cache.
		if obj, exists, _, err := c.cacher.get(ctx, key, "0"); err != nil {
			klog.Errorf("GetByKey returned error: %v", err)
		} else if exists && preconditions.Check(key, obj) == nil {
			// DeepCopy the object since we modify resource version when serializing the
			// current object.
			return c.storage.Delete(ctx, key, out, preconditions, validateDeletion, obj.DeepCopyObject(), opts)
		}
		// If we couldn't get the object, fallback to no-suggestion.
		return c.storage.Delete(ctx, key, out, preconditions, validateDeletion, nil, opts)
	}

	key, err := c.cacher.prepareKey(key, false)
	if err != nil {
		return err
	}
	deleteOpts := opts
	deleteOpts.FailOnConflict = true
	fresh := false
	currObj, exists, readRV, err := c.cacher.get(ctx, key, "0")
	for {
		var cachedRV uint64
		var hasCachedRV bool
		if err == nil {
			if !exists {
				err = storage.NewKeyNotFoundError(key, int64(readRV))
			} else {
				if rv, rvErr := c.cacher.versioner.ObjectResourceVersion(currObj); rvErr == nil {
					cachedRV = rv
					hasCachedRV = true
				}
				err = c.storage.Delete(ctx, key, out, preconditions, validateDeletion, currObj.DeepCopyObject(), deleteOpts)
				if err == nil {
					return nil
				}
			}
		}
		if rv, ok := storage.ResourceVersion(err); ok && storage.IsConflict(err) {
			klog.V(4).Infof("Delete of %s conflicted, waiting for watch cache to reach revision %s", key, rv)
			if currObj, exists, readRV, err = c.cacher.get(ctx, key, rv); err != nil {
				if storage.IsTooLargeResourceVersion(err) {
					return c.storage.Delete(ctx, key, out, preconditions, validateDeletion, nil, opts)
				}
				return err
			}
			fresh = true
			continue
		}
		if !fresh {
			fresh = true
			var getErr error
			currObj, exists, readRV, getErr = c.cacher.get(ctx, key, "")
			if storage.IsTooLargeResourceVersion(getErr) {
				return c.storage.Delete(ctx, key, out, preconditions, validateDeletion, nil, opts)
			}
			if getErr == nil && exists && hasCachedRV {
				if freshRV, rvErr := c.cacher.versioner.ObjectResourceVersion(currObj); rvErr == nil && freshRV == cachedRV {
					return err
				}
			}
			err = getErr
			continue
		}
		return err
	}
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
	return c.cacher.Watch(ctx, key, opts)
}

func (c *CacheDelegator) Get(ctx context.Context, key string, opts storage.GetOptions, objPtr runtime.Object) error {
	result := delegator.ShouldDelegateGet(opts)
	if result.ShouldDelegate {
		return c.storage.Get(ctx, key, opts, objPtr)
	}

	if !c.cacher.Ready() {
		// If Cache is not initialized, delegator Get requests to storage
		// as described in https://kep.k8s.io/4568
		return c.storage.Get(ctx, key, opts, objPtr)
	}
	if _, err := c.cacher.versioner.ParseResourceVersion(opts.ResourceVersion); err != nil {
		return err
	}
	err := c.cacher.Get(ctx, key, opts, objPtr)
	success := "true"
	fallback := "false"
	if err != nil {
		if result.ConsistentRead {
			// IsTooLargeResourceVersion occurs when the requested RV is higher than cache's current RV
			// and cache hasn't caught up within the timeout period.
			if storage.IsTooLargeResourceVersion(err) {
				if utilfeature.DefaultFeatureGate.Enabled(features.ConsistentListFromCacheSkipTimeoutFallback) {
					fallback = "skipped"
					err = errors.NewTooManyRequests(err.Error(), resourceVersionTooHighRetrySeconds)
				} else {
					fallback = "true"
					err = c.storage.Get(ctx, key, opts, objPtr)
				}
			}
			if err != nil && !storage.IsNotFound(err) {
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

	if _, err := c.cacher.versioner.ParseResourceVersion(opts.ResourceVersion); err != nil {
		return err
	}

	if !c.cacher.Ready() && shouldDelegateListOnNotReadyCache(opts) {
		// If Cacher is not initialized, delegator List requests to storage
		// as described in https://kep.k8s.io/4568
		return c.storage.GetList(ctx, key, opts, listObj)
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
			// and cache hasn't caught up within the timeout period.
			if storage.IsTooLargeResourceVersion(err) {
				if utilfeature.DefaultFeatureGate.Enabled(features.ConsistentListFromCacheSkipTimeoutFallback) {
					fallback = "skipped"
					err = errors.NewTooManyRequests(err.Error(), resourceVersionTooHighRetrySeconds)
				} else {
					fallback = "true"
					err = c.storage.GetList(ctx, key, opts, listObj)
				}
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

func (c *CacheDelegator) Update(ctx context.Context, key string, destination runtime.Object, ignoreNotFound bool, preconditions *storage.Preconditions, tryUpdate storage.UpdateFunc, cachedExistingObject runtime.Object) error {
	return fmt.Errorf("cacher doesn't support Update")
}

func (c *CacheDelegator) GuaranteedUpdate(ctx context.Context, key string, destination runtime.Object, ignoreNotFound bool, preconditions *storage.Preconditions, tryUpdate storage.UpdateFunc, cachedExistingObject runtime.Object) error {
	if !c.cacher.Ready() {
		return c.storage.GuaranteedUpdate(ctx, key, destination, ignoreNotFound, preconditions, tryUpdate, cachedExistingObject)
	}
	if delegator.ShouldDelegateGet(storage.GetOptions{}).ShouldDelegate {
		// Ignore the suggestion and try to pass down the current version of the object
		// read from cache.
		if obj, exists, _, err := c.cacher.get(ctx, key, "0"); err != nil {
			klog.Errorf("GetByKey returned error: %v", err)
		} else if exists && preconditions.Check(key, obj) == nil {
			// DeepCopy the object since we modify resource version when serializing the
			// current object.
			return c.storage.GuaranteedUpdate(ctx, key, destination, ignoreNotFound, preconditions, tryUpdate, obj.DeepCopyObject())
		}
		// If we couldn't get the object, fallback to no-suggestion.
		return c.storage.GuaranteedUpdate(ctx, key, destination, ignoreNotFound, preconditions, tryUpdate, nil)
	}

	key, err := c.cacher.prepareKey(key, false)
	if err != nil {
		return err
	}
	fresh := false
	currObj, exists, readRV, err := c.cacher.get(ctx, key, "0")
	for {
		var cachedRV uint64
		var hasCachedRV bool
		if err == nil {
			if !exists {
				if !ignoreNotFound {
					err = storage.NewKeyNotFoundError(key, int64(readRV))
				} else {
					currObj = c.cacher.newFunc()
				}
			}
			if err == nil {
				if rv, rvErr := c.cacher.versioner.ObjectResourceVersion(currObj); rvErr == nil {
					cachedRV = rv
					hasCachedRV = true
				}
				var objToPass runtime.Object
				if exists {
					objToPass = currObj.DeepCopyObject()
				} else {
					objToPass = currObj
				}
				err = c.storage.Update(ctx, key, destination, ignoreNotFound, preconditions, tryUpdate, objToPass)
				if err == nil {
					if fresh {
						return nil
					}
					destRV, destErr := c.cacher.versioner.ObjectResourceVersion(destination)
					if hasCachedRV && destErr == nil && cachedRV != destRV {
						return nil
					}
				}
			}
		}
		if rv, ok := storage.ResourceVersion(err); ok && storage.IsConflict(err) {
			klog.V(4).Infof("GuaranteedUpdate of %s conflicted, waiting for watch cache to reach revision %s", key, rv)
			if currObj, exists, readRV, err = c.cacher.get(ctx, key, rv); err != nil {
				if storage.IsTooLargeResourceVersion(err) {
					return c.storage.GuaranteedUpdate(ctx, key, destination, ignoreNotFound, preconditions, tryUpdate, nil)
				}
				return err
			}
			fresh = true
			continue
		}
		if !fresh {
			fresh = true
			var getErr error
			currObj, exists, readRV, getErr = c.cacher.get(ctx, key, "")
			if storage.IsTooLargeResourceVersion(getErr) {
				return c.storage.GuaranteedUpdate(ctx, key, destination, ignoreNotFound, preconditions, tryUpdate, nil)
			}
			if getErr == nil && hasCachedRV {
				if !exists && ignoreNotFound && cachedRV == 0 {
					return err
				}
				if exists {
					if freshRV, rvErr := c.cacher.versioner.ObjectResourceVersion(currObj); rvErr == nil && freshRV == cachedRV {
						return err
					}
				}
			}
			err = getErr
			continue
		}
		return err
	}
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
