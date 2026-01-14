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

package consistency

import (
	"context"
	"fmt"
	"hash"
	"hash/fnv"
	"os"
	"strconv"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	"k8s.io/klog/v2"
)

var (
	// CheckPeriod is the period of checking consistency between etcd and cache.
	// 5 minutes were proposed to match the default compaction period. It's magnitute higher than
	// List latency SLO (30 seconds) and timeout (1 minute).
	CheckPeriod = 5 * time.Minute
	// PanicOnCacheInconsistency enables the consistency checking mechanism for cache.
	// Based on KUBE_WATCHCACHE_CONSISTENCY_CHECKER environment variable.
	PanicOnCacheInconsistency = false
	// checkerListPageSize is the checker's request chunk size of list operations.
	checkerListPageSize = int64(10000)
)

func init() {
	PanicOnCacheInconsistency, _ = strconv.ParseBool(os.Getenv("KUBE_WATCHCACHE_CONSISTENCY_CHECKER"))
}

func NewChecker(resourcePrefix string, groupResource schema.GroupResource, newListFunc func() runtime.Object, cacher cacher, etcd getLister) *Checker {
	return &Checker{
		groupResource:  groupResource,
		resourcePrefix: resourcePrefix,
		newListFunc:    newListFunc,
		cacher:         cacher,
		etcd:           etcd,
	}
}

type Checker struct {
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

func (c Checker) Run(stopCh <-chan struct{}) {
	klog.V(3).InfoS("Cache consistency check start", "group", c.groupResource.Group, "resource", c.groupResource.Resource)
	jitter := 0.5 // Period between [interval, interval * (1.0 + jitter)]
	sliding := true
	// wait.JitterUntilWithContext starts work immediately, so wait first.
	select {
	case <-time.After(wait.Jitter(CheckPeriod, jitter)):
	case <-stopCh:
	}
	wait.JitterUntilWithContext(wait.ContextForChannel(stopCh), c.check, CheckPeriod, jitter, sliding)
}

func (c *Checker) check(ctx context.Context) {
	digests, err := c.CalculateDigests(ctx)
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
	if PanicOnCacheInconsistency {
		panic(fmt.Sprintf("Cache consistency check failed, group: %q, resource: %q, resourceVersion: %q, etcdDigest: %q, cacheDigest: %q", c.groupResource.Group, c.groupResource.Resource, digests.ResourceVersion, digests.EtcdDigest, digests.CacheDigest))
	}
	c.cacher.MarkConsistent(false)
}

func (c *Checker) CalculateDigests(ctx context.Context) (*Digest, error) {
	if !c.cacher.Ready() {
		return nil, fmt.Errorf("cache is not ready")
	}
	cacheDigest, cacheResourceVersion, err := c.calculateStoreDigest(ctx, c.cacher, "0", 0)
	if err != nil {
		return nil, fmt.Errorf("failed calculating cache digest: %w", err)
	}
	etcdDigest, etcdResourceVersion, err := c.calculateStoreDigest(ctx, c.etcd, cacheResourceVersion, checkerListPageSize)
	if err != nil {
		return nil, fmt.Errorf("failed calculating etcd digest: %w", err)
	}
	if cacheResourceVersion != etcdResourceVersion {
		return nil, fmt.Errorf("etcd returned different resource version then expected, cache: %q, etcd: %q", cacheResourceVersion, etcdResourceVersion)
	}
	return &Digest{
		ResourceVersion: cacheResourceVersion,
		CacheDigest:     cacheDigest,
		EtcdDigest:      etcdDigest,
	}, nil
}

type Digest struct {
	ResourceVersion string
	CacheDigest     string
	EtcdDigest      string
}

func (c *Checker) calculateStoreDigest(ctx context.Context, store getLister, resourceVersion string, limit int64) (digest, rv string, err error) {
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
