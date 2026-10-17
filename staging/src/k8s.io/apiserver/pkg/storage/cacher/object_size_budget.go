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
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	"k8s.io/apiserver/pkg/storage/cacher/store"
	"k8s.io/klog/v2"
)

// The watch cache keeps every object of a resource in memory, decoded, which
// costs several times the object's serialized size. For resources whose
// objects are large on average -- reports, bundles, blobs written by a
// controller and rarely read -- that cost is out of all proportion to the
// benefit. When Config.MaxAverageObjectSizeBytes is set, such resources are
// served directly from storage instead, exactly as if their watch cache had
// been disabled.
//
// The decision is made twice:
//
//   - Before the watch cache is first populated, by probing storage (see
//     probeObjectSize). Populating the cache and then discarding it would cause
//     the very memory spike this is meant to prevent.
//   - Periodically while the cache is running, by measuring the cached objects
//     (see checkObjectSize). This catches resources that grow after the cache
//     was started, such as a CRD that is created empty and filled later.
//
// Both only take a resource out of the watch cache when its average object size
// is known to exceed the budget, never on the strength of a sample. Taking a
// resource out of the watch cache is permanent for the lifetime of the cacher.
//
// Object sizes are measured by encoding objects with the storage codec. That is
// the size they have in storage, before any encryption at rest.

const (
	// objectSizeProbePageSize is the number of objects fetched per request
	// while probing, which bounds the probe's transient memory use.
	objectSizeProbePageSize = 50
	// objectSizeProbeMaxObjects bounds how many objects the probe examines.
	// If the budget has not been proven exceeded by then, the resource is
	// cached and left to the periodic check.
	objectSizeProbeMaxObjects = 1000
	// objectSizeProbeTimeout bounds the duration of the probe.
	objectSizeProbeTimeout = time.Minute
)

// Variables for testing.
var (
	// objectSizeCheckPeriod is how often a running cacher measures its cached
	// objects against the budget.
	objectSizeCheckPeriod = time.Minute
	// objectSizeCheckMaxObjects and objectSizeCheckMaxBytes bound the work of a
	// single periodic check. The next check continues where it stopped.
	objectSizeCheckMaxObjects       = 1000
	objectSizeCheckMaxBytes   int64 = 64 << 20
)

// objectSizeProbeResult is the outcome of probeObjectSize.
type objectSizeProbeResult struct {
	exceeded bool
	// averageObjectSize is the average object size in bytes if every object
	// was examined, and a lower bound on it otherwise.
	averageObjectSize int64
	// objects is the number of objects in the resource, if known, and the
	// number examined otherwise.
	objects int64
	// examined is the number of objects whose size was measured.
	examined int64
}

// probeObjectSize determines whether the average object size of the resource
// exceeds the budget, without loading the resource into memory.
//
// Objects are fetched from storage a page at a time and measured, then
// discarded. The total object count comes with the first page, which allows
// stopping early: once the bytes seen so far, divided by the total count,
// exceed the budget, the true average must as well. Otherwise the budget is
// only reported exceeded once every object has been measured.
func (c *Cacher) probeObjectSize(ctx context.Context) (objectSizeProbeResult, error) {
	var (
		res        objectSizeProbeResult
		bytesSeen  int64
		totalCount int64 // -1 if unknown
		continueAt string
	)
	for page := 0; ; page++ {
		listObj := c.newListFunc()
		opts := storage.ListOptions{
			Recursive: true,
			Predicate: storage.SelectionPredicate{
				Label:    labels.Everything(),
				Field:    fields.Everything(),
				Limit:    objectSizeProbePageSize,
				Continue: continueAt,
			},
		}
		if err := c.storage.GetList(ctx, c.resourcePrefix, opts, listObj); err != nil {
			return res, err
		}
		var pageCount int64
		err := meta.EachListItem(listObj, func(obj runtime.Object) error {
			size, err := c.encodedSize(obj)
			if err != nil {
				return err
			}
			bytesSeen += size
			pageCount++
			return nil
		})
		if err != nil {
			return res, err
		}
		res.examined += pageCount

		listMeta, err := meta.ListAccessor(listObj)
		if err != nil {
			return res, err
		}
		continueAt = listMeta.GetContinue()
		if page == 0 {
			switch remaining := listMeta.GetRemainingItemCount(); {
			case continueAt == "":
				totalCount = pageCount
			case remaining != nil:
				totalCount = pageCount + *remaining
			default:
				totalCount = -1
			}
		}

		if res.examined == 0 {
			return res, nil
		}
		if continueAt == "" {
			// Every object has been measured.
			res.objects = res.examined
			res.averageObjectSize = bytesSeen / res.examined
			res.exceeded = res.averageObjectSize > c.maxAverageObjectSize
			return res, nil
		}
		if totalCount > 0 {
			res.objects = totalCount
			res.averageObjectSize = bytesSeen / totalCount
			if res.averageObjectSize > c.maxAverageObjectSize {
				res.exceeded = true
				return res, nil
			}
		} else {
			res.objects = res.examined
			res.averageObjectSize = bytesSeen / res.examined
		}
		if res.examined >= objectSizeProbeMaxObjects {
			// Not proven: leave the resource cached.
			return res, nil
		}
	}
}

// admitByObjectSize runs before the watch cache is first populated. It returns
// false if the resource was taken out of the watch cache, in which case the
// caller must not start caching.
//
// It is called from a goroutine tracked by stopWg.
func (c *Cacher) admitByObjectSize() bool {
	if c.maxAverageObjectSize <= 0 {
		return true
	}
	ctx, cancel := context.WithTimeout(wait.ContextForChannel(c.stopCh), objectSizeProbeTimeout)
	defer cancel()
	res, err := c.probeObjectSize(ctx)
	if err != nil {
		// Never let the budget make caching less available than it would be
		// without it. The periodic check still applies.
		klog.InfoS("Failed to probe object sizes, enabling watch cache", "group", c.groupResource.Group, "resource", c.groupResource.Resource, "err", err)
		return true
	}
	if !res.exceeded {
		klog.V(3).InfoS("Object size probe within budget, enabling watch cache", "group", c.groupResource.Group, "resource", c.groupResource.Resource,
			"averageObjectSizeBytes", res.averageObjectSize, "objects", res.objects, "examined", res.examined, "maxAverageObjectSizeBytes", c.maxAverageObjectSize)
		return true
	}
	c.bypassForObjectSize("probe", res.averageObjectSize, res.objects)
	return false
}

// monitorObjectSize periodically checks the cached objects against the budget
// once the cache is ready.
//
// It must not be tracked by stopWg, since checkObjectSize waits on it.
func (c *Cacher) monitorObjectSize() {
	ctx := wait.ContextForChannel(c.stopCh)
	if err := c.ready.wait(ctx); err != nil {
		return
	}
	wait.JitterUntilWithContext(ctx, func(context.Context) {
		c.checkObjectSize()
	}, objectSizeCheckPeriod, 0.1, true)
}

// objectSizeScan is the progress of periodic checks through the cached objects.
type objectSizeScan struct {
	// next is the index of the next cached object to measure.
	next int
	// bytes and examined accumulate over the current pass through the cache.
	bytes    int64
	examined int
}

// checkObjectSize measures cached objects, and takes the resource out of the
// watch cache if their average size exceeds the budget. It returns true if it
// did.
//
// A call measures a bounded number of objects, continuing where the previous
// call stopped, so that successive calls make a pass through the cache. The
// budget is exceeded if the bytes measured so far in the current pass, divided
// by the number of cached objects, exceed it -- which shows that the average
// does, without waiting for the pass to complete -- or if the average over a
// complete pass does.
//
// It must not be called concurrently with itself, nor from a goroutine tracked
// by stopWg, since it waits on it.
func (c *Cacher) checkObjectSize() bool {
	c.watchCache.RLock()
	items := c.watchCache.storage.List()
	c.watchCache.RUnlock()

	n := len(items)
	scan := &c.sizeScan
	if scan.next >= n || scan.examined >= n {
		// The cache shrank since the previous call; start a new pass.
		*scan = objectSizeScan{}
	}
	if n == 0 {
		return false
	}

	var bytes int64
	var examined int
	for examined < n-scan.examined && examined < objectSizeCheckMaxObjects && bytes < objectSizeCheckMaxBytes {
		item := items[(scan.next+examined)%n]
		elem, ok := item.(*store.Element)
		if !ok {
			klog.ErrorS(nil, "Unexpected object in watch cache", "group", c.groupResource.Group, "resource", c.groupResource.Resource, "type", fmt.Sprintf("%T", item))
			return false
		}
		size, err := c.encodedSize(elem.Object)
		if err != nil {
			klog.V(4).InfoS("Failed to measure cached object", "group", c.groupResource.Group, "resource", c.groupResource.Resource, "key", elem.Key, "err", err)
			return false
		}
		bytes += size
		examined++
	}
	scan.next = (scan.next + examined) % n
	scan.bytes += bytes
	scan.examined += examined

	var average int64
	switch {
	case scan.bytes/int64(n) > c.maxAverageObjectSize:
		average = scan.bytes / int64(n)
	case scan.examined >= n:
		average = scan.bytes / int64(scan.examined)
		*scan = objectSizeScan{}
		if average <= c.maxAverageObjectSize {
			return false
		}
	default:
		return false
	}

	if !c.bypassForObjectSize("monitor", average, int64(n)) {
		return false
	}
	// Wait for the reflector to stop, so that nothing repopulates the cache,
	// then drop what it holds. The cacher itself stays referenced by the
	// delegator, so without this its memory would not be freed.
	start := time.Now()
	c.stopWg.Wait()
	c.watchCache.Release()
	klog.InfoS("Released watch cache contents", "group", c.groupResource.Group, "resource", c.groupResource.Resource, "duration", time.Since(start))
	return true
}

// encodedSize returns the size of obj encoded with the storage codec.
func (c *Cacher) encodedSize(obj runtime.Object) (int64, error) {
	data, err := runtime.Encode(c.codec, obj)
	if err != nil {
		return 0, fmt.Errorf("encoding object: %w", err)
	}
	return int64(len(data)), nil
}

// bypassForObjectSize takes the resource out of the watch cache: requests are
// served from storage from now on, and the cacher is stopped. It returns false
// if the resource had already been taken out.
//
// It does not wait for the cacher's goroutines to finish, so that it can be
// called from one of them.
func (c *Cacher) bypassForObjectSize(detectedBy string, averageObjectSize, objects int64) bool {
	if !c.bypassed.CompareAndSwap(false, true) {
		return false
	}
	close(c.bypassedCh)
	klog.InfoS("Serving resource without watch cache: average object size exceeds budget",
		"group", c.groupResource.Group, "resource", c.groupResource.Resource,
		"averageObjectSizeBytes", averageObjectSize, "objects", objects,
		"maxAverageObjectSizeBytes", c.maxAverageObjectSize, "detectedBy", detectedBy)
	metrics.RecordObjectSizeBudgetExceeded(c.groupResource)
	// The storage's size estimator, if enabled, lists keys from this cacher.
	// Once the cacher is stopped that would fail, so fall back to plain object
	// counts, which is what storage without a watch cache reports.
	c.storage.DisableResourceSizeEstimation()
	c.signalStop()
	return true
}

// isBypassed reports whether the resource has been taken out of the watch cache
// and must be served from storage.
func (c *Cacher) isBypassed() bool {
	return c.bypassed.Load()
}
