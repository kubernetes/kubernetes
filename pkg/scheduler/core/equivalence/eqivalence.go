/*
Copyright 2016 The Kubernetes Authors.

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

// Package equivalence defines Pod equivalence classes and the equivalence class
// cache.
package equivalence

import (
	"fmt"
	"hash/fnv"
	"sync"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
)

// Cache is a thread safe map saves and reuses the output of predicate functions,
// it uses node name as key to access those cached results.
//
// Internally, results are keyed by predicate name, and "equivalence
// class". (Equivalence class is defined in the `Class` type.) Saved results
// will be reused until an appropriate invalidation function is called.
type Cache struct {
	// i.e. map[string]*NodeCache
	sync.Map
}

// NewCache create an empty equiv class cache.
func NewCache() *Cache {
	return new(Cache)
}

// NodeCache saves and reuses the output of predicate functions. Use RunPredicate to
// get or update the cached results. An appropriate Invalidate* function should
// be called when some predicate results are no longer valid.
//
// Internally, results are keyed by predicate name, and "equivalence
// class". (Equivalence class is defined in the `Class` type.) Saved results
// will be reused until an appropriate invalidation function is called.
//
// NodeCache objects are thread safe within the context of NodeCache,
type NodeCache struct {
	mu    sync.RWMutex
	cache predicateMap
}

// newNodeCache returns an empty NodeCache.
func newNodeCache() *NodeCache {
	return &NodeCache{
		cache: make(predicateMap),
	}
}

// GetNodeCache returns the existing NodeCache for given node if present. Otherwise,
// it creates the NodeCache and returns it.
// The boolean flag is true if the value was loaded, false if created.
func (c *Cache) GetNodeCache(name string) (nodeCache *NodeCache, exists bool) {
	v, exists := c.LoadOrStore(name, newNodeCache())
	nodeCache = v.(*NodeCache)
	return
}

// InvalidatePredicates clears all cached results for the given predicates.
func (c *Cache) InvalidatePredicates(predicateKeys sets.String) {
	if len(predicateKeys) == 0 {
		return
	}
	c.Range(func(k, v interface{}) bool {
		n := v.(*NodeCache)
		n.invalidatePreds(predicateKeys)
		return true
	})
	glog.V(5).Infof("Cache invalidation: node=*,predicates=%v", predicateKeys)

}

// InvalidatePredicatesOnNode clears cached results for the given predicates on one node.
func (c *Cache) InvalidatePredicatesOnNode(nodeName string, predicateKeys sets.String) {
	if len(predicateKeys) == 0 {
		return
	}
	if v, ok := c.Load(nodeName); ok {
		n := v.(*NodeCache)
		n.invalidatePreds(predicateKeys)
	}
	glog.V(5).Infof("Cache invalidation: node=%s,predicates=%v", nodeName, predicateKeys)
}

// InvalidateAllPredicatesOnNode clears all cached results for one node.
func (c *Cache) InvalidateAllPredicatesOnNode(nodeName string) {
	c.Delete(nodeName)
	glog.V(5).Infof("Cache invalidation: node=%s,predicates=*", nodeName)
}

// InvalidateCachedPredicateItemForPodAdd is a wrapper of
// InvalidateCachedPredicateItem for pod add case
// TODO: This does not belong with the equivalence cache implementation.
func (c *Cache) InvalidateCachedPredicateItemForPodAdd(pod *v1.Pod, nodeName string) {
	// MatchInterPodAffinity: we assume scheduler can make sure newly bound pod
	// will not break the existing inter pod affinity. So we does not need to
	// invalidate MatchInterPodAffinity when pod added.
	//
	// But when a pod is deleted, existing inter pod affinity may become invalid.
	// (e.g. this pod was preferred by some else, or vice versa)
	//
	// NOTE: assumptions above will not stand when we implemented features like
	// RequiredDuringSchedulingRequiredDuringExecutioc.

	// NoDiskConflict: the newly scheduled pod fits to existing pods on this node,
	// it will also fits to equivalence class of existing pods

	// GeneralPredicates: will always be affected by adding a new pod
	invalidPredicates := sets.NewString(predicates.GeneralPred)

	// MaxPDVolumeCountPredicate: we check the volumes of pod to make decisioc.
	for _, vol := range pod.Spec.Volumes {
		if vol.PersistentVolumeClaim != nil {
			invalidPredicates.Insert(
				predicates.MaxEBSVolumeCountPred,
				predicates.MaxGCEPDVolumeCountPred,
				predicates.MaxAzureDiskVolumeCountPred)
			if utilfeature.DefaultFeatureGate.Enabled(features.AttachVolumeLimit) {
				invalidPredicates.Insert(predicates.MaxCSIVolumeCountPred)
			}
		} else {
			// We do not consider CSI volumes here because CSI
			// volumes can not be used inline.
			if vol.AWSElasticBlockStore != nil {
				invalidPredicates.Insert(predicates.MaxEBSVolumeCountPred)
			}
			if vol.GCEPersistentDisk != nil {
				invalidPredicates.Insert(predicates.MaxGCEPDVolumeCountPred)
			}
			if vol.AzureDisk != nil {
				invalidPredicates.Insert(predicates.MaxAzureDiskVolumeCountPred)
			}
		}
	}
	c.InvalidatePredicatesOnNode(nodeName, invalidPredicates)
}

// Class represents a set of pods which are equivalent from the perspective of
// the scheduler. i.e. the scheduler would make the same decision for any pod
// from the same class.
type Class struct {
	// Equivalence hash
	hash uint64
}

// NewClass returns the equivalence class for a given Pod. The returned Class
// objects will be equal for two Pods in the same class. nil values should not
// be considered equal to each other.
//
// NOTE: Make sure to compare types of Class and not *Class.
// TODO(misterikkit): Return error instead of nil *Class.
func NewClass(pod *v1.Pod) *Class {
	equivalencePod := getEquivalencePod(pod)
	if equivalencePod != nil {
		hash := fnv.New32a()
		hashutil.DeepHashObject(hash, equivalencePod)
		return &Class{
			hash: uint64(hash.Sum32()),
		}
	}
	return nil
}

// predicateMap stores resultMaps with predicate name as the key.
type predicateMap map[string]resultMap

// resultMap stores PredicateResult with pod equivalence hash as the key.
type resultMap map[uint64]predicateResult

// predicateResult stores the output of a FitPredicate.
type predicateResult struct {
	Fit         bool
	FailReasons []algorithm.PredicateFailureReason
}

// RunPredicate returns a cached predicate result. In case of a cache miss, the predicate will be
// run and its results cached for the next call.
//
// NOTE: RunPredicate will not update the equivalence cache if the given NodeInfo is stale.
func (n *NodeCache) RunPredicate(
	pred algorithm.FitPredicate,
	predicateKey string,
	pod *v1.Pod,
	meta algorithm.PredicateMetadata,
	nodeInfo *schedulercache.NodeInfo,
	equivClass *Class,
	cache schedulercache.Cache,
) (bool, []algorithm.PredicateFailureReason, error) {
	if nodeInfo == nil || nodeInfo.Node() == nil {
		// This may happen during tests.
		return false, []algorithm.PredicateFailureReason{}, fmt.Errorf("nodeInfo is nil or node is invalid")
	}

	result, ok := n.lookupResult(pod.GetName(), nodeInfo.Node().GetName(), predicateKey, equivClass.hash)
	if ok {
		return result.Fit, result.FailReasons, nil
	}
	fit, reasons, err := pred(pod, meta, nodeInfo)
	if err != nil {
		return fit, reasons, err
	}
	if cache != nil {
		n.updateResult(pod.GetName(), predicateKey, fit, reasons, equivClass.hash, cache, nodeInfo)
	}
	return fit, reasons, nil
}

// updateResult updates the cached result of a predicate.
func (n *NodeCache) updateResult(
	podName, predicateKey string,
	fit bool,
	reasons []algorithm.PredicateFailureReason,
	equivalenceHash uint64,
	cache schedulercache.Cache,
	nodeInfo *schedulercache.NodeInfo,
) {
	if nodeInfo == nil || nodeInfo.Node() == nil {
		// This may happen during tests.
		metrics.EquivalenceCacheWrites.WithLabelValues("discarded_bad_node").Inc()
		return
	}
	// Skip update if NodeInfo is stale.
	if !cache.IsUpToDate(nodeInfo) {
		metrics.EquivalenceCacheWrites.WithLabelValues("discarded_stale").Inc()
		return
	}

	predicateItem := predicateResult{
		Fit:         fit,
		FailReasons: reasons,
	}

	n.mu.Lock()
	defer n.mu.Unlock()
	// If cached predicate map already exists, just update the predicate by key
	if predicates, ok := n.cache[predicateKey]; ok {
		// maps in golang are references, no need to add them back
		predicates[equivalenceHash] = predicateItem
	} else {
		n.cache[predicateKey] =
			resultMap{
				equivalenceHash: predicateItem,
			}
	}

	glog.V(5).Infof("Cache update: node=%s, predicate=%s,pod=%s,value=%v",
		nodeInfo.Node().Name, predicateKey, podName, predicateItem)
}

// lookupResult returns cached predicate results and a bool saying whether a
// cache entry was found.
func (n *NodeCache) lookupResult(
	podName, nodeName, predicateKey string,
	equivalenceHash uint64,
) (value predicateResult, ok bool) {
	n.mu.RLock()
	defer n.mu.RUnlock()
	value, ok = n.cache[predicateKey][equivalenceHash]
	if ok {
		metrics.EquivalenceCacheHits.Inc()
	} else {
		metrics.EquivalenceCacheMisses.Inc()
	}
	return value, ok
}

// invalidatePreds deletes cached predicates by given keys.
func (n *NodeCache) invalidatePreds(predicateKeys sets.String) {
	n.mu.Lock()
	defer n.mu.Unlock()
	for predicateKey := range predicateKeys {
		delete(n.cache, predicateKey)
	}
}

// equivalencePod is the set of pod attributes which must match for two pods to
// be considered equivalent for scheduling purposes. For correctness, this must
// include any Pod field which is used by a FitPredicate.
//
// NOTE: For equivalence hash to be formally correct, lists and maps in the
// equivalencePod should be normalized. (e.g. by sorting them) However, the vast
// majority of equivalent pod classes are expected to be created from a single
// pod template, so they will all have the same ordering.
type equivalencePod struct {
	Namespace      *string
	Labels         map[string]string
	Affinity       *v1.Affinity
	Containers     []v1.Container // See note about ordering
	InitContainers []v1.Container // See note about ordering
	NodeName       *string
	NodeSelector   map[string]string
	Tolerations    []v1.Toleration
	Volumes        []v1.Volume // See note about ordering
}

// getEquivalencePod returns a normalized representation of a pod so that two
// "equivalent" pods will hash to the same value.
func getEquivalencePod(pod *v1.Pod) *equivalencePod {
	ep := &equivalencePod{
		Namespace:      &pod.Namespace,
		Labels:         pod.Labels,
		Affinity:       pod.Spec.Affinity,
		Containers:     pod.Spec.Containers,
		InitContainers: pod.Spec.InitContainers,
		NodeName:       &pod.Spec.NodeName,
		NodeSelector:   pod.Spec.NodeSelector,
		Tolerations:    pod.Spec.Tolerations,
		Volumes:        pod.Spec.Volumes,
	}
	// DeepHashObject considers nil and empty slices to be different. Normalize them.
	if len(ep.Containers) == 0 {
		ep.Containers = nil
	}
	if len(ep.InitContainers) == 0 {
		ep.InitContainers = nil
	}
	if len(ep.Tolerations) == 0 {
		ep.Tolerations = nil
	}
	if len(ep.Volumes) == 0 {
		ep.Volumes = nil
	}
	// Normalize empty maps also.
	if len(ep.Labels) == 0 {
		ep.Labels = nil
	}
	if len(ep.NodeSelector) == 0 {
		ep.NodeSelector = nil
	}
	// TODO(misterikkit): Also normalize nested maps and slices.
	return ep
}
