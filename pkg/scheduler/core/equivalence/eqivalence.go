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

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
)

// nodeMap stores a *NodeCache for each node.
type nodeMap map[string]*NodeCache

// Cache is a thread safe map saves and reuses the output of predicate functions,
// it uses node name as key to access those cached results.
//
// Internally, results are keyed by predicate name, and "equivalence
// class". (Equivalence class is defined in the `Class` type.) Saved results
// will be reused until an appropriate invalidation function is called.
type Cache struct {
	// NOTE(harry): Theoretically sync.Map has better performance in machine with 8+ CPUs, while
	// the reality is lock contention in first level cache is rare.
	mu             sync.RWMutex
	nodeToCache    nodeMap
	predicateIDMap map[string]int
}

// NewCache create an empty equiv class cache.
func NewCache(predicates []string) *Cache {
	predicateIDMap := make(map[string]int, len(predicates))
	for id, predicate := range predicates {
		predicateIDMap[predicate] = id
	}
	return &Cache{
		nodeToCache:    make(nodeMap),
		predicateIDMap: predicateIDMap,
	}
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
	// generation is current generation of node cache, incremented on node
	// invalidation.
	generation uint64
	// snapshotGeneration saves snapshot of generation of node cache.
	snapshotGeneration uint64
	// predicateGenerations stores generation numbers for predicates, incremented on
	// predicate invalidation. Created on first update. Use 0 if does not
	// exist.
	predicateGenerations []uint64
	// snapshotPredicateGenerations saves snapshot of generation numbers for predicates.
	snapshotPredicateGenerations []uint64
}

// newNodeCache returns an empty NodeCache.
func newNodeCache(n int) *NodeCache {
	return &NodeCache{
		cache:                        make(predicateMap, n),
		predicateGenerations:         make([]uint64, n),
		snapshotPredicateGenerations: make([]uint64, n),
	}
}

// Snapshot snapshots current generations of cache.
// NOTE: We snapshot generations of all node caches before using it and these
// operations are serialized, we can save snapshot as member of node cache
// itself.
func (c *Cache) Snapshot() {
	c.mu.RLock()
	defer c.mu.RUnlock()
	for _, n := range c.nodeToCache {
		n.mu.Lock()
		// snapshot predicate generations
		copy(n.snapshotPredicateGenerations, n.predicateGenerations)
		// snapshot node generation
		n.snapshotGeneration = n.generation
		n.mu.Unlock()
	}
	return
}

// GetNodeCache returns the existing NodeCache for given node if present. Otherwise,
// it creates the NodeCache and returns it.
// The boolean flag is true if the value was loaded, false if created.
func (c *Cache) GetNodeCache(name string) (nodeCache *NodeCache, exists bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if nodeCache, exists = c.nodeToCache[name]; !exists {
		nodeCache = newNodeCache(len(c.predicateIDMap))
		c.nodeToCache[name] = nodeCache
	}
	return
}

// LoadNodeCache returns the existing NodeCache for given node, nil if not
// present.
func (c *Cache) LoadNodeCache(node string) *NodeCache {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.nodeToCache[node]
}

func (c *Cache) predicateKeysToIDs(predicateKeys sets.String) []int {
	predicateIDs := make([]int, 0, len(predicateKeys))
	for predicateKey := range predicateKeys {
		if id, ok := c.predicateIDMap[predicateKey]; ok {
			predicateIDs = append(predicateIDs, id)
		} else {
			klog.Errorf("predicate key %q not found", predicateKey)
		}
	}
	return predicateIDs
}

// InvalidatePredicates clears all cached results for the given predicates.
func (c *Cache) InvalidatePredicates(predicateKeys sets.String) {
	if len(predicateKeys) == 0 {
		return
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	predicateIDs := c.predicateKeysToIDs(predicateKeys)
	for _, n := range c.nodeToCache {
		n.invalidatePreds(predicateIDs)
	}
	klog.V(5).Infof("Cache invalidation: node=*,predicates=%v", predicateKeys)

}

// InvalidatePredicatesOnNode clears cached results for the given predicates on one node.
func (c *Cache) InvalidatePredicatesOnNode(nodeName string, predicateKeys sets.String) {
	if len(predicateKeys) == 0 {
		return
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	predicateIDs := c.predicateKeysToIDs(predicateKeys)
	if n, ok := c.nodeToCache[nodeName]; ok {
		n.invalidatePreds(predicateIDs)
	}
	klog.V(5).Infof("Cache invalidation: node=%s,predicates=%v", nodeName, predicateKeys)
}

// InvalidateAllPredicatesOnNode clears all cached results for one node.
func (c *Cache) InvalidateAllPredicatesOnNode(nodeName string) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if node, ok := c.nodeToCache[nodeName]; ok {
		node.invalidate()
	}
	klog.V(5).Infof("Cache invalidation: node=%s,predicates=*", nodeName)
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

// predicateMap stores resultMaps with predicate ID as the key.
type predicateMap []resultMap

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
// NOTE: RunPredicate will not update the equivalence cache if generation does not match live version.
func (n *NodeCache) RunPredicate(
	pred algorithm.FitPredicate,
	predicateKey string,
	predicateID int,
	pod *v1.Pod,
	meta algorithm.PredicateMetadata,
	nodeInfo *schedulercache.NodeInfo,
	equivClass *Class,
) (bool, []algorithm.PredicateFailureReason, error) {
	if nodeInfo == nil || nodeInfo.Node() == nil {
		// This may happen during tests.
		return false, []algorithm.PredicateFailureReason{}, fmt.Errorf("nodeInfo is nil or node is invalid")
	}

	result, ok := n.lookupResult(pod.GetName(), nodeInfo.Node().GetName(), predicateKey, predicateID, equivClass.hash)
	if ok {
		return result.Fit, result.FailReasons, nil
	}
	fit, reasons, err := pred(pod, meta, nodeInfo)
	if err != nil {
		return fit, reasons, err
	}
	n.updateResult(pod.GetName(), predicateKey, predicateID, fit, reasons, equivClass.hash, nodeInfo)
	return fit, reasons, nil
}

// updateResult updates the cached result of a predicate.
func (n *NodeCache) updateResult(
	podName, predicateKey string,
	predicateID int,
	fit bool,
	reasons []algorithm.PredicateFailureReason,
	equivalenceHash uint64,
	nodeInfo *schedulercache.NodeInfo,
) {
	if nodeInfo == nil || nodeInfo.Node() == nil {
		// This may happen during tests.
		metrics.EquivalenceCacheWrites.WithLabelValues("discarded_bad_node").Inc()
		return
	}

	predicateItem := predicateResult{
		Fit:         fit,
		FailReasons: reasons,
	}

	n.mu.Lock()
	defer n.mu.Unlock()
	if (n.snapshotGeneration != n.generation) || (n.snapshotPredicateGenerations[predicateID] != n.predicateGenerations[predicateID]) {
		// Generation of node or predicate has been updated since we last took
		// a snapshot, this indicates that we received a invalidation request
		// during this time. Cache may be stale, skip update.
		metrics.EquivalenceCacheWrites.WithLabelValues("discarded_stale").Inc()
		return
	}
	// If cached predicate map already exists, just update the predicate by key
	if predicates := n.cache[predicateID]; predicates != nil {
		// maps in golang are references, no need to add them back
		predicates[equivalenceHash] = predicateItem
	} else {
		n.cache[predicateID] =
			resultMap{
				equivalenceHash: predicateItem,
			}
	}
	n.predicateGenerations[predicateID]++

	klog.V(5).Infof("Cache update: node=%s, predicate=%s,pod=%s,value=%v",
		nodeInfo.Node().Name, predicateKey, podName, predicateItem)
}

// lookupResult returns cached predicate results and a bool saying whether a
// cache entry was found.
func (n *NodeCache) lookupResult(
	podName, nodeName, predicateKey string,
	predicateID int,
	equivalenceHash uint64,
) (value predicateResult, ok bool) {
	n.mu.RLock()
	defer n.mu.RUnlock()
	value, ok = n.cache[predicateID][equivalenceHash]
	if ok {
		metrics.EquivalenceCacheHits.Inc()
	} else {
		metrics.EquivalenceCacheMisses.Inc()
	}
	return value, ok
}

// invalidatePreds deletes cached predicates by given IDs.
func (n *NodeCache) invalidatePreds(predicateIDs []int) {
	n.mu.Lock()
	defer n.mu.Unlock()
	for _, predicateID := range predicateIDs {
		n.cache[predicateID] = nil
		n.predicateGenerations[predicateID]++
	}
}

// invalidate invalidates node cache.
func (n *NodeCache) invalidate() {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.cache = make(predicateMap, len(n.cache))
	n.generation++
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
