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

package core

import (
	"hash/fnv"
	"sync"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"

	"github.com/golang/glog"
	"github.com/golang/groupcache/lru"
)

// we use predicate names as cache's key, its count is limited
const maxCacheEntries = 100

type HostPredicate struct {
	Fit         bool
	FailReasons []algorithm.PredicateFailureReason
}

type AlgorithmCache struct {
	// Only consider predicates for now, priorities rely on: #31606
	predicatesCache *lru.Cache
}

// PredicateMap use equivalence hash as key
type PredicateMap map[uint64]HostPredicate

func newAlgorithmCache() AlgorithmCache {
	return AlgorithmCache{
		predicatesCache: lru.New(maxCacheEntries),
	}
}

// Store a map of predicate cache with maxsize
type EquivalenceCache struct {
	sync.RWMutex
	getEquivalencePod algorithm.GetEquivalencePodFunc
	algorithmCache    map[string]AlgorithmCache
}

func NewEquivalenceCache(getEquivalencePodFunc algorithm.GetEquivalencePodFunc) *EquivalenceCache {
	return &EquivalenceCache{
		getEquivalencePod: getEquivalencePodFunc,
		algorithmCache:    make(map[string]AlgorithmCache),
	}
}

// UpdateCachedPredicateItem updates pod predicate for equivalence class
func (ec *EquivalenceCache) UpdateCachedPredicateItem(podName, nodeName, predicateKey string, fit bool, reasons []algorithm.PredicateFailureReason, equivalenceHash uint64) {
	ec.Lock()
	defer ec.Unlock()
	if _, exist := ec.algorithmCache[nodeName]; !exist {
		ec.algorithmCache[nodeName] = newAlgorithmCache()
	}
	predicateItem := HostPredicate{
		Fit:         fit,
		FailReasons: reasons,
	}
	// if cached predicate map already exists, just update the predicate by key
	if v, ok := ec.algorithmCache[nodeName].predicatesCache.Get(predicateKey); ok {
		predicateMap := v.(PredicateMap)
		// maps in golang are references, no need to add them back
		predicateMap[equivalenceHash] = predicateItem
	} else {
		ec.algorithmCache[nodeName].predicatesCache.Add(predicateKey,
			PredicateMap{
				equivalenceHash: predicateItem,
			})
	}
	glog.V(5).Infof("Updated cached predicate: %v for pod: %v on node: %s, with item %v", predicateKey, podName, nodeName, predicateItem)
}

// PredicateWithECache returns:
// 1. if fit
// 2. reasons if not fit
// 3. if this cache is invalid
// based on cached predicate results
func (ec *EquivalenceCache) PredicateWithECache(podName, nodeName, predicateKey string, equivalenceHash uint64) (bool, []algorithm.PredicateFailureReason, bool) {
	ec.RLock()
	defer ec.RUnlock()
	glog.V(5).Infof("Begin to calculate predicate: %v for pod: %s on node: %s based on equivalence cache", predicateKey, podName, nodeName)
	if algorithmCache, exist := ec.algorithmCache[nodeName]; exist {
		if cachePredicate, exist := algorithmCache.predicatesCache.Get(predicateKey); exist {
			predicateMap := cachePredicate.(PredicateMap)
			// TODO(resouer) Is it possible a race that cache failed to update immediately?
			if hostPredicate, ok := predicateMap[equivalenceHash]; ok {
				if hostPredicate.Fit {
					return true, []algorithm.PredicateFailureReason{}, false
				} else {
					return false, hostPredicate.FailReasons, false
				}
			} else {
				// is invalid
				return false, []algorithm.PredicateFailureReason{}, true
			}
		}
	}
	return false, []algorithm.PredicateFailureReason{}, true
}

// InvalidateCachedPredicateItem marks all items of given predicateKeys, of all pods, on the given node as invalid
func (ec *EquivalenceCache) InvalidateCachedPredicateItem(nodeName string, predicateKeys sets.String) {
	if len(predicateKeys) == 0 {
		return
	}
	ec.Lock()
	defer ec.Unlock()
	if algorithmCache, exist := ec.algorithmCache[nodeName]; exist {
		for predicateKey := range predicateKeys {
			algorithmCache.predicatesCache.Remove(predicateKey)
		}
	}
	glog.V(5).Infof("Done invalidating cached predicates: %v on node: %s", predicateKeys, nodeName)
}

// InvalidateCachedPredicateItemOfAllNodes marks all items of given predicateKeys, of all pods, on all node as invalid
func (ec *EquivalenceCache) InvalidateCachedPredicateItemOfAllNodes(predicateKeys sets.String) {
	if len(predicateKeys) == 0 {
		return
	}
	ec.Lock()
	defer ec.Unlock()
	// algorithmCache uses nodeName as key, so we just iterate it and invalid given predicates
	for _, algorithmCache := range ec.algorithmCache {
		for predicateKey := range predicateKeys {
			// just use keys is enough
			algorithmCache.predicatesCache.Remove(predicateKey)
		}
	}
	glog.V(5).Infof("Done invalidating cached predicates: %v on all node", predicateKeys)
}

// InvalidateAllCachedPredicateItemOfNode marks all cached items on given node as invalid
func (ec *EquivalenceCache) InvalidateAllCachedPredicateItemOfNode(nodeName string) {
	ec.Lock()
	defer ec.Unlock()
	delete(ec.algorithmCache, nodeName)
	glog.V(5).Infof("Done invalidating all cached predicates on node: %s", nodeName)
}

// InvalidateCachedPredicateItemForPodAdd is a wrapper of InvalidateCachedPredicateItem for pod add case
func (ec *EquivalenceCache) InvalidateCachedPredicateItemForPodAdd(pod *v1.Pod, nodeName string) {
	// MatchInterPodAffinity: we assume scheduler can make sure newly binded pod
	// will not break the existing inter pod affinity. So we does not need to invalidate
	// MatchInterPodAffinity when pod added.
	//
	// But when a pod is deleted, existing inter pod affinity may become invalid.
	// (e.g. this pod was preferred by some else, or vice versa)
	//
	// NOTE: assumptions above will not stand when we implemented features like
	// RequiredDuringSchedulingRequiredDuringExecution.

	// NoDiskConflict: the newly scheduled pod fits to existing pods on this node,
	// it will also fits to equivalence class of existing pods

	// GeneralPredicates: will always be affected by adding a new pod
	invalidPredicates := sets.NewString("GeneralPredicates")
	ec.InvalidateCachedPredicateItem(nodeName, invalidPredicates)
}

// getHashEquivalencePod returns the hash of equivalence pod.
// if no equivalence pod found, return 0
func (ec *EquivalenceCache) getHashEquivalencePod(pod *v1.Pod) uint64 {
	equivalencePod := ec.getEquivalencePod(pod)
	if equivalencePod != nil {
		hash := fnv.New32a()
		hashutil.DeepHashObject(hash, equivalencePod)
		return uint64(hash.Sum32())
	}
	return 0
}
