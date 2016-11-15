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

package scheduler

import (
	"github.com/golang/groupcache/lru"
	"hash/adler32"

	"k8s.io/kubernetes/pkg/api"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"sync"
)

// TODO(harryz) figure out the right number for this, 4096 may be too big
const maxCacheEntries = 4096

type HostPredicate struct {
	Fit         bool
	FailReasons []algorithm.PredicateFailureReason
}

type AlgorithmCache struct {
	// Only consider predicates for now, priorities rely on: #31606
	predicatesCache *lru.Cache
}

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

// addPodPredicate adds pod predicate for equivalence class
func (ec *EquivalenceCache) addPodPredicate(podKey uint64, nodeName string, fit bool, failReasons []algorithm.PredicateFailureReason) {
	if _, exist := ec.algorithmCache[nodeName]; !exist {
		ec.algorithmCache[nodeName] = newAlgorithmCache()
	}
	ec.algorithmCache[nodeName].predicatesCache.Add(podKey, HostPredicate{Fit: fit, FailReasons: failReasons})
}

// AddPodPredicatesCache cache pod predicate for equivalence class
func (ec *EquivalenceCache) AddPodPredicatesCache(pod *api.Pod, fitNodeList []*api.Node, failedPredicates *FailedPredicateMap) {
	equivalenceHash := ec.hashEquivalencePod(pod)

	for _, fitNode := range fitNodeList {
		ec.addPodPredicate(equivalenceHash, fitNode.Name, true, nil)
	}
	for failNodeName, failReasons := range *failedPredicates {
		ec.addPodPredicate(equivalenceHash, failNodeName, false, failReasons)
	}
}

// GetCachedPredicates gets cached predicates for equivalence class
func (ec *EquivalenceCache) GetCachedPredicates(pod *api.Pod, nodes []*api.Node) ([]*api.Node, FailedPredicateMap, []*api.Node) {
	fitNodeList := []*api.Node{}
	failedPredicates := FailedPredicateMap{}
	noCacheNodeList := []*api.Node{}
	equivalenceHash := ec.hashEquivalencePod(pod)
	for _, node := range nodes {
		findCache := false
		if algorithmCache, exist := ec.algorithmCache[node.Name]; exist {
			if cachePredicate, exist := algorithmCache.predicatesCache.Get(equivalenceHash); exist {
				hostPredicate := cachePredicate.(HostPredicate)
				if hostPredicate.Fit {
					fitNodeList = append(fitNodeList, node)
				} else {
					failedPredicates[node.Name] = hostPredicate.FailReasons
				}
				findCache = true
			}
		}
		if !findCache {
			noCacheNodeList = append(noCacheNodeList, node)
		}
	}
	return fitNodeList, failedPredicates, noCacheNodeList
}

// SendInvalidAlgorithmCacheReq marks AlgorithmCache item as invalid
func (ec *EquivalenceCache) SendInvalidAlgorithmCacheReq(nodeName string) {
	ec.Lock()
	defer ec.Unlock()
	// clear the cache of this node
	delete(ec.algorithmCache, nodeName)
}

// SendClearAllCacheReq marks all cached item as invalid
func (ec *EquivalenceCache) SendClearAllCacheReq() {
	ec.Lock()
	defer ec.Unlock()
	// clear cache of all nodes
	for nodeName := range ec.algorithmCache {
		delete(ec.algorithmCache, nodeName)
	}
}

// hashEquivalencePod returns the hash of equivalence pod.
func (ec *EquivalenceCache) hashEquivalencePod(pod *api.Pod) uint64 {
	equivalencePod := ec.getEquivalencePod(pod)
	hash := adler32.New()
	hashutil.DeepHashObject(hash, equivalencePod)
	return uint64(hash.Sum32())
}
