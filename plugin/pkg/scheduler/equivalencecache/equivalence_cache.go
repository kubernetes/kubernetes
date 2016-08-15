/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package equivalencecache

import (
	//	"github.com/golang/glog"
	"github.com/golang/groupcache/lru"
	"hash/adler32"

	"k8s.io/kubernetes/pkg/api"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"sync"
)

const maxCacheEntries = 4096

type HostPredicate struct {
	Fit        bool
	FailReason sets.String
}

type podCacheEnty struct {
	Fit        bool
	FailReason sets.String
	Score      schedulerapi.HostPriority
}

type AlgorithmCache struct {
	predicatesCache *lru.Cache
	prioritiesCache *lru.Cache
}

func newAlgorithmCache() AlgorithmCache {
	return AlgorithmCache{
		predicatesCache: lru.New(maxCacheEntries),
		prioritiesCache: lru.New(maxCacheEntries),
	}
}

// Store a map of predicate cache with maxsize
type EquivalenceCache struct {
	getEquivalencePod algorithm.GetEquivalencePodFunc
	algorithmCache    map[string]AlgorithmCache
	//	realCacheLock     *sync.RWMutex

	invalidAlgorithmCacheList sets.String
	allCacheExpired           bool
	expireLock                *sync.RWMutex
}

func NewEquivalenceCache(getEquivalencePodFunc algorithm.GetEquivalencePodFunc) *EquivalenceCache {
	return &EquivalenceCache{
		getEquivalencePod: getEquivalencePodFunc,
		algorithmCache:    make(map[string]AlgorithmCache),
		//		realCacheLock:         new(sync.RWMutex),
		invalidAlgorithmCacheList: sets.NewString(),
		allCacheExpired:           false,
		expireLock:                new(sync.RWMutex),
	}
}

func (ec *EquivalenceCache) addPodPriority(podKey uint64, nodeName string, score int) {
	if _, exist := ec.algorithmCache[nodeName]; !exist {
		ec.algorithmCache[nodeName] = newAlgorithmCache()
	}
	ec.algorithmCache[nodeName].prioritiesCache.Add(podKey, schedulerapi.HostPriority{Host: nodeName, Score: score})
}

func (ec *EquivalenceCache) addPodPredicate(podKey uint64, nodeName string, fit bool, failReason sets.String) {
	if _, exist := ec.algorithmCache[nodeName]; !exist {
		ec.algorithmCache[nodeName] = newAlgorithmCache()
	}
	ec.algorithmCache[nodeName].predicatesCache.Add(podKey, HostPredicate{Fit: fit, FailReason: failReason})
}

func (ec *EquivalenceCache) AddPodPredicatesCache(pod *api.Pod, fitNodeList *api.NodeList, failedPredicates *FailedPredicateMap) {
	equivalenceHash := ec.hashEquivalencePod(pod)

	for _, fitNode := range fitNodeList.Items {
		ec.addPodPredicate(equivalenceHash, fitNode.Name, true, sets.String{})
	}
	for failNodeName, failReason := range *failedPredicates {
		ec.addPodPredicate(equivalenceHash, failNodeName, false, failReason)
	}
}

func (ec *EquivalenceCache) AddPodPrioritiesCache(pod *api.Pod, priorities schedulerapi.HostPriorityList) {
	equivalenceHash := ec.hashEquivalencePod(pod)

	for _, priority := range priorities {
		ec.addPodPriority(equivalenceHash, priority.Host, priority.Score)
	}
}

func (ec *EquivalenceCache) GetCachedPredicates(pod *api.Pod, nodes api.NodeList) (api.NodeList, FailedPredicateMap, api.NodeList) {
	fitNodeList := api.NodeList{}
	failedPredicates := FailedPredicateMap{}
	noCacheNodeList := api.NodeList{}

	equivalenceHash := ec.hashEquivalencePod(pod)

	for _, node := range nodes.Items {
		findCache := false
		if algorithmCache, exist := ec.algorithmCache[node.Name]; exist {
			if cachePredicate, exist := algorithmCache.predicatesCache.Get(equivalenceHash); exist {
				hostPredicate := cachePredicate.(HostPredicate)
				if hostPredicate.Fit {
					fitNodeList.Items = append(fitNodeList.Items, node)
				} else {
					failedPredicates[node.Name] = hostPredicate.FailReason
				}
				findCache = true
			}
		}
		if !findCache {
			noCacheNodeList.Items = append(noCacheNodeList.Items, node)
		}
	}
	//	glog.Infof("Get predicate cache: %v ----%v, nodes: %v has no cache date.", fitNodeList.Items, failedPredicates, noCacheNodeList.Items)
	return fitNodeList, failedPredicates, noCacheNodeList
}

func (ec *EquivalenceCache) GetCachedPriorities(pod *api.Pod, nodes api.NodeList) (schedulerapi.HostPriorityList, api.NodeList) {
	cachedPriorities := schedulerapi.HostPriorityList{}
	noCacheNodeList := api.NodeList{}

	equivalenceHash := ec.hashEquivalencePod(pod)

	for _, node := range nodes.Items {
		findCache := false
		if algorithmCache, exist := ec.algorithmCache[node.Name]; exist {
			if cachePriority, exist := algorithmCache.prioritiesCache.Get(equivalenceHash); exist {
				hostPriotity := cachePriority.(schedulerapi.HostPriority)
				cachedPriorities = append(cachedPriorities, hostPriotity)
				findCache = true
			}
		}
		if !findCache {
			noCacheNodeList.Items = append(noCacheNodeList.Items, node)
		}
	}

	return cachedPriorities, noCacheNodeList
}

func (ec *EquivalenceCache) SendInvalidAlgorithmCacheReq(nodeName string) {
	ec.expireLock.RLock()
	allExpired := ec.allCacheExpired
	ec.expireLock.RUnlock()

	if !allExpired {
		ec.expireLock.Lock()
		defer ec.expireLock.Unlock()
		ec.invalidAlgorithmCacheList.Insert(nodeName)
	}
}

func (ec *EquivalenceCache) SendClearAllCacheReq() {
	ec.expireLock.RLock()
	allExpired := ec.allCacheExpired
	ec.expireLock.RUnlock()

	if !allExpired {
		ec.expireLock.Lock()
		ec.allCacheExpired = true
		ec.expireLock.Unlock()
	}
}

func (ec *EquivalenceCache) HandleExpireDate() {
	ec.expireLock.Lock()
	defer ec.expireLock.Unlock()

	// Remove expired AlgorithmCache
	if ec.allCacheExpired {
		ec.algorithmCache = make(map[string]AlgorithmCache)
	} else {
		for _, node := range ec.invalidAlgorithmCacheList.List() {
			delete(ec.algorithmCache, node)
		}
	}

	// Clear expired data records for next cycle
	ec.invalidAlgorithmCacheList = sets.NewString()
	ec.allCacheExpired = false
}

// hashEquivalencePod returns the hash of equivalence pod.
func (ec *EquivalenceCache) hashEquivalencePod(pod *api.Pod) uint64 {
	equivalencePod := ec.getEquivalencePod(pod)
	hash := adler32.New()
	hashutil.DeepHashObject(hash, equivalencePod)
	return uint64(hash.Sum32())
}
