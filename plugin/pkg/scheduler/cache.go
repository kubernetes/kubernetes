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

package scheduler

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

type NodeCache struct {
	predicatesCache *lru.Cache
	prioritiesCache *lru.Cache
}

func newNodeCache() NodeCache {
	return NodeCache{
		predicatesCache: lru.New(maxCacheEntries),
		prioritiesCache: lru.New(maxCacheEntries),
	}
}

// Store a map of predicate cache with maxsize
type SchedulerCache struct {
	getEquivalencePod algorithm.GetEquivalencePodFunc
	nodesCache        map[string]NodeCache
	//	realCacheLock     *sync.RWMutex

	invalidNodeCacheList sets.String
	allCacheExpired      bool
	expireLock           *sync.RWMutex
}

func NewSchedulerCache(getEquivalencePodFunc algorithm.GetEquivalencePodFunc) *SchedulerCache {
	return &SchedulerCache{
		getEquivalencePod: getEquivalencePodFunc,
		nodesCache:        make(map[string]NodeCache),
		//		realCacheLock:         new(sync.RWMutex),
		invalidNodeCacheList: sets.NewString(),
		allCacheExpired:      false,
		expireLock:           new(sync.RWMutex),
	}
}

func (sc *SchedulerCache) addPodPriority(podKey uint64, nodeName string, score int) {
	if _, exist := sc.nodesCache[nodeName]; !exist {
		sc.nodesCache[nodeName] = newNodeCache()
	}
	sc.nodesCache[nodeName].prioritiesCache.Add(podKey, schedulerapi.HostPriority{Host: nodeName, Score: score})
}

func (sc *SchedulerCache) addPodPredicate(podKey uint64, nodeName string, fit bool, failReason sets.String) {
	if _, exist := sc.nodesCache[nodeName]; !exist {
		sc.nodesCache[nodeName] = newNodeCache()
	}
	sc.nodesCache[nodeName].predicatesCache.Add(podKey, HostPredicate{Fit: fit, FailReason: failReason})
}

func (sc *SchedulerCache) AddPodPredicatesCache(pod *api.Pod, fitNodeList *api.NodeList, failedPredicates *FailedPredicateMap) {
	equivalenceHash := sc.hashEquivalencePod(pod)

	for _, fitNode := range fitNodeList.Items {
		sc.addPodPredicate(equivalenceHash, fitNode.Name, true, sets.String{})
	}
	for failNodeName, failReason := range *failedPredicates {
		sc.addPodPredicate(equivalenceHash, failNodeName, false, failReason)
	}
}

func (sc *SchedulerCache) AddPodPrioritiesCache(pod *api.Pod, priorities schedulerapi.HostPriorityList) {
	equivalenceHash := sc.hashEquivalencePod(pod)

	for _, priority := range priorities {
		sc.addPodPriority(equivalenceHash, priority.Host, priority.Score)
	}
}

func (sc *SchedulerCache) GetCachedPredicates(pod *api.Pod, nodes api.NodeList) (api.NodeList, FailedPredicateMap, api.NodeList) {
	fitNodeList := api.NodeList{}
	failedPredicates := FailedPredicateMap{}
	noCacheNodeList := api.NodeList{}

	equivalenceHash := sc.hashEquivalencePod(pod)

	for _, node := range nodes.Items {
		findCache := false
		if nodeCache, exist := sc.nodesCache[node.Name]; exist {
			if cachePredicate, exist := nodeCache.predicatesCache.Get(equivalenceHash); exist {
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

func (sc *SchedulerCache) GetCachedPriorities(pod *api.Pod, nodes api.NodeList) (schedulerapi.HostPriorityList, api.NodeList) {
	cachedPriorities := schedulerapi.HostPriorityList{}
	noCacheNodeList := api.NodeList{}

	equivalenceHash := sc.hashEquivalencePod(pod)

	for _, node := range nodes.Items {
		findCache := false
		if nodeCache, exist := sc.nodesCache[node.Name]; exist {
			if cachePriority, exist := nodeCache.prioritiesCache.Get(equivalenceHash); exist {
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

func (sc *SchedulerCache) SendInvalidNodeCacheReq(nodeName string) {
	sc.expireLock.RLock()
	allExpired := sc.allCacheExpired
	sc.expireLock.RUnlock()

	if !allExpired {
		sc.expireLock.Lock()
		defer sc.expireLock.Unlock()
		sc.invalidNodeCacheList.Insert(nodeName)
	}
}

func (sc *SchedulerCache) SendClearAllCacheReq() {
	sc.expireLock.RLock()
	allExpired := sc.allCacheExpired
	sc.expireLock.RUnlock()

	if !allExpired {
		sc.expireLock.Lock()
		sc.allCacheExpired = true
		sc.expireLock.Unlock()
	}
}

func (sc *SchedulerCache) HandleExpireDate() {
	sc.expireLock.Lock()
	defer sc.expireLock.Unlock()

	// Remove expired NodeCache
	if sc.allCacheExpired {
		sc.nodesCache = make(map[string]NodeCache)
	} else {
		for _, node := range sc.invalidNodeCacheList.List() {
			delete(sc.nodesCache, node)
		}
	}

	// Clear expired data records for next cycle
	sc.invalidNodeCacheList = sets.NewString()
	sc.allCacheExpired = false
}

// hashEquivalencePod returns the hash of equivalence pod.
func (sc *SchedulerCache) hashEquivalencePod(pod *api.Pod) uint64 {
	equivalencePod := sc.getEquivalencePod(pod)
	hash := adler32.New()
	hashutil.DeepHashObject(hash, equivalencePod)
	return uint64(hash.Sum32())
}
