/*
Copyright 2022 The Kubernetes Authors.

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

package dra

import (
	"fmt"
	"sync"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// claimInfo holds information required
// to prepare and unprepare a resource claim.
type claimInfo struct {
	sync.RWMutex

	// name of the DRA driver
	driverName string

	// claimUID is an UID of the resource claim
	claimUID types.UID

	// claimName is a name of the resource claim
	claimName string

	// namespace is a claim namespace
	namespace string

	// podUIDs is a set of pod UIDs that reference a resource
	podUIDs sets.Set[string]

	// cdiDevices is a list of CDI devices returned by the
	// GRPC API call NodePrepareResource
	cdiDevices []string

	// annotations is a list of container annotations associated with
	// a prepared resource
	annotations []kubecontainer.Annotation
}

func (res *claimInfo) addPodReference(podUID types.UID) {
	res.Lock()
	defer res.Unlock()

	res.podUIDs.Insert(string(podUID))
}

func (res *claimInfo) deletePodReference(podUID types.UID) {
	res.Lock()
	defer res.Unlock()

	res.podUIDs.Delete(string(podUID))
}

// claimInfoCache is a cache of processed resource claims keyed by namespace + claim name.
type claimInfoCache struct {
	sync.RWMutex
	claimInfo map[string]*claimInfo
}

// newClaimInfoCache is a function that returns an instance of the claimInfoCache.
func newClaimInfoCache() *claimInfoCache {
	return &claimInfoCache{
		claimInfo: make(map[string]*claimInfo),
	}
}

func (cache *claimInfoCache) add(claim, namespace string, res *claimInfo) error {
	cache.Lock()
	defer cache.Unlock()

	key := claim + namespace
	if _, ok := cache.claimInfo[key]; ok {
		return fmt.Errorf("claim %s, namespace %s already cached", claim, namespace)
	}

	cache.claimInfo[claim+namespace] = res

	return nil
}

func (cache *claimInfoCache) get(claimName, namespace string) *claimInfo {
	cache.RLock()
	defer cache.RUnlock()

	return cache.claimInfo[claimName+namespace]
}

func (cache *claimInfoCache) delete(claimName, namespace string) {
	cache.Lock()
	defer cache.Unlock()

	delete(cache.claimInfo, claimName+namespace)
}

// hasPodReference checks if there is at least one claim
// that is referenced by the pod with the given UID
// This function is used indirectly by the status manager
// to check if pod can enter termination status
func (cache *claimInfoCache) hasPodReference(UID types.UID) bool {
	cache.RLock()
	defer cache.RUnlock()

	for _, claimInfo := range cache.claimInfo {
		if claimInfo.podUIDs.Has(string(UID)) {
			return true
		}
	}

	return false
}
