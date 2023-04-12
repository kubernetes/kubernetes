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
	"k8s.io/kubernetes/pkg/kubelet/cm/dra/state"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// ClaimInfo holds information required
// to prepare and unprepare a resource claim.
type ClaimInfo struct {
	sync.RWMutex
	state.ClaimInfoState
	// annotations is a list of container annotations associated with
	// a prepared resource
	annotations []kubecontainer.Annotation
}

func (info *ClaimInfo) addPodReference(podUID types.UID) {
	info.Lock()
	defer info.Unlock()

	info.PodUIDs.Insert(string(podUID))
}

func (info *ClaimInfo) deletePodReference(podUID types.UID) {
	info.Lock()
	defer info.Unlock()

	info.PodUIDs.Delete(string(podUID))
}

func (info *ClaimInfo) addCDIDevices(pluginName string, cdiDevices []string) error {
	info.Lock()
	defer info.Unlock()

	// NOTE: Passing CDI device names as annotations is a temporary solution
	// It will be removed after all runtimes are updated
	// to get CDI device names from the ContainerConfig.CDIDevices field
	annotations, err := generateCDIAnnotations(info.ClaimUID, info.DriverName, cdiDevices)
	if err != nil {
		return fmt.Errorf("failed to generate container annotations, err: %+v", err)
	}

	if info.CDIDevices == nil {
		info.CDIDevices = make(map[string][]string)
	}

	info.CDIDevices[pluginName] = cdiDevices
	info.annotations = append(info.annotations, annotations...)

	return nil
}

// claimInfoCache is a cache of processed resource claims keyed by namespace + claim name.
type claimInfoCache struct {
	sync.RWMutex
	state     state.CheckpointState
	claimInfo map[string]*ClaimInfo
}

func newClaimInfo(driverName, className string, claimUID types.UID, claimName, namespace string, podUIDs sets.Set[string]) *ClaimInfo {
	claimInfoState := state.ClaimInfoState{
		DriverName: driverName,
		ClassName:  className,
		ClaimUID:   claimUID,
		ClaimName:  claimName,
		Namespace:  namespace,
		PodUIDs:    podUIDs,
	}
	claimInfo := ClaimInfo{
		ClaimInfoState: claimInfoState,
	}
	return &claimInfo
}

// newClaimInfoCache is a function that returns an instance of the claimInfoCache.
func newClaimInfoCache(stateDir, checkpointName string) (*claimInfoCache, error) {
	stateImpl, err := state.NewCheckpointState(stateDir, checkpointName)
	if err != nil {
		return nil, fmt.Errorf("could not initialize checkpoint manager, please drain node and remove dra state file, err: %+v", err)
	}

	curState, err := stateImpl.GetOrCreate()
	if err != nil {
		return nil, fmt.Errorf("error calling GetOrCreate() on checkpoint state: %v", err)
	}

	cache := &claimInfoCache{
		state:     stateImpl,
		claimInfo: make(map[string]*ClaimInfo),
	}

	for _, entry := range curState {
		info := newClaimInfo(
			entry.DriverName,
			entry.ClassName,
			entry.ClaimUID,
			entry.ClaimName,
			entry.Namespace,
			entry.PodUIDs,
		)
		for pluginName, cdiDevices := range entry.CDIDevices {
			err := info.addCDIDevices(pluginName, cdiDevices)
			if err != nil {
				return nil, fmt.Errorf("failed to add CDIDevices to claimInfo %+v: %+v", info, err)
			}
		}
		cache.add(info)
	}

	return cache, nil
}

func (cache *claimInfoCache) add(res *ClaimInfo) {
	cache.Lock()
	defer cache.Unlock()

	cache.claimInfo[res.ClaimName+res.Namespace] = res
}

func (cache *claimInfoCache) get(claimName, namespace string) *ClaimInfo {
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
		if claimInfo.PodUIDs.Has(string(UID)) {
			return true
		}
	}

	return false
}

func (cache *claimInfoCache) syncToCheckpoint() error {
	cache.RLock()
	defer cache.RUnlock()

	claimInfoStateList := make(state.ClaimInfoStateList, 0, len(cache.claimInfo))
	for _, infoClaim := range cache.claimInfo {
		claimInfoStateList = append(claimInfoStateList, infoClaim.ClaimInfoState)
	}

	return cache.state.Store(claimInfoStateList)
}
