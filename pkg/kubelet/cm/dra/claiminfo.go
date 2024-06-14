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

	resourceapi "k8s.io/api/resource/v1alpha3"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/kubelet/cm/dra/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/util/cdi"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// ClaimInfo holds information required
// to prepare and unprepare a resource claim.
// +k8s:deepcopy-gen=true
type ClaimInfo struct {
	state.ClaimInfoState
	// annotations is a mapping of container annotations per DRA plugin associated with
	// a prepared resource
	annotations map[string][]kubecontainer.Annotation
	prepared    bool
}

// claimInfoCache is a cache of processed resource claims keyed by namespace/claimname.
type claimInfoCache struct {
	sync.RWMutex
	state     state.CheckpointState
	claimInfo map[string]*ClaimInfo
}

// newClaimInfoFromClaim creates a new claim info from a resource claim.
func newClaimInfoFromClaim(claim *resourceapi.ResourceClaim) *ClaimInfo {
	// Grab the allocation.resourceHandles. If there are no
	// allocation.resourceHandles, create a single resourceHandle with no
	// content. This will trigger processing of this claim by a single
	// kubelet plugin whose name matches resourceClaim.Status.DriverName.
	resourceHandles := claim.Status.Allocation.ResourceHandles
	if len(resourceHandles) == 0 {
		resourceHandles = make([]resourceapi.ResourceHandle, 1)
	}
	claimInfoState := state.ClaimInfoState{
		DriverName:      claim.Status.DriverName,
		ClassName:       claim.Spec.ResourceClassName,
		ClaimUID:        claim.UID,
		ClaimName:       claim.Name,
		Namespace:       claim.Namespace,
		PodUIDs:         sets.New[string](),
		ResourceHandles: resourceHandles,
		CDIDevices:      make(map[string][]string),
	}
	info := &ClaimInfo{
		ClaimInfoState: claimInfoState,
		annotations:    make(map[string][]kubecontainer.Annotation),
		prepared:       false,
	}
	return info
}

// newClaimInfoFromClaim creates a new claim info from a checkpointed claim info state object.
func newClaimInfoFromState(state *state.ClaimInfoState) *ClaimInfo {
	info := &ClaimInfo{
		ClaimInfoState: *state.DeepCopy(),
		annotations:    make(map[string][]kubecontainer.Annotation),
		prepared:       false,
	}
	for pluginName, devices := range info.CDIDevices {
		annotations, _ := cdi.GenerateAnnotations(info.ClaimUID, info.DriverName, devices)
		info.annotations[pluginName] = append(info.annotations[pluginName], annotations...)
	}
	return info
}

// setCDIDevices adds a set of CDI devices to the claim info.
func (info *ClaimInfo) setCDIDevices(pluginName string, cdiDevices []string) error {
	// NOTE: Passing CDI device names as annotations is a temporary solution
	// It will be removed after all runtimes are updated
	// to get CDI device names from the ContainerConfig.CDIDevices field
	annotations, err := cdi.GenerateAnnotations(info.ClaimUID, info.DriverName, cdiDevices)
	if err != nil {
		return fmt.Errorf("failed to generate container annotations, err: %+v", err)
	}

	if info.CDIDevices == nil {
		info.CDIDevices = make(map[string][]string)
	}

	if info.annotations == nil {
		info.annotations = make(map[string][]kubecontainer.Annotation)
	}

	info.CDIDevices[pluginName] = cdiDevices
	info.annotations[pluginName] = annotations

	return nil
}

// annotationsAsList returns container annotations as a single list.
func (info *ClaimInfo) annotationsAsList() []kubecontainer.Annotation {
	var lst []kubecontainer.Annotation
	for _, v := range info.annotations {
		lst = append(lst, v...)
	}
	return lst
}

// cdiDevicesAsList returns a list of CDIDevices from the provided claim info.
func (info *ClaimInfo) cdiDevicesAsList() []kubecontainer.CDIDevice {
	var cdiDevices []kubecontainer.CDIDevice
	for _, devices := range info.CDIDevices {
		for _, device := range devices {
			cdiDevices = append(cdiDevices, kubecontainer.CDIDevice{Name: device})
		}
	}
	return cdiDevices
}

// addPodReference adds a pod reference to the claim info.
func (info *ClaimInfo) addPodReference(podUID types.UID) {
	info.PodUIDs.Insert(string(podUID))
}

// hasPodReference checks if a pod reference exists in the claim info.
func (info *ClaimInfo) hasPodReference(podUID types.UID) bool {
	return info.PodUIDs.Has(string(podUID))
}

// deletePodReference deletes a pod reference from the claim info.
func (info *ClaimInfo) deletePodReference(podUID types.UID) {
	info.PodUIDs.Delete(string(podUID))
}

// setPrepared marks the claim info as prepared.
func (info *ClaimInfo) setPrepared() {
	info.prepared = true
}

// isPrepared checks if claim info is prepared or not.
func (info *ClaimInfo) isPrepared() bool {
	return info.prepared
}

// newClaimInfoCache creates a new claim info cache object, pre-populated from a checkpoint (if present).
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
		info := newClaimInfoFromState(&entry)
		cache.claimInfo[info.Namespace+"/"+info.ClaimName] = info
	}

	return cache, nil
}

// withLock runs a function while holding the claimInfoCache lock.
func (cache *claimInfoCache) withLock(f func() error) error {
	cache.Lock()
	defer cache.Unlock()
	return f()
}

// withRLock runs a function while holding the claimInfoCache rlock.
func (cache *claimInfoCache) withRLock(f func() error) error {
	cache.RLock()
	defer cache.RUnlock()
	return f()
}

// add adds a new claim info object into the claim info cache.
func (cache *claimInfoCache) add(info *ClaimInfo) *ClaimInfo {
	cache.claimInfo[info.Namespace+"/"+info.ClaimName] = info
	return info
}

// contains checks to see if a specific claim info object is already in the cache.
func (cache *claimInfoCache) contains(claimName, namespace string) bool {
	_, exists := cache.claimInfo[namespace+"/"+claimName]
	return exists
}

// get gets a specific claim info object from the cache.
func (cache *claimInfoCache) get(claimName, namespace string) (*ClaimInfo, bool) {
	info, exists := cache.claimInfo[namespace+"/"+claimName]
	return info, exists
}

// delete deletes a specific claim info object from the cache.
func (cache *claimInfoCache) delete(claimName, namespace string) {
	delete(cache.claimInfo, namespace+"/"+claimName)
}

// hasPodReference checks if there is at least one claim
// that is referenced by the pod with the given UID
// This function is used indirectly by the status manager
// to check if pod can enter termination status
func (cache *claimInfoCache) hasPodReference(UID types.UID) bool {
	for _, claimInfo := range cache.claimInfo {
		if claimInfo.hasPodReference(UID) {
			return true
		}
	}
	return false
}

// syncToCheckpoint syncs the full claim info cache state to a checkpoint.
func (cache *claimInfoCache) syncToCheckpoint() error {
	claimInfoStateList := make(state.ClaimInfoStateList, 0, len(cache.claimInfo))
	for _, infoClaim := range cache.claimInfo {
		claimInfoStateList = append(claimInfoStateList, infoClaim.ClaimInfoState)
	}
	return cache.state.Store(claimInfoStateList)
}
