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
	"errors"
	"fmt"
	"slices"
	"sync"

	resourceapi "k8s.io/api/resource/v1alpha3"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/kubelet/cm/dra/state"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// ClaimInfo holds information required
// to prepare and unprepare a resource claim.
// +k8s:deepcopy-gen=true
type ClaimInfo struct {
	state.ClaimInfoState
	prepared bool
}

// claimInfoCache is a cache of processed resource claims keyed by namespace/claimname.
type claimInfoCache struct {
	sync.RWMutex
	state     state.CheckpointState
	claimInfo map[string]*ClaimInfo
}

// newClaimInfoFromClaim creates a new claim info from a resource claim.
// It verifies that the kubelet can handle the claim.
func newClaimInfoFromClaim(claim *resourceapi.ResourceClaim) (*ClaimInfo, error) {
	claimInfoState := state.ClaimInfoState{
		ClaimUID:    claim.UID,
		ClaimName:   claim.Name,
		Namespace:   claim.Namespace,
		PodUIDs:     sets.New[string](),
		DriverState: make(map[string]state.DriverState),
	}
	if claim.Status.Allocation == nil {
		return nil, errors.New("not allocated")
	}
	for _, result := range claim.Status.Allocation.Devices.Results {
		claimInfoState.DriverState[result.Driver] = state.DriverState{}
	}
	info := &ClaimInfo{
		ClaimInfoState: claimInfoState,
		prepared:       false,
	}
	return info, nil
}

// newClaimInfoFromClaim creates a new claim info from a checkpointed claim info state object.
func newClaimInfoFromState(state *state.ClaimInfoState) *ClaimInfo {
	info := &ClaimInfo{
		ClaimInfoState: *state.DeepCopy(),
		prepared:       false,
	}
	return info
}

// setCDIDevices adds a set of CDI devices to the claim info.
func (info *ClaimInfo) addDevice(driverName string, device state.Device) {
	if info.DriverState == nil {
		info.DriverState = make(map[string]state.DriverState)
	}
	driverState := info.DriverState[driverName]
	driverState.Devices = append(driverState.Devices, device)
	info.DriverState[driverName] = driverState
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

// cdiDevicesAsList returns a list of CDIDevices from the provided claim info.
// When the request name is non-empty, only devices relevant for that request
// are returned.
func (info *ClaimInfo) cdiDevicesAsList(requestName string) []kubecontainer.CDIDevice {
	var cdiDevices []kubecontainer.CDIDevice
	for _, driverData := range info.DriverState {
		for _, device := range driverData.Devices {
			if requestName == "" || len(device.RequestNames) == 0 || slices.Contains(device.RequestNames, requestName) {
				for _, cdiDeviceID := range device.CDIDeviceIDs {
					cdiDevices = append(cdiDevices, kubecontainer.CDIDevice{Name: cdiDeviceID})
				}
			}
		}
	}
	return cdiDevices
}
