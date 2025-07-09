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
	"strings"
	"sync"

	"github.com/go-logr/logr"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/component-base/metrics"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/cm/dra/state"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
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
	logger klog.Logger

	sync.RWMutex
	checkpointer state.Checkpointer
	claimInfo    map[string]*ClaimInfo
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
func (info *ClaimInfo) addDevice(driverName string, deviceState state.Device) {
	if info.DriverState == nil {
		info.DriverState = make(map[string]state.DriverState)
	}
	driverState := info.DriverState[driverName]
	driverState.Devices = append(driverState.Devices, deviceState)
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

// newClaimInfoCache creates a new claim info cache object, pre-populated from a checkpoint (if present).
func newClaimInfoCache(logger klog.Logger, stateDir, checkpointName string) (*claimInfoCache, error) {
	checkpointer, err := state.NewCheckpointer(stateDir, checkpointName)
	if err != nil {
		return nil, fmt.Errorf("could not initialize checkpoint manager, please drain node and remove DRA state file, err: %w", err)
	}

	checkpoint, err := checkpointer.GetOrCreate()
	if err != nil {
		return nil, fmt.Errorf("GetOrCreate() on checkpoint state: %w", err)
	}

	cache := &claimInfoCache{
		logger:       logger,
		checkpointer: checkpointer,
		claimInfo:    make(map[string]*ClaimInfo),
	}

	entries, err := checkpoint.GetClaimInfoStateList()
	if err != nil {
		return nil, fmt.Errorf("GetEntries() on checkpoint: %w", err)

	}
	for _, entry := range entries {
		info := newClaimInfoFromState(&entry)
		cache.claimInfo[info.Namespace+"/"+info.ClaimName] = info
	}

	return cache, nil
}

// withLock runs a function while holding the claimInfoCache lock.
// It logs changes.
func (cache *claimInfoCache) withLock(f func() error) error {
	cache.Lock()
	defer cache.Unlock()

	if loggerV := cache.logger.V(5); loggerV.Enabled() {
		claimsInUseBefore := cache.claimsInUse()
		defer func() {
			claimsInUseAfter := cache.claimsInUse()
			delta := claimsInUseDelta(claimsInUseBefore, claimsInUseAfter)

			changed := false
			for _, inUse := range delta {
				if inUse.Delta != 0 {
					changed = true
					break
				}
			}

			if changed {
				cache.logger.V(5).Info("ResourceClaim usage changed", "claimsInUse", delta)
			}
		}()
	}

	return f()
}

// withRLock runs a function while holding the claimInfoCache rlock.
func (cache *claimInfoCache) withRLock(f func() error) error {
	cache.RLock()
	defer cache.RUnlock()
	return f()
}

// add adds a new claim info object into the claim info cache.
func (cache *claimInfoCache) add(info *ClaimInfo) {
	cache.claimInfo[info.Namespace+"/"+info.ClaimName] = info
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
func (cache *claimInfoCache) hasPodReference(uid types.UID) bool {
	for _, claimInfo := range cache.claimInfo {
		if claimInfo.hasPodReference(uid) {
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
	checkpoint, err := state.NewCheckpoint(claimInfoStateList)
	if err != nil {
		return err
	}
	return cache.checkpointer.Store(checkpoint)
}

// claimsInUse computes the the current counter vector for DRAResourceClaimsInUse.
// It returns a map of driver name to number of claims which have been prepared using
// the driver. The [kubeletmetrics.DRAResourceClaimsInUseAnyDriver] key stands for
// all prepared claims.
//
// Must be called while the rlock is held.
func (cache *claimInfoCache) claimsInUse() map[string]int {
	counts := make(map[string]int)
	total := 0
	for _, claimInfo := range cache.claimInfo {
		if !claimInfo.isPrepared() {
			continue
		}
		total++

		for driverName := range claimInfo.DriverState {
			counts[driverName]++
		}
	}
	counts[kubeletmetrics.DRAResourceClaimsInUseAnyDriver] = total
	return counts
}

// claimsInUseDelta compares two maps returned by claimsInUse.
// The type can be used as value in structured logging.
func claimsInUseDelta(before, after map[string]int) ClaimsInUseDelta {
	var delta ClaimsInUseDelta
	for driverName, count := range before {
		if _, stillSet := after[driverName]; !stillSet {
			delta = append(delta, ClaimsInUse{DriverName: driverName, Count: 0, Delta: -count})
		}
	}
	for driverName, count := range after {
		delta = append(delta, ClaimsInUse{DriverName: driverName, Count: count, Delta: count - before[driverName]})
	}
	return delta
}

// ClaimsInUseDelta provides String (for text logging) and MarshalLog (for structured logging).
type ClaimsInUseDelta []ClaimsInUse

var _ fmt.Stringer = ClaimsInUseDelta{}
var _ logr.Marshaler = ClaimsInUseDelta{}

func (d ClaimsInUseDelta) String() string {
	d = d.sort()
	var buffer strings.Builder
	for i, inUse := range d {
		if i > 0 {
			buffer.WriteByte('\n')
		}
		buffer.WriteString(fmt.Sprintf("%s: %d (%+d)", inUse.DriverName, inUse.Count, inUse.Delta))
	}
	return buffer.String()
}

func (d ClaimsInUseDelta) MarshalLog() any {
	d = d.sort()
	return []ClaimsInUse(d)
}

// sort returns a sorted copy of the slice.
func (d ClaimsInUseDelta) sort() ClaimsInUseDelta {
	d = slices.Clone(d)
	slices.SortFunc(d, func(a, b ClaimsInUse) int {
		return strings.Compare(a.DriverName, b.DriverName)
	})
	return d
}

type ClaimsInUse struct {
	DriverName string
	Count      int
	Delta      int
}

// claimInfoCollector provides metrics for a claimInfoCache.
type claimInfoCollector struct {
	metrics.BaseStableCollector
	cache *claimInfoCache
}

var _ metrics.StableCollector = &claimInfoCollector{}

// DescribeWithStability implements the metrics.StableCollector interface.
func (collector *claimInfoCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- kubeletmetrics.DRAResourceClaimsInUseDesc
}

// CollectWithStability implements the metrics.StableCollector interface.
func (collector *claimInfoCollector) CollectWithStability(ch chan<- metrics.Metric) {
	var claimsInUse map[string]int
	_ = collector.cache.withRLock(func() error {
		claimsInUse = collector.cache.claimsInUse()
		return nil
	})

	// Only currently known drivers are listed. If a driver had active
	// claims in the past, no longer does and then gets uninstalled, it no
	// longer shows up. This avoids the memory leak problem in a normal
	// GaugeVec which could grow over time unless obsolete drivers are
	// actively deleted.
	//
	// The empty driver name provides the overall count of all active
	// ResourceClaims regardless of the driver.
	for driverName, count := range claimsInUse {
		ch <- metrics.NewLazyConstMetric(kubeletmetrics.DRAResourceClaimsInUseDesc, metrics.GaugeValue, float64(count), driverName)
	}
}
