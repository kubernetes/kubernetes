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

package volumeclaimbinder

import (
	"fmt"
	"sort"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client/cache"
)

const (
	// A PV created specifically for one claim must contain this annotation in order to bind to the claim.
	// The value must be the namespace and name of the claim being bound to (i.e, claim.Namespace/claim.Name)
	// This is an experimental feature and likely to change in the future.
	createdForKey = "volume.experimental.kubernetes.io/provisioned-for"
)

// persistentVolumeOrderedIndex is a cache.Store that keeps persistent volumes indexed by AccessModes and ordered by storage capacity.
type persistentVolumeOrderedIndex struct {
	cache.Indexer
}

var _ cache.Store = &persistentVolumeOrderedIndex{} // persistentVolumeOrderedIndex is a Store

func NewPersistentVolumeOrderedIndex() *persistentVolumeOrderedIndex {
	return &persistentVolumeOrderedIndex{
		cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"accessmodes": accessModesIndexFunc}),
	}
}

// accessModesIndexFunc is an indexing function that returns a persistent volume's AccessModes as a string
func accessModesIndexFunc(obj interface{}) ([]string, error) {
	if pv, ok := obj.(*api.PersistentVolume); ok {
		modes := api.GetAccessModesAsString(pv.Spec.AccessModes)
		return []string{modes}, nil
	}
	return []string{""}, fmt.Errorf("object is not a persistent volume: %v", obj)
}

// ListByAccessModes returns all volumes with the given set of AccessModeTypes *in order* of their storage capacity (low to high)
func (pvIndex *persistentVolumeOrderedIndex) ListByAccessModes(modes []api.PersistentVolumeAccessMode) ([]*api.PersistentVolume, error) {
	pv := &api.PersistentVolume{
		Spec: api.PersistentVolumeSpec{
			AccessModes: modes,
		},
	}

	objs, err := pvIndex.Index("accessmodes", pv)
	if err != nil {
		return nil, err
	}

	volumes := make([]*api.PersistentVolume, len(objs))
	for i, obj := range objs {
		volumes[i] = obj.(*api.PersistentVolume)
	}

	sort.Sort(byCapacity{volumes})
	return volumes, nil
}

// matchPredicate is a function that indicates that a persistent volume matches another
type matchPredicate func(compareThis, toThis *api.PersistentVolume) bool

// Find returns the nearest PV from the ordered list or nil if a match is not found
func (pvIndex *persistentVolumeOrderedIndex) Find(searchPV *api.PersistentVolume, matchPredicate matchPredicate) (*api.PersistentVolume, error) {
	// the 'searchPV' argument is a synthetic PV with capacity and accessmodes set according to the user's PersistentVolumeClaim.
	// the synthetic pv arg is, therefore, a request for a storage resource.
	//
	// PVs are indexed by their access modes to allow easier searching.  Each index is the string representation of a set of access modes.
	// There is a finite number of possible sets and PVs will only be indexed in one of them (whichever index matches the PV's modes).
	//
	// A request for resources will always specify its desired access modes.  Any matching PV must have at least that number
	// of access modes, but it can have more.  For example, a user asks for ReadWriteOnce but a GCEPD is available, which is ReadWriteOnce+ReadOnlyMany.
	//
	// Searches are performed against a set of access modes, so we can attempt not only the exact matching modes but also
	// potential matches (the GCEPD example above).
	allPossibleModes := pvIndex.allPossibleMatchingAccessModes(searchPV.Spec.AccessModes)

	for _, modes := range allPossibleModes {
		volumes, err := pvIndex.ListByAccessModes(modes)
		if err != nil {
			return nil, err
		}

		// volumes are sorted by size but some may be bound or earmarked for a specific claim.
		// filter those volumes for easy binary search by size
		// return the exact pre-binding match, if found
		unboundVolumes := []*api.PersistentVolume{}
		for _, volume := range volumes {
			// check for current binding
			if volume.Spec.ClaimRef != nil {
				continue
			}

			// check for pre-bind where the volume is intended for one specific claim
			if createdFor, ok := volume.Annotations[createdForKey]; ok {
				if createdFor != searchPV.Annotations[createdForKey] {
					// the volume is pre-bound and does not match the search criteria.
					continue
				}
				// exact annotation match! No search required.
				return volume, nil
			}

			// volume isn't currently bound or pre-bound.
			unboundVolumes = append(unboundVolumes, volume)
		}

		i := sort.Search(len(unboundVolumes), func(i int) bool { return matchPredicate(searchPV, unboundVolumes[i]) })
		if i < len(unboundVolumes) {
			return unboundVolumes[i], nil
		}
	}
	return nil, nil
}

// findByAccessModesAndStorageCapacity is a convenience method that calls Find w/ requisite matchPredicate for storage
func (pvIndex *persistentVolumeOrderedIndex) findByAccessModesAndStorageCapacity(prebindKey string, modes []api.PersistentVolumeAccessMode, qty resource.Quantity) (*api.PersistentVolume, error) {
	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Annotations: map[string]string{
				createdForKey: prebindKey,
			},
		},
		Spec: api.PersistentVolumeSpec{
			AccessModes: modes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): qty,
			},
		},
	}
	return pvIndex.Find(pv, matchStorageCapacity)
}

// FindBestMatchForClaim is a convenience method that finds a volume by the claim's AccessModes and requests for Storage
func (pvIndex *persistentVolumeOrderedIndex) FindBestMatchForClaim(claim *api.PersistentVolumeClaim) (*api.PersistentVolume, error) {
	return pvIndex.findByAccessModesAndStorageCapacity(fmt.Sprintf("%s/%s", claim.Namespace, claim.Name), claim.Spec.AccessModes, claim.Spec.Resources.Requests[api.ResourceName(api.ResourceStorage)])
}

// byCapacity is used to order volumes by ascending storage size
type byCapacity struct {
	volumes []*api.PersistentVolume
}

func (c byCapacity) Less(i, j int) bool {
	return matchStorageCapacity(c.volumes[i], c.volumes[j])
}

func (c byCapacity) Swap(i, j int) {
	c.volumes[i], c.volumes[j] = c.volumes[j], c.volumes[i]
}

func (c byCapacity) Len() int {
	return len(c.volumes)
}

// matchStorageCapacity is a matchPredicate used to sort and find volumes
func matchStorageCapacity(pvA, pvB *api.PersistentVolume) bool {
	aQty := pvA.Spec.Capacity[api.ResourceStorage]
	bQty := pvB.Spec.Capacity[api.ResourceStorage]
	aSize := aQty.Value()
	bSize := bQty.Value()
	return aSize <= bSize
}

// allPossibleMatchingAccessModes returns an array of AccessMode arrays that can satisfy a user's requested modes.
//
// see comments in the Find func above regarding indexing.
//
// allPossibleMatchingAccessModes gets all stringified accessmodes from the index and returns all those that
// contain at least all of the requested mode.
//
// For example, assume the index contains 2 types of PVs where the stringified accessmodes are:
//
// "RWO,ROX" -- some number of GCEPDs
// "RWO,ROX,RWX" -- some number of NFS volumes
//
// A request for RWO could be satisfied by both sets of indexed volumes, so allPossibleMatchingAccessModes returns:
//
// [][]api.PersistentVolumeAccessMode {
//      []api.PersistentVolumeAccessMode {
//			api.ReadWriteOnce, api.ReadOnlyMany,
//		},
//      []api.PersistentVolumeAccessMode {
//			api.ReadWriteOnce, api.ReadOnlyMany, api.ReadWriteMany,
//		},
// }
//
// A request for RWX can be satisfied by only one set of indexed volumes, so the return is:
//
// [][]api.PersistentVolumeAccessMode {
//      []api.PersistentVolumeAccessMode {
//			api.ReadWriteOnce, api.ReadOnlyMany, api.ReadWriteMany,
//		},
// }
//
// This func returns modes with ascending levels of modes to give the user what is closest to what they actually asked for.
//
func (pvIndex *persistentVolumeOrderedIndex) allPossibleMatchingAccessModes(requestedModes []api.PersistentVolumeAccessMode) [][]api.PersistentVolumeAccessMode {
	matchedModes := [][]api.PersistentVolumeAccessMode{}
	keys := pvIndex.Indexer.ListIndexFuncValues("accessmodes")
	for _, key := range keys {
		indexedModes := api.GetAccessModesFromString(key)
		if containedInAll(indexedModes, requestedModes) {
			matchedModes = append(matchedModes, indexedModes)
		}
	}

	// sort by the number of modes in each array with the fewest number of modes coming first.
	// this allows searching for volumes by the minimum number of modes required of the possible matches.
	sort.Sort(byAccessModes{matchedModes})
	return matchedModes
}

func contains(modes []api.PersistentVolumeAccessMode, mode api.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

func containedInAll(indexedModes []api.PersistentVolumeAccessMode, requestedModes []api.PersistentVolumeAccessMode) bool {
	for _, mode := range requestedModes {
		if !contains(indexedModes, mode) {
			return false
		}
	}
	return true
}

// byAccessModes is used to order access modes by size, with the fewest modes first
type byAccessModes struct {
	modes [][]api.PersistentVolumeAccessMode
}

func (c byAccessModes) Less(i, j int) bool {
	return len(c.modes[i]) < len(c.modes[j])
}

func (c byAccessModes) Swap(i, j int) {
	c.modes[i], c.modes[j] = c.modes[j], c.modes[i]
}

func (c byAccessModes) Len() int {
	return len(c.modes)
}
