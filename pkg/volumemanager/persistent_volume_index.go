/*
Copyright 2014 Google Inc. All rights reserved.

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

package volumemanager

import (
	"sort"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

type PersistentVolumeIndex interface {
	Add(volume *api.PersistentVolume) error
	Match(claim *api.PersistentVolumeClaim) *api.PersistentVolume
	Exists(volume *api.PersistentVolume) bool
}

// generic implementation creates an index of volumes like so:
//
//		RWO:		[]api.PersistentVolume		-- sorted by Size, smallest to largest
//		RWOROXRWO:	[]api.PersistentVolume		-- sorted by Size, smallest to largest
//		RWOROXRWX:	[]api.PersistentVolume		-- sorted by Size, smallest to largest
//
// This allow fast identification of a volume by its capabilities (accessModeType) and then
// to find the closet-without-going-under size request
type genericPersistentVolumeIndex struct {
	cache map[string][]*api.PersistentVolume
}

// TODO make this store keys to objects, not objects.  use the store/reflector.
func NewPersistentVolumeIndex() PersistentVolumeIndex {
	cache := make(map[string][]*api.PersistentVolume)
	return &genericPersistentVolumeIndex{
		cache: cache,
	}
}

// given a set of volumes, match the one that closest fits the claim
func (binder *genericPersistentVolumeIndex) Match(claim *api.PersistentVolumeClaim) *api.PersistentVolume {

	desiredModes := GetAccessModesAsString(claim.Spec.AccessModes)

	// this is better handled by AccessModeType constant validation from YAML/JSON
	// the YAML constants could be incorrect and we arrive here without correct desired modes to mount a volume
	if desiredModes == "" {
		return nil
	}

	quantity := claim.Spec.Resources.Requests[api.ResourceStorage]
	desiredSize := quantity.Value()
	volumes := binder.cache[desiredModes]

	for _, v := range volumes {
		qty := v.Spec.Capacity[api.ResourceStorage]
		if qty.Value() >= desiredSize && v.ClaimRef == nil {
			return v
		}
	}

	return nil

}

// TODO make this index a set of volumes.  see util.StringSet
func (binder *genericPersistentVolumeIndex) Add(volume *api.PersistentVolume) error {

	modes := GetAccessModeType(volume.Spec.Source)
	modesStr := GetAccessModesAsString(modes)

	if _, ok := binder.cache[modesStr]; !ok {
		binder.cache[modesStr] = []*api.PersistentVolume{}
	}

	if !binder.Exists(volume) {
		binder.cache[string(volume.ObjectMeta.UID)] = append([]*api.PersistentVolume{}, volume)
		binder.cache[modesStr] = append(binder.cache[modesStr], volume)
	}

	sort.Sort(PersistentVolumeComparator(binder.cache[modesStr]))

	return nil
}

func (binder *genericPersistentVolumeIndex) Exists(volume *api.PersistentVolume) bool {
	if _, ok := binder.cache[string(volume.UID)]; !ok {
		return false
	}
	return true
}

func GetAccessModesAsString(modes []api.AccessModeType) string {

	modesAsString := ""

	if contains(modes, api.ReadWriteOnce) {
		modesAsString += "RWO"
	}
	if contains(modes, api.ReadOnlyMany) {
		modesAsString += "ROX"
	}
	if contains(modes, api.ReadWriteMany) {
		modesAsString += "RWX"
	}

	return modesAsString
}

func contains(modes []api.AccessModeType, mode api.AccessModeType) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

// would this be better on api.VolumeSource?
func GetAccessModeType(source api.VolumeSource) []api.AccessModeType {

	if source.AWSElasticBlockStore != nil || source.HostPath != nil {
		return []api.AccessModeType{api.ReadWriteOnce}
	}

	if source.GCEPersistentDisk != nil {
		return []api.AccessModeType{
			api.ReadWriteOnce,
			api.ReadOnlyMany,
		}
	}

	if source.NFSMount != nil {
		return []api.AccessModeType{
			api.ReadWriteOnce,
			api.ReadOnlyMany,
			api.ReadWriteMany,
		}
	}

	return []api.AccessModeType{}
}

type PersistentVolumeComparator []*api.PersistentVolume

func (comp PersistentVolumeComparator) Len() int      { return len(comp) }
func (comp PersistentVolumeComparator) Swap(i, j int) { comp[i], comp[j] = comp[j], comp[i] }
func (comp PersistentVolumeComparator) Less(i, j int) bool {
	aQty := comp[i].Spec.Capacity[api.ResourceStorage]
	bQty := comp[j].Spec.Capacity[api.ResourceStorage]
	aSize := aQty.Value()
	bSize := bQty.Value()
	return aSize < bSize
}
