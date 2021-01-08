/*
Copyright 2019 The Kubernetes Authors.

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

package plugins

import (
	"errors"
	"fmt"
	"sort"
	"strings"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

// InTreePlugin handles translations between CSI and in-tree sources in a PV
type InTreePlugin interface {

	// TranslateInTreeStorageClassToCSI takes in-tree volume options
	// and translates them to a volume options consumable by CSI plugin
	TranslateInTreeStorageClassToCSI(sc *storage.StorageClass) (*storage.StorageClass, error)

	// TranslateInTreeInlineVolumeToCSI takes a inline volume and will translate
	// the in-tree inline volume source to a CSIPersistentVolumeSource
	// A PV object containing the CSIPersistentVolumeSource in it's spec is returned
	TranslateInTreeInlineVolumeToCSI(volume *v1.Volume) (*v1.PersistentVolume, error)

	// TranslateInTreePVToCSI takes a persistent volume and will translate
	// the in-tree pv source to a CSI Source. The input persistent volume can be modified
	TranslateInTreePVToCSI(pv *v1.PersistentVolume) (*v1.PersistentVolume, error)

	// TranslateCSIPVToInTree takes a PV with a CSI PersistentVolume Source and will translate
	// it to a in-tree Persistent Volume Source for the in-tree volume
	// by the `Driver` field in the CSI Source. The input PV object can be modified
	TranslateCSIPVToInTree(pv *v1.PersistentVolume) (*v1.PersistentVolume, error)

	// CanSupport tests whether the plugin supports a given persistent volume
	// specification from the API.
	CanSupport(pv *v1.PersistentVolume) bool

	// CanSupportInline tests whether the plugin supports a given inline volume
	// specification from the API.
	CanSupportInline(vol *v1.Volume) bool

	// GetInTreePluginName returns the in-tree plugin name this migrates
	GetInTreePluginName() string

	// GetCSIPluginName returns the name of the CSI plugin that supersedes the in-tree plugin
	GetCSIPluginName() string

	// RepairVolumeHandle generates a correct volume handle based on node ID information.
	RepairVolumeHandle(volumeHandle, nodeID string) (string, error)
}

const (
	// fsTypeKey is the deprecated storage class parameter key for fstype
	fsTypeKey = "fstype"
	// csiFsTypeKey is the storage class parameter key for CSI fstype
	csiFsTypeKey = "csi.storage.k8s.io/fstype"
	// zoneKey is the deprecated storage class parameter key for zone
	zoneKey = "zone"
	// zonesKey is the deprecated storage class parameter key for zones
	zonesKey = "zones"
)

// replaceTopology overwrites an existing topology key by a new one.
func replaceTopology(pv *v1.PersistentVolume, oldKey, newKey string) error {
	for i := range pv.Spec.NodeAffinity.Required.NodeSelectorTerms {
		for j, r := range pv.Spec.NodeAffinity.Required.NodeSelectorTerms[i].MatchExpressions {
			if r.Key == oldKey {
				pv.Spec.NodeAffinity.Required.NodeSelectorTerms[i].MatchExpressions[j].Key = newKey
			}
		}
	}
	return nil
}

// getTopologyValues returns all unique topology values with the given key found in the PV.
func getTopologyValues(pv *v1.PersistentVolume, key string) []string {
	if pv.Spec.NodeAffinity == nil ||
		pv.Spec.NodeAffinity.Required == nil ||
		len(pv.Spec.NodeAffinity.Required.NodeSelectorTerms) < 1 {
		return nil
	}

	values := make(map[string]bool)
	for i := range pv.Spec.NodeAffinity.Required.NodeSelectorTerms {
		for _, r := range pv.Spec.NodeAffinity.Required.NodeSelectorTerms[i].MatchExpressions {
			if r.Key == key {
				for _, v := range r.Values {
					values[v] = true
				}
			}
		}
	}
	var re []string
	for k := range values {
		re = append(re, k)
	}
	sort.Strings(re)
	return re
}

// addTopology appends the topology to the given PV.
func addTopology(pv *v1.PersistentVolume, topologyKey string, zones []string) error {
	// Make sure there are no duplicate or empty strings
	filteredZones := sets.String{}
	for i := range zones {
		zone := strings.TrimSpace(zones[i])
		if len(zone) > 0 {
			filteredZones.Insert(zone)
		}
	}

	zones = filteredZones.List()
	if len(zones) < 1 {
		return errors.New("there are no valid zones to add to pv")
	}

	// Make sure the necessary fields exist
	if pv.Spec.NodeAffinity == nil {
		pv.Spec.NodeAffinity = new(v1.VolumeNodeAffinity)
	}

	if pv.Spec.NodeAffinity.Required == nil {
		pv.Spec.NodeAffinity.Required = new(v1.NodeSelector)
	}

	if len(pv.Spec.NodeAffinity.Required.NodeSelectorTerms) == 0 {
		pv.Spec.NodeAffinity.Required.NodeSelectorTerms = make([]v1.NodeSelectorTerm, 1)
	}

	topology := v1.NodeSelectorRequirement{
		Key:      topologyKey,
		Operator: v1.NodeSelectorOpIn,
		Values:   zones,
	}

	// add the CSI topology to each term
	for i := range pv.Spec.NodeAffinity.Required.NodeSelectorTerms {
		pv.Spec.NodeAffinity.Required.NodeSelectorTerms[i].MatchExpressions = append(
			pv.Spec.NodeAffinity.Required.NodeSelectorTerms[i].MatchExpressions,
			topology,
		)
	}

	return nil
}

// removeTopology removes the topology from the given PV. Return false
// if the topology key is not found
func removeTopology(pv *v1.PersistentVolume, topologyKey string) bool {
	// Make sure the necessary fields exist
	if pv == nil || pv.Spec.NodeAffinity == nil || pv.Spec.NodeAffinity.Required == nil ||
		pv.Spec.NodeAffinity.Required.NodeSelectorTerms == nil || len(pv.Spec.NodeAffinity.Required.NodeSelectorTerms) == 0 {
		return false
	}

	found := true
	succeed := false
	for found == true {
		found = false
		var termIndexRemoved []int
		for termIndex, nodeSelectorTerms := range pv.Spec.NodeAffinity.Required.NodeSelectorTerms {
			nsrequirements := nodeSelectorTerms.MatchExpressions

			index := -1
			for i, nodeSelectorRequirement := range nsrequirements {
				if nodeSelectorRequirement.Key == topologyKey {
					index = i
					break
				}
			}
			// We found the key that need to be removed
			if index != -1 {
				nsrequirements[len(nsrequirements)-1], nsrequirements[index] = nsrequirements[index], nsrequirements[len(nsrequirements)-1]
				pv.Spec.NodeAffinity.Required.NodeSelectorTerms[termIndex].MatchExpressions = nsrequirements[:len(nsrequirements)-1]
				if len(nsrequirements)-1 == 0 {
					// No other expression left, the whole term should be removed
					termIndexRemoved = append(termIndexRemoved, termIndex)
				}
				succeed = true
				found = true
			}
		}

		if len(termIndexRemoved) > 0 {
			for i, index := range termIndexRemoved {
				index = index - i
				nodeSelectorTerms := pv.Spec.NodeAffinity.Required.NodeSelectorTerms
				nodeSelectorTerms[len(nodeSelectorTerms)-1], nodeSelectorTerms[index] = nodeSelectorTerms[index], nodeSelectorTerms[len(nodeSelectorTerms)-1]
				pv.Spec.NodeAffinity.Required.NodeSelectorTerms = nodeSelectorTerms[:len(nodeSelectorTerms)-1]
			}
		}
	}

	return succeed
}

// translateTopologyFromInTreeToCSI converts existing zone labels or in-tree topology to CSI topology.
// In-tree topology has precedence over zone labels.
func translateTopologyFromInTreeToCSI(pv *v1.PersistentVolume, topologyKey string) error {
	// If topology is already set, assume the content is accurate
	if len(getTopologyValues(pv, topologyKey)) > 0 {
		return nil
	}

	_, zoneLabel, _ := getTopologyLabel(pv)

	// migrate topology node affinity
	zones := getTopologyValues(pv, zoneLabel)
	if len(zones) > 0 {
		return replaceTopology(pv, zoneLabel, topologyKey)
	}

	// if nothing is in the NodeAffinity, try to fetch the topology from PV labels
	label, ok := pv.Labels[zoneLabel]
	if ok {
		zones = strings.Split(label, labelMultiZoneDelimiter)
		if len(zones) > 0 {
			return addTopology(pv, topologyKey, zones)
		}
	}

	return nil
}

// getTopologyLabel checks if the kubernetes topology used in this PV is GA
// and return the zone/region label used.
// It first check the NodeAffinity to find topology. If nothing is found,
// It checks the PV labels. If it is empty in both places, it will return
// GA label by default
func getTopologyLabel(pv *v1.PersistentVolume) (bool, string, string) {

	// Check the NodeAffinity first for the topology version
	zoneGA := TopologyKeyExist(v1.LabelTopologyZone, pv.Spec.NodeAffinity)
	regionGA := TopologyKeyExist(v1.LabelTopologyRegion, pv.Spec.NodeAffinity)
	if zoneGA || regionGA {
		// GA NodeAffinity exists
		return true, v1.LabelTopologyZone, v1.LabelTopologyRegion
	}

	// If no GA topology in NodeAffinity, check the beta one
	zoneBeta := TopologyKeyExist(v1.LabelFailureDomainBetaZone, pv.Spec.NodeAffinity)
	regionBeta := TopologyKeyExist(v1.LabelFailureDomainBetaRegion, pv.Spec.NodeAffinity)
	if zoneBeta || regionBeta {
		// Beta NodeAffinity exists, GA NodeAffinity not exist
		return false, v1.LabelFailureDomainBetaZone, v1.LabelFailureDomainBetaRegion
	}

	// If nothing is in the NodeAfinity, we should check pv labels
	_, zoneGA = pv.Labels[v1.LabelTopologyZone]
	_, regionGA = pv.Labels[v1.LabelTopologyRegion]
	if zoneGA || regionGA {
		// NodeAffinity not exist, GA label exists
		return true, v1.LabelTopologyZone, v1.LabelTopologyRegion
	}

	// If GA label not exists, check beta version
	_, zoneBeta = pv.Labels[v1.LabelFailureDomainBetaZone]
	_, regionBeta = pv.Labels[v1.LabelFailureDomainBetaRegion]
	if zoneBeta || regionBeta {
		// Beta label exists, NodeAffinity not exist, GA label not exists
		return false, v1.LabelFailureDomainBetaZone, v1.LabelFailureDomainBetaRegion
	}

	// No labels or NodeAffinity exist, default to GA version
	return true, v1.LabelTopologyZone, v1.LabelTopologyRegion
}

// TopologyKeyExist checks if a certain key exists in a VolumeNodeAffinity
func TopologyKeyExist(key string, vna *v1.VolumeNodeAffinity) bool {
	if vna == nil || vna.Required == nil || vna.Required.NodeSelectorTerms == nil || len(vna.Required.NodeSelectorTerms) == 0 {
		return false
	}

	for _, nodeSelectorTerms := range vna.Required.NodeSelectorTerms {
		nsrequirements := nodeSelectorTerms.MatchExpressions
		for _, nodeSelectorRequirement := range nsrequirements {
			if nodeSelectorRequirement.Key == key {
				return true
			}
		}
	}
	return false
}

type regionParser func(pv *v1.PersistentVolume) (string, error)

// translateTopologyFromCSIToInTree translate a CSI PV topology to
// Kubernetes topology and add labels to it. Note that this function
// will only work for plugin with single topologyKey. If a plugin has
// more than one topologyKey, it will need to be processed separately
// by the plugin.
// regionParser is a function to generate region val based on PV
// if the function is not set, we will not set region topology.
// 1. Remove any existing CSI topologyKey from NodeAffinity
// 2. Add Kubernetes Topology in the NodeAffinity if it does not exist
//    2.1 Try to use CSI topology values to recover the Kubernetes topology first
//    2.2 If CSI topology values does not exist, try to use PV label
// 3. Add Kubernetes Topology labels(zone and region)
func translateTopologyFromCSIToInTree(pv *v1.PersistentVolume, topologyKey string, regionParser regionParser) error {

	csiTopologyZoneValues := getTopologyValues(pv, topologyKey)

	// 1. Frist remove the CSI topology Key
	removeTopology(pv, topologyKey)

	_, zoneLabel, regionLabel := getTopologyLabel(pv)
	zoneLabelVal, zoneOK := pv.Labels[zoneLabel]
	regionLabelVal, regionOK := pv.Labels[regionLabel]

	// 2. Add Kubernetes Topology in the NodeAffinity if it does not exist
	// Check if Kubernetes Zone Topology already exist
	topologyZoneValue := getTopologyValues(pv, zoneLabel)
	if len(topologyZoneValue) == 0 {
		// No Kubernetes Topology exist in the current PV, we need to add it
		// 2.1 Let's try to use CSI topology to recover the zone topology first
		if len(csiTopologyZoneValues) > 0 {
			err := addTopology(pv, zoneLabel, csiTopologyZoneValues)
			if err != nil {
				return fmt.Errorf("Failed to add Kubernetes topology zone to PV NodeAffinity. %v", err)
			}
		} else if zoneOK {
			// 2.2 If no CSI topology values exist, try to search PV labels for zone labels
			zones := strings.Split(zoneLabelVal, labelMultiZoneDelimiter)
			if len(zones) > 0 {
				err := addTopology(pv, zoneLabel, zones)
				if err != nil {
					return fmt.Errorf("Failed to add Kubernetes topology zone to PV NodeAffinity. %v", err)
				}
			}
		}
	}

	// Check if Kubernetes Region Topology already exist
	topologyRegionValue := getTopologyValues(pv, regionLabel)
	if len(topologyRegionValue) == 0 {
		// 2.1 Let's try to use CSI topology to recover the region topology first
		if len(csiTopologyZoneValues) > 0 && regionParser != nil {
			regionVal, err := regionParser(pv)
			if err != nil {
				return fmt.Errorf("Failed to parse zones value(%v) to region label %v", csiTopologyZoneValues, err)
			}
			err = addTopology(pv, regionLabel, []string{regionVal})
			if err != nil {
				return fmt.Errorf("Failed to add Kubernetes topology region to PV NodeAffinity. %v", err)
			}
		} else if regionOK {
			// 2.2 If no CSI topology values exist, try to search PV labels for region labels
			err := addTopology(pv, regionLabel, []string{regionLabelVal})
			if err != nil {
				return fmt.Errorf("Failed to add Kubernetes topology region to PV NodeAffinity. %v", err)
			}
		}
	}

	// 3. Add Kubernetes Topology labels, if it already exists, we trust it
	if len(csiTopologyZoneValues) > 0 {
		if pv.Labels == nil {
			pv.Labels = make(map[string]string)
		}
		if !zoneOK {
			csiTopologyZoneValStr := strings.Join(csiTopologyZoneValues, labelMultiZoneDelimiter)
			pv.Labels[zoneLabel] = csiTopologyZoneValStr
		}
		if !regionOK && regionParser != nil {
			regionVal, err := regionParser(pv)
			if err != nil {
				return fmt.Errorf("Failed to parse zones value(%v) to region label %v", csiTopologyZoneValues, err)
			}
			pv.Labels[regionLabel] = regionVal
		}
	}

	return nil
}

// translateAllowedTopologies translates allowed topologies within storage class
// from legacy failure domain to given CSI topology key
func translateAllowedTopologies(terms []v1.TopologySelectorTerm, key string) ([]v1.TopologySelectorTerm, error) {
	if terms == nil {
		return nil, nil
	}

	newTopologies := []v1.TopologySelectorTerm{}
	for _, term := range terms {
		newTerm := v1.TopologySelectorTerm{}
		for _, exp := range term.MatchLabelExpressions {
			var newExp v1.TopologySelectorLabelRequirement
			if exp.Key == v1.LabelFailureDomainBetaZone || exp.Key == v1.LabelTopologyZone {
				newExp = v1.TopologySelectorLabelRequirement{
					Key:    key,
					Values: exp.Values,
				}
			} else if exp.Key == key {
				newExp = exp
			} else {
				return nil, fmt.Errorf("unknown topology key: %v", exp.Key)
			}
			newTerm.MatchLabelExpressions = append(newTerm.MatchLabelExpressions, newExp)
		}
		newTopologies = append(newTopologies, newTerm)
	}
	return newTopologies, nil
}
