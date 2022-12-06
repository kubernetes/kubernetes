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
	// podNamespace is only needed for azurefile to fetch secret namespace, no need to be set for other plugins.
	TranslateInTreeInlineVolumeToCSI(volume *v1.Volume, podNamespace string) (*v1.PersistentVolume, error)

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

// replaceTopology overwrites an existing key in NodeAffinity by a new one.
// If there are any newKey already exist in an expression of a term, we will
// not combine the replaced key Values with the existing ones.
// So there might be duplication if there is any newKey expression
// already in the terms.
func replaceTopology(pv *v1.PersistentVolume, oldKey, newKey string) error {
	// Make sure the necessary fields exist
	if pv == nil || pv.Spec.NodeAffinity == nil || pv.Spec.NodeAffinity.Required == nil ||
		pv.Spec.NodeAffinity.Required.NodeSelectorTerms == nil || len(pv.Spec.NodeAffinity.Required.NodeSelectorTerms) == 0 {
		return nil
	}
	for i := range pv.Spec.NodeAffinity.Required.NodeSelectorTerms {
		for j, r := range pv.Spec.NodeAffinity.Required.NodeSelectorTerms[i].MatchExpressions {
			if r.Key == oldKey {
				pv.Spec.NodeAffinity.Required.NodeSelectorTerms[i].MatchExpressions[j].Key = newKey
			}
		}
	}

	return nil
}

// getTopologyValues returns all unique topology values with the given key found in
// the PV NodeAffinity. Sort by alphabetical order.
// This function collapses multiple zones into a list that is ORed. This assumes that
// the plugin does not support a constraint like key in "zone1" AND "zone2"
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
	// remove duplication and sort them in order for better usage
	var re []string
	for k := range values {
		re = append(re, k)
	}
	sort.Strings(re)
	return re
}

// addTopology appends the topology to the given PV to all Terms.
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

// translateTopologyFromInTreeToCSI converts existing zone labels or in-tree topology to CSI topology.
// In-tree topology has precedence over zone labels. When both in-tree topology and zone labels exist
// for a particular CSI topology, in-tree topology will be used.
// This function will remove the Beta version Kubernetes topology label in case the node upgrade to a
// newer version where it does not have any Beta topology label anymore
func translateTopologyFromInTreeToCSI(pv *v1.PersistentVolume, csiTopologyKey string) error {

	zoneLabel, regionLabel := getTopologyLabel(pv)

	// If Zone kubernetes topology exist, replace it to use csiTopologyKey
	zones := getTopologyValues(pv, zoneLabel)
	if len(zones) > 0 {
		replaceTopology(pv, zoneLabel, csiTopologyKey)
	} else {
		// if nothing is in the NodeAffinity, try to fetch the topology from PV labels
		if label, ok := pv.Labels[zoneLabel]; ok {
			zones = strings.Split(label, labelMultiZoneDelimiter)
			if len(zones) > 0 {
				addTopology(pv, csiTopologyKey, zones)
			}
		}
	}

	// if the in-tree PV has beta region label, replace it with GA label to ensure
	// the scheduler is able to schedule it on new nodes with only GA kubernetes label
	// No need to check it for zone label because it has already been replaced if exist
	if regionLabel == v1.LabelFailureDomainBetaRegion {
		regions := getTopologyValues(pv, regionLabel)
		if len(regions) > 0 {
			replaceTopology(pv, regionLabel, v1.LabelTopologyRegion)
		}
	}

	return nil
}

// getTopologyLabel checks if the kubernetes topology label used in this
// PV is GA and return the zone/region label used.
// The version checking follows the following orders
//  1. Check NodeAffinity
//     1.1 Check if zoneGA exists, if yes return GA labels
//     1.2 Check if zoneBeta exists, if yes return Beta labels
//  2. Check PV labels
//     2.1 Check if zoneGA exists, if yes return GA labels
//     2.2 Check if zoneBeta exists, if yes return Beta labels
func getTopologyLabel(pv *v1.PersistentVolume) (zoneLabel string, regionLabel string) {

	if zoneGA := TopologyKeyExist(v1.LabelTopologyZone, pv.Spec.NodeAffinity); zoneGA {
		return v1.LabelTopologyZone, v1.LabelTopologyRegion
	}
	if zoneBeta := TopologyKeyExist(v1.LabelFailureDomainBetaZone, pv.Spec.NodeAffinity); zoneBeta {
		return v1.LabelFailureDomainBetaZone, v1.LabelFailureDomainBetaRegion
	}
	if _, zoneGA := pv.Labels[v1.LabelTopologyZone]; zoneGA {
		return v1.LabelTopologyZone, v1.LabelTopologyRegion
	}
	if _, zoneBeta := pv.Labels[v1.LabelFailureDomainBetaZone]; zoneBeta {
		return v1.LabelFailureDomainBetaZone, v1.LabelFailureDomainBetaRegion
	}
	// No labels or NodeAffinity exist, default to GA version
	return v1.LabelTopologyZone, v1.LabelTopologyRegion
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

type regionParserFn func([]string) (string, error)

// translateTopologyFromCSIToInTree translate a CSI topology to
// Kubernetes topology and add topology labels to it. Note that this function
// will only work for plugin with a single topologyKey that translates to
// Kubernetes zone(and region if regionParser is passed in).
// If a plugin has more than one topologyKey, it will need to be processed
// separately by the plugin.
// If regionParser is nil, no region NodeAffinity will be added. If not nil,
// it'll be passed to regionTopologyHandler, which will add region topology NodeAffinity
// and labels for the given PV. It assumes the Zone NodeAffinity already exists.
// In short this function will,
// 1. Replace all CSI topology to Kubernetes Zone topology label
// 2. Process and generate region topology if a regionParser is passed
// 3. Add Kubernetes Topology labels(zone) if they do not exist
func translateTopologyFromCSIToInTree(pv *v1.PersistentVolume, csiTopologyKey string, regionParser regionParserFn) error {

	zoneLabel, _ := getTopologyLabel(pv)

	// 1. Replace all CSI topology to Kubernetes Zone label
	err := replaceTopology(pv, csiTopologyKey, zoneLabel)
	if err != nil {
		return fmt.Errorf("Failed to replace CSI topology to Kubernetes topology, error: %v", err)
	}

	// 2. Take care of region topology if a regionParser is passed
	if regionParser != nil {
		// let's make less strict on this one. Even if there is an error in the region processing, just ignore it
		err = regionTopologyHandler(pv, regionParser)
		if err != nil {
			return fmt.Errorf("Failed to handle region topology. error: %v", err)
		}
	}

	// 3. Add labels about Kubernetes Topology
	zoneVals := getTopologyValues(pv, zoneLabel)
	if len(zoneVals) > 0 {
		if pv.Labels == nil {
			pv.Labels = make(map[string]string)
		}
		_, zoneOK := pv.Labels[zoneLabel]
		if !zoneOK {
			zoneValStr := strings.Join(zoneVals, labelMultiZoneDelimiter)
			pv.Labels[zoneLabel] = zoneValStr
		}
	}

	return nil
}

// translateAllowedTopologies translates allowed topologies within storage class or PV
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
			} else {
				// Other topologies are passed through unchanged.
				newExp = exp
			}
			newTerm.MatchLabelExpressions = append(newTerm.MatchLabelExpressions, newExp)
		}
		newTopologies = append(newTopologies, newTerm)
	}
	return newTopologies, nil
}

// regionTopologyHandler will process the PV and add region
// kubernetes topology label to its NodeAffinity and labels
// It assumes the Zone NodeAffinity already exists
// Each provider is responsible for providing their own regionParser
func regionTopologyHandler(pv *v1.PersistentVolume, regionParser regionParserFn) error {

	// Make sure the necessary fields exist
	if pv == nil || pv.Spec.NodeAffinity == nil || pv.Spec.NodeAffinity.Required == nil ||
		pv.Spec.NodeAffinity.Required.NodeSelectorTerms == nil || len(pv.Spec.NodeAffinity.Required.NodeSelectorTerms) == 0 {
		return nil
	}

	zoneLabel, regionLabel := getTopologyLabel(pv)

	// process each term
	for index, nodeSelectorTerm := range pv.Spec.NodeAffinity.Required.NodeSelectorTerms {
		// In the first loop, see if regionLabel already exist
		regionExist := false
		var zoneVals []string
		for _, nsRequirement := range nodeSelectorTerm.MatchExpressions {
			if nsRequirement.Key == regionLabel {
				regionExist = true
				break
			} else if nsRequirement.Key == zoneLabel {
				zoneVals = append(zoneVals, nsRequirement.Values...)
			}
		}
		if regionExist {
			// Regionlabel already exist in this term, skip it
			continue
		}
		// If no regionLabel found, generate region label from the zoneLabel we collect from this term
		regionVal, err := regionParser(zoneVals)
		if err != nil {
			return err
		}
		// Add the regionVal to this term
		pv.Spec.NodeAffinity.Required.NodeSelectorTerms[index].MatchExpressions =
			append(pv.Spec.NodeAffinity.Required.NodeSelectorTerms[index].MatchExpressions, v1.NodeSelectorRequirement{
				Key:      regionLabel,
				Operator: v1.NodeSelectorOpIn,
				Values:   []string{regionVal},
			})

	}

	// Add region label
	regionVals := getTopologyValues(pv, regionLabel)
	if len(regionVals) == 1 {
		// We should only have exactly 1 region value
		if pv.Labels == nil {
			pv.Labels = make(map[string]string)
		}
		_, regionOK := pv.Labels[regionLabel]
		if !regionOK {
			pv.Labels[regionLabel] = regionVals[0]
		}
	}

	return nil
}
