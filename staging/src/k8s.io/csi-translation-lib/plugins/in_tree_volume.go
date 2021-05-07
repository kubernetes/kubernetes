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
	"strings"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	cloudvolume "k8s.io/cloud-provider/volume"
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

// getTopologyZones returns all topology zones with the given key found in the PV.
func getTopologyZones(pv *v1.PersistentVolume, key string) []string {
	if pv.Spec.NodeAffinity == nil ||
		pv.Spec.NodeAffinity.Required == nil ||
		len(pv.Spec.NodeAffinity.Required.NodeSelectorTerms) < 1 {
		return nil
	}

	var values []string
	for i := range pv.Spec.NodeAffinity.Required.NodeSelectorTerms {
		for _, r := range pv.Spec.NodeAffinity.Required.NodeSelectorTerms[i].MatchExpressions {
			if r.Key == key {
				values = append(values, r.Values...)
			}
		}
	}
	return values
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
	pv.Spec.NodeAffinity = new(v1.VolumeNodeAffinity)
	pv.Spec.NodeAffinity.Required = new(v1.NodeSelector)
	pv.Spec.NodeAffinity.Required.NodeSelectorTerms = make([]v1.NodeSelectorTerm, 1)

	topology := v1.NodeSelectorRequirement{
		Key:      topologyKey,
		Operator: v1.NodeSelectorOpIn,
		Values:   zones,
	}

	pv.Spec.NodeAffinity.Required.NodeSelectorTerms[0].MatchExpressions = append(
		pv.Spec.NodeAffinity.Required.NodeSelectorTerms[0].MatchExpressions,
		topology,
	)

	return nil
}

// translateTopology converts existing zone labels or in-tree topology to CSI topology.
// In-tree topology has precedence over zone labels.
func translateTopology(pv *v1.PersistentVolume, topologyKey string) error {
	// If topology is already set, assume the content is accurate
	if len(getTopologyZones(pv, topologyKey)) > 0 {
		return nil
	}

	zones := getTopologyZones(pv, v1.LabelZoneFailureDomain)
	if len(zones) > 0 {
		return replaceTopology(pv, v1.LabelZoneFailureDomain, topologyKey)
	}

	if label, ok := pv.Labels[v1.LabelZoneFailureDomain]; ok {
		zones = strings.Split(label, cloudvolume.LabelMultiZoneDelimiter)
		if len(zones) > 0 {
			return addTopology(pv, topologyKey, zones)
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
			if exp.Key == v1.LabelZoneFailureDomain {
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
