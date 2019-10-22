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

// getTopology returns the current topology zones with the given key.
func getTopology(pv *v1.PersistentVolume, key string) []string {
	if pv.Spec.NodeAffinity == nil ||
		pv.Spec.NodeAffinity.Required == nil ||
		len(pv.Spec.NodeAffinity.Required.NodeSelectorTerms) < 1 {
		return nil
	}
	for i := range pv.Spec.NodeAffinity.Required.NodeSelectorTerms {
		for _, r := range pv.Spec.NodeAffinity.Required.NodeSelectorTerms[i].MatchExpressions {
			if r.Key == key {
				return r.Values
			}
		}
	}
	return nil
}

// recordTopology writes the topology to the given PV.
// If topology already exists with the given key, assume the content is already accurate.
func recordTopology(pv *v1.PersistentVolume, key string, zones []string) error {
	// Make sure there are no duplicate or empty strings
	filteredZones := sets.String{}
	for i := range zones {
		zone := strings.TrimSpace(zones[i])
		if len(zone) > 0 {
			filteredZones.Insert(zone)
		}
	}

	if filteredZones.Len() < 1 {
		return fmt.Errorf("there are no valid zones for topology")
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

	// Don't overwrite if topology is already set
	if len(getTopology(pv, key)) > 0 {
		return nil
	}

	pv.Spec.NodeAffinity.Required.NodeSelectorTerms[0].MatchExpressions = append(pv.Spec.NodeAffinity.Required.NodeSelectorTerms[0].MatchExpressions,
		v1.NodeSelectorRequirement{
			Key:      key,
			Operator: v1.NodeSelectorOpIn,
			Values:   filteredZones.List(),
		})

	return nil
}

// translateTopology converts existing zone labels or in-tree topology to CSI topology.
// In-tree topology has precedence over zone labels.
func translateTopology(pv *v1.PersistentVolume, topologyKey string) error {
	zones := getTopology(pv, v1.LabelZoneFailureDomain)
	if len(zones) > 0 {
		return recordTopology(pv, topologyKey, zones)
	}

	if label, ok := pv.Labels[v1.LabelZoneFailureDomain]; ok {
		zones = strings.Split(label, cloudvolume.LabelMultiZoneDelimiter)
		if len(zones) > 0 {
			return recordTopology(pv, topologyKey, zones)
		}
	}

	return nil
}
