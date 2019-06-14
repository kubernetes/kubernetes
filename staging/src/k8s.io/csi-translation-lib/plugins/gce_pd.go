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
	"strconv"
	"strings"

	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	cloudvolume "k8s.io/cloud-provider/volume"
)

const (
	// GCEPDDriverName is the name of the CSI driver for GCE PD
	GCEPDDriverName = "pd.csi.storage.gke.io"
	// GCEPDInTreePluginName is the name of the intree plugin for GCE PD
	GCEPDInTreePluginName = "kubernetes.io/gce-pd"

	// GCEPDTopologyKey is the zonal topology key for GCE PD CSI Driver
	GCEPDTopologyKey = "topology.gke.io/zone"

	// Volume ID Expected Format
	// "projects/{projectName}/zones/{zoneName}/disks/{diskName}"
	volIDZonalFmt = "projects/%s/zones/%s/disks/%s"
	// "projects/{projectName}/regions/{regionName}/disks/{diskName}"
	volIDRegionalFmt   = "projects/%s/regions/%s/disks/%s"
	volIDDiskNameValue = 5
	volIDTotalElements = 6

	// UnspecifiedValue is used for an unknown zone string
	UnspecifiedValue = "UNSPECIFIED"
)

var _ InTreePlugin = &gcePersistentDiskCSITranslator{}

// gcePersistentDiskCSITranslator handles translation of PV spec from In-tree
// GCE PD to CSI GCE PD and vice versa
type gcePersistentDiskCSITranslator struct{}

// NewGCEPersistentDiskCSITranslator returns a new instance of gcePersistentDiskTranslator
func NewGCEPersistentDiskCSITranslator() InTreePlugin {
	return &gcePersistentDiskCSITranslator{}
}

func translateAllowedTopologies(terms []v1.TopologySelectorTerm) ([]v1.TopologySelectorTerm, error) {
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
					Key:    GCEPDTopologyKey,
					Values: exp.Values,
				}
			} else if exp.Key == GCEPDTopologyKey {
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

func generateToplogySelectors(key string, values []string) []v1.TopologySelectorTerm {
	return []v1.TopologySelectorTerm{
		{
			MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
				{
					Key:    key,
					Values: values,
				},
			},
		},
	}
}

// TranslateInTreeStorageClassParametersToCSI translates InTree GCE storage class parameters to CSI storage class
func (g *gcePersistentDiskCSITranslator) TranslateInTreeStorageClassToCSI(sc *storage.StorageClass) (*storage.StorageClass, error) {
	var generatedTopologies []v1.TopologySelectorTerm

	np := map[string]string{}
	for k, v := range sc.Parameters {
		switch strings.ToLower(k) {
		case "fstype":
			// prefixed fstype parameter is stripped out by external provisioner
			np["csi.storage.k8s.io/fstype"] = v
		// Strip out zone and zones parameters and translate them into topologies instead
		case "zone":
			generatedTopologies = generateToplogySelectors(GCEPDTopologyKey, []string{v})
		case "zones":
			generatedTopologies = generateToplogySelectors(GCEPDTopologyKey, strings.Split(v, ","))
		default:
			np[k] = v
		}
	}

	if len(generatedTopologies) > 0 && len(sc.AllowedTopologies) > 0 {
		return nil, fmt.Errorf("cannot simultaneously set allowed topologies and zone/zones parameters")
	} else if len(generatedTopologies) > 0 {
		sc.AllowedTopologies = generatedTopologies
	} else if len(sc.AllowedTopologies) > 0 {
		newTopologies, err := translateAllowedTopologies(sc.AllowedTopologies)
		if err != nil {
			return nil, fmt.Errorf("failed translating allowed topologies: %v", err)
		}
		sc.AllowedTopologies = newTopologies
	}

	sc.Parameters = np

	return sc, nil
}

// backwardCompatibleAccessModes translates all instances of ReadWriteMany
// access mode from the in-tree plugin to ReadWriteOnce. This is because in-tree
// plugin never supported ReadWriteMany but also did not validate or enforce
// this access mode for pre-provisioned volumes. The GCE PD CSI Driver validates
// and enforces (fails) ReadWriteMany. Therefore we treat all in-tree
// ReadWriteMany as ReadWriteOnce volumes to not break legacy volumes.
func backwardCompatibleAccessModes(ams []v1.PersistentVolumeAccessMode) []v1.PersistentVolumeAccessMode {
	if ams == nil {
		return nil
	}
	newAM := []v1.PersistentVolumeAccessMode{}
	for _, am := range ams {
		if am == v1.ReadWriteMany {
			newAM = append(newAM, v1.ReadWriteOnce)
		} else {
			newAM = append(newAM, am)
		}
	}
	return newAM
}

// TranslateInTreeInlineVolumeToCSI takes a Volume with GCEPersistentDisk set from in-tree
// and converts the GCEPersistentDisk source to a CSIPersistentVolumeSource
func (g *gcePersistentDiskCSITranslator) TranslateInTreeInlineVolumeToCSI(volume *v1.Volume) (*v1.PersistentVolume, error) {
	if volume == nil || volume.GCEPersistentDisk == nil {
		return nil, fmt.Errorf("volume is nil or GCE PD not defined on volume")
	}

	pdSource := volume.GCEPersistentDisk

	partition := ""
	if pdSource.Partition != 0 {
		partition = strconv.Itoa(int(pdSource.Partition))
	}

	pv := &v1.PersistentVolume{
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				CSI: &v1.CSIPersistentVolumeSource{
					Driver:       GCEPDDriverName,
					VolumeHandle: fmt.Sprintf(volIDZonalFmt, UnspecifiedValue, UnspecifiedValue, pdSource.PDName),
					ReadOnly:     pdSource.ReadOnly,
					FSType:       pdSource.FSType,
					VolumeAttributes: map[string]string{
						"partition": partition,
					},
				},
			},
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
		},
	}
	return pv, nil
}

// TranslateInTreePVToCSI takes a PV with GCEPersistentDisk set from in-tree
// and converts the GCEPersistentDisk source to a CSIPersistentVolumeSource
func (g *gcePersistentDiskCSITranslator) TranslateInTreePVToCSI(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	var volID string

	if pv == nil || pv.Spec.GCEPersistentDisk == nil {
		return nil, fmt.Errorf("pv is nil or GCE Persistent Disk source not defined on pv")
	}

	zonesLabel := pv.Labels[v1.LabelZoneFailureDomain]
	zones := strings.Split(zonesLabel, cloudvolume.LabelMultiZoneDelimiter)
	if len(zones) == 1 && len(zones[0]) != 0 {
		// Zonal
		volID = fmt.Sprintf(volIDZonalFmt, UnspecifiedValue, zones[0], pv.Spec.GCEPersistentDisk.PDName)
	} else if len(zones) > 1 {
		// Regional
		region, err := getRegionFromZones(zones)
		if err != nil {
			return nil, fmt.Errorf("failed to get region from zones: %v", err)
		}
		volID = fmt.Sprintf(volIDZonalFmt, UnspecifiedValue, region, pv.Spec.GCEPersistentDisk.PDName)
	} else {
		// Unspecified
		volID = fmt.Sprintf(volIDZonalFmt, UnspecifiedValue, UnspecifiedValue, pv.Spec.GCEPersistentDisk.PDName)
	}

	gceSource := pv.Spec.PersistentVolumeSource.GCEPersistentDisk

	partition := ""
	if gceSource.Partition != 0 {
		partition = strconv.Itoa(int(gceSource.Partition))
	}

	csiSource := &v1.CSIPersistentVolumeSource{
		Driver:       GCEPDDriverName,
		VolumeHandle: volID,
		ReadOnly:     gceSource.ReadOnly,
		FSType:       gceSource.FSType,
		VolumeAttributes: map[string]string{
			"partition": partition,
		},
	}

	pv.Spec.PersistentVolumeSource.GCEPersistentDisk = nil
	pv.Spec.PersistentVolumeSource.CSI = csiSource
	pv.Spec.AccessModes = backwardCompatibleAccessModes(pv.Spec.AccessModes)

	return pv, nil
}

// TranslateCSIPVToInTree takes a PV with CSIPersistentVolumeSource set and
// translates the GCE PD CSI source to a GCEPersistentDisk source.
func (g *gcePersistentDiskCSITranslator) TranslateCSIPVToInTree(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.CSI == nil {
		return nil, fmt.Errorf("pv is nil or CSI source not defined on pv")
	}
	csiSource := pv.Spec.CSI

	pdName, err := pdNameFromVolumeID(csiSource.VolumeHandle)
	if err != nil {
		return nil, err
	}

	gceSource := &v1.GCEPersistentDiskVolumeSource{
		PDName:   pdName,
		FSType:   csiSource.FSType,
		ReadOnly: csiSource.ReadOnly,
	}
	if partition, ok := csiSource.VolumeAttributes["partition"]; ok && partition != "" {
		partInt, err := strconv.Atoi(partition)
		if err != nil {
			return nil, fmt.Errorf("Failed to convert partition %v to integer: %v", partition, err)
		}
		gceSource.Partition = int32(partInt)
	}

	// TODO: Take the zone/regional information and stick it into the label.

	pv.Spec.CSI = nil
	pv.Spec.GCEPersistentDisk = gceSource

	return pv, nil
}

// CanSupport tests whether the plugin supports a given persistent volume
// specification from the API.  The spec pointer should be considered
// const.
func (g *gcePersistentDiskCSITranslator) CanSupport(pv *v1.PersistentVolume) bool {
	return pv != nil && pv.Spec.GCEPersistentDisk != nil
}

// CanSupportInline tests whether the plugin supports a given inline volume
// specification from the API.  The spec pointer should be considered
// const.
func (g *gcePersistentDiskCSITranslator) CanSupportInline(volume *v1.Volume) bool {
	return volume != nil && volume.GCEPersistentDisk != nil
}

// GetInTreePluginName returns the name of the intree plugin driver
func (g *gcePersistentDiskCSITranslator) GetInTreePluginName() string {
	return GCEPDInTreePluginName
}

// GetCSIPluginName returns the name of the CSI plugin
func (g *gcePersistentDiskCSITranslator) GetCSIPluginName() string {
	return GCEPDDriverName
}

func pdNameFromVolumeID(id string) (string, error) {
	splitID := strings.Split(id, "/")
	if len(splitID) != volIDTotalElements {
		return "", fmt.Errorf("failed to get id components. Expected projects/{project}/zones/{zone}/disks/{name}. Got: %s", id)
	}
	return splitID[volIDDiskNameValue], nil
}

// TODO: Replace this with the imported one from GCE PD CSI Driver when
// the driver removes all k8s/k8s dependencies
func getRegionFromZones(zones []string) (string, error) {
	regions := sets.String{}
	if len(zones) < 1 {
		return "", fmt.Errorf("no zones specified")
	}
	for _, zone := range zones {
		// Zone expected format {locale}-{region}-{zone}
		splitZone := strings.Split(zone, "-")
		if len(splitZone) != 3 {
			return "", fmt.Errorf("zone in unexpected format, expected: {locale}-{region}-{zone}, got: %v", zone)
		}
		regions.Insert(strings.Join(splitZone[0:2], "-"))
	}
	if regions.Len() != 1 {
		return "", fmt.Errorf("multiple or no regions gotten from zones, got: %v", regions)
	}
	return regions.UnsortedList()[0], nil
}
