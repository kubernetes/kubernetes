/*
Copyright 2018 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/util/sets"
)

const (
	// GCE PD CSI driver constants
	GCEPDDriverName       = "com.google.csi.gcepd"
	GCEPDInTreePluginName = "kubernetes.io/gce-pd"

	UnspecifiedValue = "UNSPECIFIED"
	// Volume ID Expected Format
	// "projects/{projectName}/zones/{zoneName}/disks/{diskName}"
	volIDZonalFmt = "projects/%s/zones/%s/disks/%s"
	// "projects/{projectName}/regions/{regionName}/disks/{diskName}"
	volIDRegionalFmt   = "projects/%s/regions/%s/disks/%s"
	volIDDiskNameValue = 5
	volIDTotalElements = 6

	// Kubernetes label constants
	LabelZoneFailureDomain  = "failure-domain.beta.kubernetes.io/zone"
	LabelMultiZoneDelimiter = "__"
)

type GCEPD struct{}

// TranslateToCSI takes a volume.Spec and will translate it to a
// CSIPersistentVolumeSource if the translation logic for that
// specific in-tree volume spec has been implemented
func (g *GCEPD) TranslateInTreePVToCSI(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	var volID string

	if pv == nil || pv.Spec.GCEPersistentDisk == nil {
		return nil, fmt.Errorf("GCE Persistent Disk source not defined on pv")
	}

	zonesLabel := pv.Labels[LabelZoneFailureDomain]
	zones := strings.Split(zonesLabel, LabelMultiZoneDelimiter)
	if len(zones) == 1 {
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
	csiSource := &v1.CSIPersistentVolumeSource{
		Driver:       GCEPDDriverName,
		VolumeHandle: volID,
		ReadOnly:     gceSource.ReadOnly,
		FSType:       gceSource.FSType,
		VolumeAttributes: map[string]string{
			"partition": strconv.FormatInt(int64(gceSource.Partition), 10),
		},
	}

	pv.Spec.PersistentVolumeSource.GCEPersistentDisk = nil
	pv.Spec.PersistentVolumeSource.CSI = csiSource

	return pv, nil
}

// TranslateToIntree takes a CSIPersistentVolumeSource and will translate
// it to a volume.Spec for the specific in-tree volume specified by
//`inTreePlugin`, if that translation logic has been implemented
func (g *GCEPD) TranslateCSIPVToInTree(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.CSI == nil {
		return nil, fmt.Errorf("CSI source not defined on pv")
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
	if partition, ok := csiSource.VolumeAttributes["partition"]; ok {
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

// CanSupport tests whether the plugin supports a given volume
// specification from the API.  The spec pointer should be considered
// const.
func (g *GCEPD) CanSupport(pv *v1.PersistentVolume) bool {
	return pv != nil && pv.Spec.GCEPersistentDisk != nil
}

func (g *GCEPD) GetInTreePluginName() string {
	return GCEPDInTreePluginName
}

func pdNameFromVolumeID(id string) (string, error) {
	splitId := strings.Split(id, "/")
	if len(splitId) != volIDTotalElements {
		return "", fmt.Errorf("failed to get id components. Expected projects/{project}/zones/{zone}/disks/{name}. Got: %s", id)
	}
	return splitId[volIDDiskNameValue], nil
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
