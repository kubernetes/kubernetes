/*
Copyright 2017 The Kubernetes Authors.

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

package azure_dd

import (
	"errors"
	"fmt"
	"strconv"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/azure"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

type azureDiskProvisioner struct {
	plugin  *azureDataDiskPlugin
	options volume.VolumeOptions
}

type azureDiskDeleter struct {
	*dataDisk
	spec   *volume.Spec
	plugin *azureDataDiskPlugin
}

var _ volume.Provisioner = &azureDiskProvisioner{}
var _ volume.Deleter = &azureDiskDeleter{}

func (d *azureDiskDeleter) GetPath() string {
	return getPath(d.podUID, d.dataDisk.diskName, d.plugin.host)
}

func (d *azureDiskDeleter) Delete() error {
	volumeSource, _, err := getVolumeSource(d.spec)
	if err != nil {
		return err
	}

	diskController, err := getDiskController(d.plugin.host)
	if err != nil {
		return err
	}

	managed := (*volumeSource.Kind == v1.AzureManagedDisk)

	if managed {
		return diskController.DeleteManagedDisk(volumeSource.DataDiskURI)
	}

	return diskController.DeleteBlobDisk(volumeSource.DataDiskURI)
}

// parseZoned parsed 'zoned' for storage class. If zoned is not specified (empty string),
// then it defaults to true for managed disks.
func parseZoned(zonedString string, kind v1.AzureDataDiskKind) (bool, error) {
	if zonedString == "" {
		return kind == v1.AzureManagedDisk, nil
	}

	zoned, err := strconv.ParseBool(zonedString)
	if err != nil {
		return false, fmt.Errorf("failed to parse 'zoned': %v", err)
	}

	return zoned, nil
}

func (p *azureDiskProvisioner) Provision(selectedNode *v1.Node, allowedTopologies []v1.TopologySelectorTerm) (*v1.PersistentVolume, error) {
	if !util.AccessModesContainedInAll(p.plugin.GetAccessModes(), p.options.PVC.Spec.AccessModes) {
		return nil, fmt.Errorf("invalid AccessModes %v: only AccessModes %v are supported", p.options.PVC.Spec.AccessModes, p.plugin.GetAccessModes())
	}
	supportedModes := p.plugin.GetAccessModes()

	// perform static validation first
	if p.options.PVC.Spec.Selector != nil {
		return nil, fmt.Errorf("azureDisk - claim.Spec.Selector is not supported for dynamic provisioning on Azure disk")
	}

	if len(p.options.PVC.Spec.AccessModes) > 1 {
		return nil, fmt.Errorf("AzureDisk - multiple access modes are not supported on AzureDisk plugin")
	}

	if len(p.options.PVC.Spec.AccessModes) == 1 {
		if p.options.PVC.Spec.AccessModes[0] != supportedModes[0] {
			return nil, fmt.Errorf("AzureDisk - mode %s is not supported by AzureDisk plugin (supported mode is %s)", p.options.PVC.Spec.AccessModes[0], supportedModes)
		}
	}

	var (
		location, account          string
		storageAccountType, fsType string
		cachingMode                v1.AzureDataDiskCachingMode
		strKind                    string
		err                        error
		resourceGroup              string

		zoned             bool
		zonePresent       bool
		zonesPresent      bool
		strZoned          string
		availabilityZone  string
		availabilityZones string
	)
	// maxLength = 79 - (4 for ".vhd") = 75
	name := util.GenerateVolumeName(p.options.ClusterName, p.options.PVName, 75)
	capacity := p.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	requestGiB, err := util.RoundUpToGiBInt(capacity)
	if err != nil {
		return nil, err
	}

	for k, v := range p.options.Parameters {
		switch strings.ToLower(k) {
		case "skuname":
			storageAccountType = v
		case "location":
			location = v
		case "storageaccount":
			account = v
		case "storageaccounttype":
			storageAccountType = v
		case "kind":
			strKind = v
		case "cachingmode":
			cachingMode = v1.AzureDataDiskCachingMode(v)
		case volume.VolumeParameterFSType:
			fsType = strings.ToLower(v)
		case "resourcegroup":
			resourceGroup = v
		case "zone":
			zonePresent = true
			availabilityZone = v
		case "zones":
			zonesPresent = true
			availabilityZones = v
		case "zoned":
			strZoned = v
		default:
			return nil, fmt.Errorf("AzureDisk - invalid option %s in storage class", k)
		}
	}

	// normalize values
	skuName, err := normalizeStorageAccountType(storageAccountType)
	if err != nil {
		return nil, err
	}

	kind, err := normalizeKind(strFirstLetterToUpper(strKind))
	if err != nil {
		return nil, err
	}

	zoned, err = parseZoned(strZoned, kind)
	if err != nil {
		return nil, err
	}

	if !zoned && (zonePresent || zonesPresent) {
		return nil, fmt.Errorf("zone or zones StorageClass parameters must be used together with zoned parameter")
	}

	if zonePresent && zonesPresent {
		return nil, fmt.Errorf("both zone and zones StorageClass parameters must not be used at the same time")
	}

	if cachingMode, err = normalizeCachingMode(cachingMode); err != nil {
		return nil, err
	}

	diskController, err := getDiskController(p.plugin.host)
	if err != nil {
		return nil, err
	}

	if resourceGroup != "" && kind != v1.AzureManagedDisk {
		return nil, errors.New("StorageClass option 'resourceGroup' can be used only for managed disks")
	}

	// create disk
	diskURI := ""
	labels := map[string]string{}
	if kind == v1.AzureManagedDisk {
		tags := make(map[string]string)
		if p.options.CloudTags != nil {
			tags = *(p.options.CloudTags)
		}

		volumeOptions := &azure.ManagedDiskOptions{
			DiskName:           name,
			StorageAccountType: skuName,
			ResourceGroup:      resourceGroup,
			PVCName:            p.options.PVC.Name,
			SizeGB:             requestGiB,
			Tags:               tags,
			Zoned:              zoned,
			ZonePresent:        zonePresent,
			ZonesPresent:       zonesPresent,
			AvailabilityZone:   availabilityZone,
			AvailabilityZones:  availabilityZones,
		}
		diskURI, err = diskController.CreateManagedDisk(volumeOptions)
		if err != nil {
			return nil, err
		}
		labels, err = diskController.GetAzureDiskLabels(diskURI)
		if err != nil {
			return nil, err
		}
	} else {
		if zoned {
			return nil, errors.New("zoned parameter is only supported for managed disks")
		}

		if kind == v1.AzureDedicatedBlobDisk {
			_, diskURI, _, err = diskController.CreateVolume(name, account, storageAccountType, location, requestGiB)
			if err != nil {
				return nil, err
			}
		} else {
			diskURI, err = diskController.CreateBlobDisk(name, skuName, requestGiB)
			if err != nil {
				return nil, err
			}
		}
	}

	var volumeMode *v1.PersistentVolumeMode
	if utilfeature.DefaultFeatureGate.Enabled(features.BlockVolume) {
		volumeMode = p.options.PVC.Spec.VolumeMode
		if volumeMode != nil && *volumeMode == v1.PersistentVolumeBlock {
			// Block volumes should not have any FSType
			fsType = ""
		}
	}

	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:   p.options.PVName,
			Labels: labels,
			Annotations: map[string]string{
				"volumehelper.VolumeDynamicallyCreatedByKey": "azure-disk-dynamic-provisioner",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: p.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   supportedModes,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", requestGiB)),
			},
			VolumeMode: volumeMode,
			PersistentVolumeSource: v1.PersistentVolumeSource{
				AzureDisk: &v1.AzureDiskVolumeSource{
					CachingMode: &cachingMode,
					DiskName:    name,
					DataDiskURI: diskURI,
					Kind:        &kind,
					FSType:      &fsType,
				},
			},
			MountOptions: p.options.MountOptions,
		},
	}

	return pv, nil
}
