/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/volume"
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
	volumeSource, err := getVolumeSource(d.spec)
	if err != nil {
		return err
	}

	wasStandAlone := (*volumeSource.Kind != v1.AzureSharedBlobDisk)
	managed := (*volumeSource.Kind == v1.AzureManagedDisk)

	if managed {
		return d.plugin.managedDiskController.DeleteDataDisk(volumeSource.DataDiskURI)
	}

	return d.plugin.blobDiskController.DeleteDataDisk(volumeSource.DataDiskURI, wasStandAlone)
}

func (p *azureDiskProvisioner) Provision() (*v1.PersistentVolume, error) {
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
			return nil, fmt.Errorf("AzureDisk - mode %s is not supporetd by AzureDisk plugin supported mode is %s", p.options.PVC.Spec.AccessModes[0], supportedModes)
		}
	}

	var (
		storageAccountType, fsType string
		cachingMode                v1.AzureDataDiskCachingMode
		kind                       v1.AzureDataDiskKind
		err                        error
	)
	// maxLength = 79 - (4 for ".vhd") = 75
	name := volume.GenerateVolumeName(p.options.ClusterName, p.options.PVName, 75)
	capacity := p.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	requestBytes := capacity.Value()
	requestGB := int(volume.RoundUpSize(requestBytes, 1024*1024*1024))

	for k, v := range p.options.Parameters {
		switch strings.ToLower(k) {
		case "skuname":
			storageAccountType = strings.ToLower(v)
		case "location":
			return nil, fmt.Errorf("AzureDisk - location parameter is not supported anymore in PVC, use PV to use named storage accounts in different locations")
		case "storageaccount":
			return nil, fmt.Errorf("AzureDisk - storage parameter is not suppoerted anymore in PVC, use PV to use named storage account")
		case "storageaccounttype":
			storageAccountType = strings.ToLower(v)
		case "kind":
			kind = v1.AzureDataDiskKind(v)
		case "cachingmode":
			cachingMode = v1.AzureDataDiskCachingMode(v)
		case "fstype":
			fsType = strings.ToLower(v)
		default:
			return nil, fmt.Errorf("AzureDisk - invalid option %s in PVC", k)
		}
	}

	// normalize  values
	fsType = normalizeFsType(fsType)

	if storageAccountType, err = normalizeStorageAccountType(storageAccountType); err != nil {
		return nil, err
	}

	if kind, err = normalizeKind(kind); err != nil {
		return nil, err
	}

	if cachingMode, err = normalizeCachingMode(cachingMode); err != nil {
		return nil, err
	}

	// create disk
	managed := (kind == v1.AzureManagedDisk)
	forceStandAlone := (kind != v1.AzureSharedBlobDisk)
	diskUri := ""
	if managed {
		diskUri, err = p.plugin.managedDiskController.CreateDataDisk(name, storageAccountType, requestGB, *(p.options.CloudTags))
		if err != nil {
			return nil, err
		}
	} else {
		diskUri, err = p.plugin.blobDiskController.CreateDataDisk(name, storageAccountType, requestGB, forceStandAlone)
		if err != nil {
			return nil, err
		}
	}

	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:   p.options.PVName,
			Labels: map[string]string{},
			Annotations: map[string]string{
				"kubernetes.io/createdby": "azure-disk-dynamic-provisioner",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: p.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   supportedModes,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", requestGB)),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				AzureDisk: &v1.AzureDiskVolumeSource{
					CachingMode: &cachingMode,
					DiskName:    name,
					DataDiskURI: diskUri,
					Kind:        &kind,
				},
			},
		},
	}
	return pv, nil
}
