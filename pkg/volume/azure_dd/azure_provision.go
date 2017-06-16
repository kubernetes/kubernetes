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

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

var _ volume.DeletableVolumePlugin = &azureDataDiskPlugin{}
var _ volume.ProvisionableVolumePlugin = &azureDataDiskPlugin{}

type azureDiskDeleter struct {
	*azureDisk
	azureProvider azureCloudProvider
}

func (plugin *azureDataDiskPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	azure, err := getAzureCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		glog.V(4).Infof("failed to get azure provider")
		return nil, err
	}

	return plugin.newDeleterInternal(spec, azure)
}

func (plugin *azureDataDiskPlugin) newDeleterInternal(spec *volume.Spec, azure azureCloudProvider) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.AzureDisk == nil {
		return nil, fmt.Errorf("invalid PV spec")
	}
	diskName := spec.PersistentVolume.Spec.AzureDisk.DiskName
	diskUri := spec.PersistentVolume.Spec.AzureDisk.DataDiskURI
	return &azureDiskDeleter{
		azureDisk: &azureDisk{
			volName:  spec.Name(),
			diskName: diskName,
			diskUri:  diskUri,
			plugin:   plugin,
		},
		azureProvider: azure,
	}, nil
}

func (plugin *azureDataDiskPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	azure, err := getAzureCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		glog.V(4).Infof("failed to get azure provider")
		return nil, err
	}
	if len(options.PVC.Spec.AccessModes) == 0 {
		options.PVC.Spec.AccessModes = plugin.GetAccessModes()
	}
	return plugin.newProvisionerInternal(options, azure)
}

func (plugin *azureDataDiskPlugin) newProvisionerInternal(options volume.VolumeOptions, azure azureCloudProvider) (volume.Provisioner, error) {
	return &azureDiskProvisioner{
		azureDisk: &azureDisk{
			plugin: plugin,
		},
		azureProvider: azure,
		options:       options,
	}, nil
}

var _ volume.Deleter = &azureDiskDeleter{}

func (d *azureDiskDeleter) GetPath() string {
	name := azureDataDiskPluginName
	return d.plugin.host.GetPodVolumeDir(d.podUID, utilstrings.EscapeQualifiedNameForDisk(name), d.volName)
}

func (d *azureDiskDeleter) Delete() error {
	glog.V(4).Infof("deleting volume %s", d.diskUri)
	return d.azureProvider.DeleteVolume(d.diskName, d.diskUri)
}

type azureDiskProvisioner struct {
	*azureDisk
	azureProvider azureCloudProvider
	options       volume.VolumeOptions
}

var _ volume.Provisioner = &azureDiskProvisioner{}

func (a *azureDiskProvisioner) Provision() (*v1.PersistentVolume, error) {
	if !volume.AccessModesContainedInAll(a.plugin.GetAccessModes(), a.options.PVC.Spec.AccessModes) {
		return nil, fmt.Errorf("invalid AccessModes %v: only AccessModes %v are supported", a.options.PVC.Spec.AccessModes, a.plugin.GetAccessModes())
	}

	var sku, location, account string

	// maxLength = 79 - (4 for ".vhd") = 75
	name := volume.GenerateVolumeName(a.options.ClusterName, a.options.PVName, 75)
	capacity := a.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	requestBytes := capacity.Value()
	requestGB := int(volume.RoundUpSize(requestBytes, 1024*1024*1024))

	// Apply ProvisionerParameters (case-insensitive). We leave validation of
	// the values to the cloud provider.
	for k, v := range a.options.Parameters {
		switch strings.ToLower(k) {
		case "skuname":
			sku = v
		case "location":
			location = v
		case "storageaccount":
			account = v
		default:
			return nil, fmt.Errorf("invalid option %q for volume plugin %s", k, a.plugin.GetPluginName())
		}
	}
	// TODO: implement c.options.ProvisionerSelector parsing
	if a.options.PVC.Spec.Selector != nil {
		return nil, fmt.Errorf("claim.Spec.Selector is not supported for dynamic provisioning on Azure disk")
	}

	diskName, diskUri, sizeGB, err := a.azureProvider.CreateVolume(name, account, sku, location, requestGB)
	if err != nil {
		return nil, err
	}

	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:   a.options.PVName,
			Labels: map[string]string{},
			Annotations: map[string]string{
				volumehelper.VolumeDynamicallyCreatedByKey: "azure-disk-dynamic-provisioner",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: a.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   a.options.PVC.Spec.AccessModes,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGB)),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				AzureDisk: &v1.AzureDiskVolumeSource{
					DiskName:    diskName,
					DataDiskURI: diskUri,
				},
			},
		},
	}
	return pv, nil
}
