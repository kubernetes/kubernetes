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

package azure_file

import (
	"fmt"
	"strings"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/azure"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

var _ volume.DeletableVolumePlugin = &azureFilePlugin{}
var _ volume.ProvisionableVolumePlugin = &azureFilePlugin{}

// Abstract interface to file share operations.
// azure cloud provider should implement it
type azureCloudProvider interface {
	// create a file share
	CreateFileShare(name, storageAccount, storageType, location string, requestGB int) (string, string, error)
	// delete a file share
	DeleteFileShare(accountName, key, name string) error
}

type azureFileDeleter struct {
	*azureFile
	accountName, accountKey, shareName string
	azureProvider                      azureCloudProvider
}

func (plugin *azureFilePlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	azure, err := getAzureCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		glog.V(4).Infof("failed to get azure provider")
		return nil, err
	}

	return plugin.newDeleterInternal(spec, &azureSvc{}, azure)
}

func (plugin *azureFilePlugin) newDeleterInternal(spec *volume.Spec, util azureUtil, azure azureCloudProvider) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.AzureFile == nil {
		return nil, fmt.Errorf("invalid PV spec")
	}
	pvSpec := spec.PersistentVolume
	if pvSpec.Spec.ClaimRef.Namespace == "" {
		glog.Errorf("namespace cannot be nil")
		return nil, fmt.Errorf("invalid PV spec: nil namespace")
	}
	nameSpace := pvSpec.Spec.ClaimRef.Namespace
	secretName := pvSpec.Spec.AzureFile.SecretName
	shareName := pvSpec.Spec.AzureFile.ShareName
	if accountName, accountKey, err := util.GetAzureCredentials(plugin.host, nameSpace, secretName); err != nil {
		return nil, err
	} else {
		return &azureFileDeleter{
			azureFile: &azureFile{
				volName: spec.Name(),
				plugin:  plugin,
			},
			shareName:     shareName,
			accountName:   accountName,
			accountKey:    accountKey,
			azureProvider: azure,
		}, nil
	}
}

func (plugin *azureFilePlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
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

func (plugin *azureFilePlugin) newProvisionerInternal(options volume.VolumeOptions, azure azureCloudProvider) (volume.Provisioner, error) {
	return &azureFileProvisioner{
		azureFile: &azureFile{
			plugin: plugin,
		},
		azureProvider: azure,
		util:          &azureSvc{},
		options:       options,
	}, nil
}

var _ volume.Deleter = &azureFileDeleter{}

func (f *azureFileDeleter) GetPath() string {
	name := azureFilePluginName
	return f.plugin.host.GetPodVolumeDir(f.podUID, utilstrings.EscapeQualifiedNameForDisk(name), f.volName)
}

func (f *azureFileDeleter) Delete() error {
	glog.V(4).Infof("deleting volume %s", f.shareName)
	return f.azureProvider.DeleteFileShare(f.accountName, f.accountKey, f.shareName)
}

type azureFileProvisioner struct {
	*azureFile
	azureProvider azureCloudProvider
	util          azureUtil
	options       volume.VolumeOptions
}

var _ volume.Provisioner = &azureFileProvisioner{}

func (a *azureFileProvisioner) Provision() (*v1.PersistentVolume, error) {
	if !volume.AccessModesContainedInAll(a.plugin.GetAccessModes(), a.options.PVC.Spec.AccessModes) {
		return nil, fmt.Errorf("invalid AccessModes %v: only AccessModes %v are supported", a.options.PVC.Spec.AccessModes, a.plugin.GetAccessModes())
	}

	var sku, location, account string

	// File share name has a length limit of 63, and it cannot contain two consecutive '-'s.
	name := volume.GenerateVolumeName(a.options.ClusterName, a.options.PVName, 63)
	name = strings.Replace(name, "--", "-", -1)
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
		return nil, fmt.Errorf("claim.Spec.Selector is not supported for dynamic provisioning on Azure file")
	}

	account, key, err := a.azureProvider.CreateFileShare(name, account, sku, location, requestGB)
	if err != nil {
		return nil, err
	}
	// create a secret for storage account and key
	secretName, err := a.util.SetAzureCredentials(a.plugin.host, a.options.PVC.Namespace, account, key)
	if err != nil {
		return nil, err
	}
	// create PV
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:   a.options.PVName,
			Labels: map[string]string{},
			Annotations: map[string]string{
				volumehelper.VolumeDynamicallyCreatedByKey: "azure-file-dynamic-provisioner",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: a.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   a.options.PVC.Spec.AccessModes,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", requestGB)),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				AzureFile: &v1.AzureFileVolumeSource{
					SecretName: secretName,
					ShareName:  name,
				},
			},
		},
	}
	return pv, nil
}

// Return cloud provider
func getAzureCloudProvider(cloudProvider cloudprovider.Interface) (azureCloudProvider, error) {
	azureCloudProvider, ok := cloudProvider.(*azure.Cloud)
	if !ok || azureCloudProvider == nil {
		return nil, fmt.Errorf("Failed to get Azure Cloud Provider. GetCloudProvider returned %v instead", cloudProvider)
	}

	return azureCloudProvider, nil
}
