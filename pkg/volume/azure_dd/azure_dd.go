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

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/azure"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

type azureDataDiskPlugin struct {
	host volume.VolumeHost

	commonController      *controllerCommon
	blobDiskController    BlobDiskController
	managedDiskController ManagedDiskController
}

var _ volume.VolumePlugin = &azureDataDiskPlugin{}
var _ volume.PersistentVolumePlugin = &azureDataDiskPlugin{}
var _ volume.DeletableVolumePlugin = &azureDataDiskPlugin{}
var _ volume.ProvisionableVolumePlugin = &azureDataDiskPlugin{}
var _ volume.AttachableVolumePlugin = &azureDataDiskPlugin{}

const (
	azureDataDiskPluginName = "kubernetes.io/azure-disk"
)

func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&azureDataDiskPlugin{}}
}

func (plugin *azureDataDiskPlugin) Init(host volume.VolumeHost) error {
	cloudProvider := host.GetCloudProvider()
	az, ok := cloudProvider.(*azure.Cloud)

	if !ok || az == nil {
		return fmt.Errorf("AzureDisk -  failed to get Azure Cloud Provider. GetCloudProvider returned %v instead", cloudProvider)
	}

	// Common controller contains the function
	// needed by both blob disk and managed disk controllers

	common := &controllerCommon{
		aadResourceEndPoint: az.Environment.ServiceManagementEndpoint,
		clientId:            az.AADClientID,
		clientSecret:        az.AADClientSecret,
		location:            az.Location,
		managementEndpoint:  az.Environment.ResourceManagerEndpoint,
		resourceGroup:       az.ResourceGroup,
		tenantId:            az.TenantID,
		tokenEndPoint:       az.Environment.ActiveDirectoryEndpoint,
		subscriptionId:      az.SubscriptionID,
	}

	// BlobDiskController: contains the function needed to
	// create/attach/detach/delete blob based (unmanaged disks)
	blobController, err := newBlobDiskController(common)
	if err != nil {
		return fmt.Errorf("AzureDisk -  failed to init Blob Disk Controller with error (%s)", err.Error())
	}

	// ManagedDiskController: contains the functions needed to
	// create/attach/detach/delete managed disks
	managedController, err := newManagedDiskController(common)
	if err != nil {
		return fmt.Errorf("AzureDisk -  failed to init Managed  Disk Controller with error (%s)", err.Error())
	}

	plugin.host = host
	plugin.blobDiskController = blobController
	plugin.managedDiskController = managedController
	plugin.commonController = common
	return nil
}

func (plugin *azureDataDiskPlugin) GetPluginName() string {
	return azureDataDiskPluginName
}

func (plugin *azureDataDiskPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	if volumeSource.Kind == nil {
		// Called with a partial volume spec
		// i.e constructed by ConstructVolumeSpec
		return volumeSource.DataDiskURI, nil
	}

	isManaged := (*volumeSource.Kind == v1.AzureManagedDisk)
	uniqueDiskNameTemplate := "%s%s"
	hashedDiskUri := makeCRC32(strings.ToLower(volumeSource.DataDiskURI))

	prefix := "b"
	if isManaged {
		prefix = "m"
	}
	diskName := fmt.Sprintf(uniqueDiskNameTemplate, prefix, hashedDiskUri)
	return diskName, nil
}

func (plugin *azureDataDiskPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.AzureDisk != nil) ||
		(spec.Volume != nil && spec.Volume.AzureDisk != nil)
}

func (plugin *azureDataDiskPlugin) RequiresRemount() bool {
	return false
}

func (plugin *azureDataDiskPlugin) SupportsMountOption() bool {
	return true
}

func (plugin *azureDataDiskPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *azureDataDiskPlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
	}
}
func (plugin *azureDataDiskPlugin) NewAttacher() (volume.Attacher, error) {
	return &azureDiskAttacher{
		plugin: plugin,
	}, nil
}

func (plugin *azureDataDiskPlugin) NewDetacher() (volume.Detacher, error) {
	return &azureDiskDetacher{
		plugin: plugin,
	}, nil
}

func (plugin *azureDataDiskPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	volumeSource, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	disk := makeDataDisk(spec.Name(), "", volumeSource.DiskName, plugin.host)

	return &azureDiskDeleter{
		spec:     spec,
		plugin:   plugin,
		dataDisk: disk,
	}, nil
}

func (plugin *azureDataDiskPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	if len(options.PVC.Spec.AccessModes) == 0 {
		options.PVC.Spec.AccessModes = plugin.GetAccessModes()
	}

	return &azureDiskProvisioner{
		plugin:  plugin,
		options: options,
	}, nil
}

func (plugin *azureDataDiskPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, options volume.VolumeOptions) (volume.Mounter, error) {
	volumeSource, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}
	disk := makeDataDisk(spec.Name(), pod.UID, volumeSource.DiskName, plugin.host)

	return &azureDiskMounter{
		plugin:   plugin,
		spec:     spec,
		options:  options,
		dataDisk: disk,
	}, nil
}

func (plugin *azureDataDiskPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	disk := makeDataDisk(volName, podUID, "", plugin.host)

	return &azureDiskUnmounter{
		plugin:   plugin,
		dataDisk: disk,
	}, nil
}

func (plugin *azureDataDiskPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	mounter := plugin.host.GetMounter()
	pluginDir := plugin.host.GetPluginDir(plugin.GetPluginName())
	sourceName, err := mounter.GetDeviceNameFromMount(mountPath, pluginDir)

	if err != nil {
		return nil, err
	}

	azureVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			AzureDisk: &v1.AzureDiskVolumeSource{
				DataDiskURI: sourceName,
			},
		},
	}
	return volume.NewSpecFromVolume(azureVolume), nil
}

func (plugin *azureDataDiskPlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	m := plugin.host.GetMounter()
	return mount.GetMountRefs(m, deviceMountPath)
}
