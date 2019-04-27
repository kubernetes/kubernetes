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

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-03-01/compute"
	"github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2017-10-01/storage"
	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/azure"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

// interface exposed by the cloud provider implementing Disk functionality
type DiskController interface {
	CreateBlobDisk(dataDiskName string, storageAccountType storage.SkuName, sizeGB int) (string, error)
	DeleteBlobDisk(diskUri string) error

	CreateManagedDisk(diskName string, storageAccountType storage.SkuName, resourceGroup string, sizeGB int, tags map[string]string) (string, error)
	DeleteManagedDisk(diskURI string) error

	// Attaches the disk to the host machine.
	AttachDisk(isManagedDisk bool, diskName, diskUri string, nodeName types.NodeName, lun int32, cachingMode compute.CachingTypes) error
	// Detaches the disk, identified by disk name or uri, from the host machine.
	DetachDiskByName(diskName, diskUri string, nodeName types.NodeName) error

	// Check if a list of volumes are attached to the node with the specified NodeName
	DisksAreAttached(diskNames []string, nodeName types.NodeName) (map[string]bool, error)

	// Get the LUN number of the disk that is attached to the host
	GetDiskLun(diskName, diskUri string, nodeName types.NodeName) (int32, error)
	// Get the next available LUN number to attach a new VHD
	GetNextDiskLun(nodeName types.NodeName) (int32, error)

	// Create a VHD blob
	CreateVolume(name, storageAccount, storageAccountType, location string, requestGB int) (string, string, int, error)
	// Delete a VHD blob
	DeleteVolume(diskURI string) error

	// Expand the disk to new size
	ResizeDisk(diskURI string, oldSize resource.Quantity, newSize resource.Quantity) (resource.Quantity, error)
}

type azureDataDiskPlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &azureDataDiskPlugin{}
var _ volume.PersistentVolumePlugin = &azureDataDiskPlugin{}
var _ volume.DeletableVolumePlugin = &azureDataDiskPlugin{}
var _ volume.ProvisionableVolumePlugin = &azureDataDiskPlugin{}
var _ volume.AttachableVolumePlugin = &azureDataDiskPlugin{}
var _ volume.VolumePluginWithAttachLimits = &azureDataDiskPlugin{}
var _ volume.ExpandableVolumePlugin = &azureDataDiskPlugin{}

const (
	azureDataDiskPluginName = "kubernetes.io/azure-disk"
)

func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&azureDataDiskPlugin{}}
}

func (plugin *azureDataDiskPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *azureDataDiskPlugin) GetPluginName() string {
	return azureDataDiskPluginName
}

func (plugin *azureDataDiskPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return volumeSource.DataDiskURI, nil
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

func (plugin *azureDataDiskPlugin) GetVolumeLimits() (map[string]int64, error) {
	volumeLimits := map[string]int64{
		util.AzureVolumeLimitKey: 16,
	}

	cloud := plugin.host.GetCloudProvider()

	// if we can't fetch cloudprovider we return an error
	// hoping external CCM or admin can set it. Returning
	// default values from here will mean, no one can
	// override them.
	if cloud == nil {
		return nil, fmt.Errorf("No cloudprovider present")
	}

	if cloud.ProviderName() != azure.CloudProviderName {
		return nil, fmt.Errorf("Expected Azure cloudprovider, got %s", cloud.ProviderName())
	}

	return volumeLimits, nil
}

func (plugin *azureDataDiskPlugin) VolumeLimitKey(spec *volume.Spec) string {
	return util.AzureVolumeLimitKey
}

func (plugin *azureDataDiskPlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
	}
}

// NewAttacher initializes an Attacher
func (plugin *azureDataDiskPlugin) NewAttacher() (volume.Attacher, error) {
	azure, err := getCloud(plugin.host)
	if err != nil {
		glog.Errorf("failed to get azure cloud in NewAttacher, plugin.host : %s, err:%v", plugin.host.GetHostName(), err)
		return nil, err
	}

	return &azureDiskAttacher{
		plugin: plugin,
		cloud:  azure,
	}, nil
}

func (plugin *azureDataDiskPlugin) NewDetacher() (volume.Detacher, error) {
	azure, err := getCloud(plugin.host)
	if err != nil {
		glog.V(4).Infof("failed to get azure cloud in NewDetacher, plugin.host : %s", plugin.host.GetHostName())
		return nil, err
	}

	return &azureDiskDetacher{
		plugin: plugin,
		cloud:  azure,
	}, nil
}

func (plugin *azureDataDiskPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	disk := makeDataDisk(spec.Name(), "", volumeSource.DiskName, plugin.host, plugin)

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
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}
	disk := makeDataDisk(spec.Name(), pod.UID, volumeSource.DiskName, plugin.host, plugin)

	return &azureDiskMounter{
		plugin:   plugin,
		spec:     spec,
		options:  options,
		dataDisk: disk,
	}, nil
}

func (plugin *azureDataDiskPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	disk := makeDataDisk(volName, podUID, "", plugin.host, plugin)

	return &azureDiskUnmounter{
		plugin:   plugin,
		dataDisk: disk,
	}, nil
}

func (plugin *azureDataDiskPlugin) RequiresFSResize() bool {
	return true
}

func (plugin *azureDataDiskPlugin) ExpandVolumeDevice(
	spec *volume.Spec,
	newSize resource.Quantity,
	oldSize resource.Quantity) (resource.Quantity, error) {
	if spec.PersistentVolume == nil || spec.PersistentVolume.Spec.AzureDisk == nil {
		return oldSize, fmt.Errorf("invalid PV spec")
	}

	diskController, err := getDiskController(plugin.host)
	if err != nil {
		return oldSize, err
	}

	return diskController.ResizeDisk(spec.PersistentVolume.Spec.AzureDisk.DataDiskURI, oldSize, newSize)
}

func (plugin *azureDataDiskPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	mounter := plugin.host.GetMounter(plugin.GetPluginName())
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
	m := plugin.host.GetMounter(plugin.GetPluginName())
	return mount.GetMountRefs(m, deviceMountPath)
}
