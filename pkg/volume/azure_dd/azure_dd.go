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
	"context"
	"fmt"
	"strings"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-04-01/compute"
	"github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2017-10-01/storage"
	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/azure"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

// interface exposed by the cloud provider implementing Disk functionality
type DiskController interface {
	CreateBlobDisk(dataDiskName string, storageAccountType storage.SkuName, sizeGB int) (string, error)
	DeleteBlobDisk(diskUri string) error

	CreateManagedDisk(options *azure.ManagedDiskOptions) (string, error)
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

	// GetAzureDiskLabels gets availability zone labels for Azuredisk.
	GetAzureDiskLabels(diskURI string) (map[string]string, error)

	// GetActiveZones returns all the zones in which k8s nodes are currently running.
	GetActiveZones() (sets.String, error)

	// GetLocation returns the location in which k8s cluster is currently running.
	GetLocation() string
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
var _ volume.DeviceMountableVolumePlugin = &azureDataDiskPlugin{}

// store vm size list in current region
var vmSizeList *[]compute.VirtualMachineSize

const (
	azureDataDiskPluginName = "kubernetes.io/azure-disk"
	defaultAzureVolumeLimit = 16
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
		util.AzureVolumeLimitKey: defaultAzureVolumeLimit,
	}

	az, err := getCloud(plugin.host)
	if err != nil {
		// if we can't fetch cloudprovider we return an error
		// hoping external CCM or admin can set it. Returning
		// default values from here will mean, no one can
		// override them.
		return nil, fmt.Errorf("failed to get azure cloud in GetVolumeLimits, plugin.host: %s", plugin.host.GetHostName())
	}

	instances, ok := az.Instances()
	if !ok {
		glog.Warningf("Failed to get instances from cloud provider")
		return volumeLimits, nil
	}

	instanceType, err := instances.InstanceType(context.TODO(), plugin.host.GetNodeName())
	if err != nil {
		glog.Errorf("Failed to get instance type from Azure cloud provider, nodeName: %s", plugin.host.GetNodeName())
		return volumeLimits, nil
	}

	if vmSizeList == nil {
		result, err := az.VirtualMachineSizesClient.List(context.TODO(), az.Location)
		if err != nil || result.Value == nil {
			glog.Errorf("failed to list vm sizes in GetVolumeLimits, plugin.host: %s, location: %s", plugin.host.GetHostName(), az.Location)
			return volumeLimits, nil
		}
		vmSizeList = result.Value
	}

	volumeLimits = map[string]int64{
		util.AzureVolumeLimitKey: getMaxDataDiskCount(instanceType, vmSizeList),
	}

	return volumeLimits, nil
}

func getMaxDataDiskCount(instanceType string, sizeList *[]compute.VirtualMachineSize) int64 {
	if sizeList == nil {
		return defaultAzureVolumeLimit
	}

	vmsize := strings.ToUpper(instanceType)
	for _, size := range *sizeList {
		if size.Name == nil || size.MaxDataDiskCount == nil {
			glog.Errorf("failed to get vm size in getMaxDataDiskCount")
			continue
		}
		if strings.ToUpper(*size.Name) == vmsize {
			glog.V(2).Infof("got a matching size in getMaxDataDiskCount, Name: %s, MaxDataDiskCount: %s", *size.Name, *size.MaxDataDiskCount)
			return int64(*size.MaxDataDiskCount)
		}
	}
	return defaultAzureVolumeLimit
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
	return m.GetMountRefs(deviceMountPath)
}

func (plugin *azureDataDiskPlugin) NewDeviceMounter() (volume.DeviceMounter, error) {
	return plugin.NewAttacher()
}

func (plugin *azureDataDiskPlugin) NewDeviceUnmounter() (volume.DeviceUnmounter, error) {
	return plugin.NewDetacher()
}
