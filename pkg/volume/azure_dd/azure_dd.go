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
	"os"
	"path"

	"github.com/Azure/azure-sdk-for-go/arm/compute"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/azure"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/keymutex"
	"k8s.io/kubernetes/pkg/util/mount"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&azureDataDiskPlugin{}}
}

type azureDataDiskPlugin struct {
	host        volume.VolumeHost
	volumeLocks keymutex.KeyMutex
}

// Abstract interface to disk operations.
// azure cloud provider should implement it
type azureCloudProvider interface {
	// Attaches the disk to the host machine.
	AttachDisk(diskName, diskUri, vmName string, lun int32, cachingMode compute.CachingTypes) error
	// Detaches the disk, identified by disk name or uri, from the host machine.
	DetachDiskByName(diskName, diskUri, vmName string) error
	// Get the LUN number of the disk that is attached to the host
	GetDiskLun(diskName, diskUri, vmName string) (int32, error)
	// Get the next available LUN number to attach a new VHD
	GetNextDiskLun(vmName string) (int32, error)
	// InstanceID returns the cloud provider ID of the specified instance.
	InstanceID(name string) (string, error)
}

var _ volume.VolumePlugin = &azureDataDiskPlugin{}
var _ volume.PersistentVolumePlugin = &azureDataDiskPlugin{}

const (
	azureDataDiskPluginName = "kubernetes.io/azure-disk"
)

func (plugin *azureDataDiskPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	plugin.volumeLocks = keymutex.NewKeyMutex()
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

	return volumeSource.DiskName, nil
}

func (plugin *azureDataDiskPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.AzureDisk != nil) ||
		(spec.Volume != nil && spec.Volume.AzureDisk != nil)
}

func (plugin *azureDataDiskPlugin) RequiresRemount() bool {
	return false
}

func (plugin *azureDataDiskPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
	}
}

func (plugin *azureDataDiskPlugin) NewMounter(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	return plugin.newMounterInternal(spec, pod.UID, plugin.host.GetMounter())
}

func (plugin *azureDataDiskPlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, mounter mount.Interface) (volume.Mounter, error) {
	// azures used directly in a pod have a ReadOnly flag set by the pod author.
	// azures used as a PersistentVolume gets the ReadOnly flag indirectly through the persistent-claim volume used to mount the PV
	azure, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	fsType := *azure.FSType
	diskName := azure.DiskName
	diskUri := azure.DataDiskURI
	cachingMode := *azure.CachingMode
	return &azureDiskMounter{
		azureDisk: &azureDisk{
			podUID:      podUID,
			volName:     spec.Name(),
			diskName:    diskName,
			diskUri:     diskUri,
			cachingMode: cachingMode,
			mounter:     mounter,
			plugin:      plugin,
		},
		fsType:      fsType,
		readOnly:    *azure.ReadOnly,
		diskMounter: &mount.SafeFormatAndMount{Interface: plugin.host.GetMounter(), Runner: exec.New()}}, nil
}

func (plugin *azureDataDiskPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, plugin.host.GetMounter())
}

func (plugin *azureDataDiskPlugin) newUnmounterInternal(volName string, podUID types.UID, mounter mount.Interface) (volume.Unmounter, error) {
	return &azureDiskUnmounter{
		&azureDisk{
			podUID:  podUID,
			volName: volName,
			mounter: mounter,
			plugin:  plugin,
		},
	}, nil
}

func (plugin *azureDataDiskPlugin) ConstructVolumeSpec(volName, mountPath string) (*volume.Spec, error) {
	mounter := plugin.host.GetMounter()
	pluginDir := plugin.host.GetPluginDir(plugin.GetPluginName())
	sourceName, err := mounter.GetDeviceNameFromMount(mountPath, pluginDir)
	if err != nil {
		return nil, err
	}
	azVolume := &api.Volume{
		Name: volName,
		VolumeSource: api.VolumeSource{
			AzureDisk: &api.AzureDiskVolumeSource{
				DiskName: sourceName,
			},
		},
	}
	return volume.NewSpecFromVolume(azVolume), nil
}

func (plugin *azureDataDiskPlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	mounter := plugin.host.GetMounter()
	return mount.GetMountRefs(mounter, deviceMountPath)
}

type azureDisk struct {
	volName     string
	podUID      types.UID
	diskName    string
	diskUri     string
	cachingMode api.AzureDataDiskCachingMode
	mounter     mount.Interface
	plugin      *azureDataDiskPlugin
	volume.MetricsNil
}

type azureDiskMounter struct {
	*azureDisk
	// Filesystem type, optional.
	fsType string
	// Specifies whether the disk will be attached as read-only.
	readOnly bool
	// diskMounter provides the interface that is used to mount the actual block device.
	diskMounter *mount.SafeFormatAndMount
}

var _ volume.Mounter = &azureDiskMounter{}

func (b *azureDiskMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         !b.readOnly,
		SupportsSELinux: true,
	}
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *azureDiskMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

// SetUpAt attaches the disk and bind mounts to the volume path.
func (b *azureDiskMounter) SetUpAt(dir string, fsGroup *int64) error {
	b.plugin.volumeLocks.LockKey(b.diskName)
	defer b.plugin.volumeLocks.UnlockKey(b.diskName)

	// TODO: handle failed mounts here.
	notMnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	glog.V(4).Infof("DataDisk set up: %s %v %v", dir, !notMnt, err)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("IsLikelyNotMountPoint failed: %v", err)
		return err
	}
	if !notMnt {
		glog.V(4).Infof("%s is a mount point", dir)
		return nil
	}

	globalPDPath := makeGlobalPDPath(b.plugin.host, b.diskName)

	if err := os.MkdirAll(dir, 0750); err != nil {
		glog.V(4).Infof("Could not create directory %s: %v", dir, err)
		return err
	}

	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	options := []string{"bind"}
	if b.readOnly {
		options = append(options, "ro")
	}
	err = b.mounter.Mount(globalPDPath, dir, "", options)
	if err != nil {
		notMnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
		if mntErr != nil {
			glog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
			return err
		}
		if !notMnt {
			if mntErr = b.mounter.Unmount(dir); mntErr != nil {
				glog.Errorf("Failed to unmount: %v", mntErr)
				return err
			}
			notMnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
			if mntErr != nil {
				glog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
				return err
			}
			if !notMnt {
				// This is very odd, we don't expect it.  We'll try again next sync loop.
				glog.Errorf("%s is still mounted, despite call to unmount().  Will try again next sync loop.", dir)
				return err
			}
		}
		os.Remove(dir)
		return err
	}

	if !b.readOnly {
		volume.SetVolumeOwnership(b, fsGroup)
	}
	glog.V(3).Infof("Azure disk volume %s mounted to %s", b.diskName, dir)
	return nil
}

func makeGlobalPDPath(host volume.VolumeHost, volume string) string {
	return path.Join(host.GetPluginDir(azureDataDiskPluginName), "mounts", volume)
}

func (azure *azureDisk) GetPath() string {
	name := azureDataDiskPluginName
	return azure.plugin.host.GetPodVolumeDir(azure.podUID, utilstrings.EscapeQualifiedNameForDisk(name), azure.volName)
}

type azureDiskUnmounter struct {
	*azureDisk
}

var _ volume.Unmounter = &azureDiskUnmounter{}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (c *azureDiskUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (c *azureDiskUnmounter) TearDownAt(dir string) error {
	notMnt, err := c.mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		glog.Errorf("Error checking if mountpoint %s: %v", dir, err)
		return err
	}
	if notMnt {
		glog.V(2).Info("Not mountpoint, deleting")
		return os.Remove(dir)
	}
	// lock the volume (and thus wait for any concurrrent SetUpAt to finish)
	c.plugin.volumeLocks.LockKey(c.diskName)
	defer c.plugin.volumeLocks.UnlockKey(c.diskName)
	refs, err := mount.GetMountRefs(c.mounter, dir)
	if err != nil {
		glog.Errorf("Error getting mountrefs for %s: %v", dir, err)
		return err
	}
	if len(refs) == 0 {
		glog.Errorf("Did not find pod-mount for %s during tear down", dir)
		return fmt.Errorf("%s is not mounted", dir)
	}
	c.diskName = path.Base(refs[0])
	glog.V(4).Infof("Found volume %s mounted to %s", c.diskName, dir)

	// Unmount the bind-mount inside this pod
	if err := c.mounter.Unmount(dir); err != nil {
		glog.Errorf("Error unmounting dir %s %v", dir, err)
		return err
	}
	notMnt, mntErr := c.mounter.IsLikelyNotMountPoint(dir)
	if mntErr != nil {
		glog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
		return err
	}
	if notMnt {
		if err := os.Remove(dir); err != nil {
			glog.Errorf("Error removing mountpoint %s %v", dir, err)
			return err
		}
	}
	return nil
}

func getVolumeSource(spec *volume.Spec) (*api.AzureDiskVolumeSource, error) {
	if spec.Volume != nil && spec.Volume.AzureDisk != nil {
		return spec.Volume.AzureDisk, nil
	}
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.AzureDisk != nil {
		return spec.PersistentVolume.Spec.AzureDisk, nil
	}

	return nil, fmt.Errorf("Spec does not reference an Azure disk volume type")
}

// Return cloud provider
func getAzureCloudProvider(cloudProvider cloudprovider.Interface) (azureCloudProvider, error) {
	azureCloudProvider, ok := cloudProvider.(*azure.Cloud)
	if !ok || azureCloudProvider == nil {
		return nil, fmt.Errorf("Failed to get Azure Cloud Provider. GetCloudProvider returned %v instead", cloudProvider)
	}

	return azureCloudProvider, nil
}
