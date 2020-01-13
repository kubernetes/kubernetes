// +build !providerless

/*
Copyright 2015 The Kubernetes Authors.

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

package cinder

import (
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"

	"k8s.io/klog"
	"k8s.io/utils/keymutex"
	"k8s.io/utils/mount"
	utilstrings "k8s.io/utils/strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/legacy-cloud-providers/openstack"
)

const (
	// DefaultCloudConfigPath is the default path for cloud configuration
	DefaultCloudConfigPath = "/etc/kubernetes/cloud-config"
)

// ProbeVolumePlugins is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&cinderPlugin{}}
}

// BlockStorageProvider is the interface for accessing cinder functionality.
type BlockStorageProvider interface {
	AttachDisk(instanceID, volumeID string) (string, error)
	DetachDisk(instanceID, volumeID string) error
	DeleteVolume(volumeID string) error
	CreateVolume(name string, size int, vtype, availability string, tags *map[string]string) (string, string, string, bool, error)
	GetDevicePath(volumeID string) string
	InstanceID() (string, error)
	GetAttachmentDiskPath(instanceID, volumeID string) (string, error)
	OperationPending(diskName string) (bool, string, error)
	DiskIsAttached(instanceID, volumeID string) (bool, error)
	DiskIsAttachedByName(nodeName types.NodeName, volumeID string) (bool, string, error)
	DisksAreAttachedByName(nodeName types.NodeName, volumeIDs []string) (map[string]bool, error)
	ShouldTrustDevicePath() bool
	Instances() (cloudprovider.Instances, bool)
	ExpandVolume(volumeID string, oldSize resource.Quantity, newSize resource.Quantity) (resource.Quantity, error)
}

type cinderPlugin struct {
	host volume.VolumeHost
	// Guarding SetUp and TearDown operations
	volumeLocks keymutex.KeyMutex
}

var _ volume.VolumePlugin = &cinderPlugin{}
var _ volume.PersistentVolumePlugin = &cinderPlugin{}
var _ volume.DeletableVolumePlugin = &cinderPlugin{}
var _ volume.ProvisionableVolumePlugin = &cinderPlugin{}

const (
	cinderVolumePluginName = "kubernetes.io/cinder"
)

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, utilstrings.EscapeQualifiedName(cinderVolumePluginName), volName)
}

func (plugin *cinderPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	plugin.volumeLocks = keymutex.NewHashed(0)
	return nil
}

func (plugin *cinderPlugin) GetPluginName() string {
	return cinderVolumePluginName
}

func (plugin *cinderPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeID, _, _, err := getVolumeInfo(spec)
	if err != nil {
		return "", err
	}

	return volumeID, nil
}

func (plugin *cinderPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.Volume != nil && spec.Volume.Cinder != nil) || (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Cinder != nil)
}

func (plugin *cinderPlugin) RequiresRemount() bool {
	return false
}

func (plugin *cinderPlugin) SupportsMountOption() bool {
	return true

}
func (plugin *cinderPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

var _ volume.VolumePluginWithAttachLimits = &cinderPlugin{}

func (plugin *cinderPlugin) GetVolumeLimits() (map[string]int64, error) {
	volumeLimits := map[string]int64{
		util.CinderVolumeLimitKey: util.DefaultMaxCinderVolumes,
	}
	cloud := plugin.host.GetCloudProvider()

	// if we can't fetch cloudprovider we return an error
	// hoping external CCM or admin can set it. Returning
	// default values from here will mean, no one can
	// override them.
	if cloud == nil {
		return nil, fmt.Errorf("No cloudprovider present")
	}

	if cloud.ProviderName() != openstack.ProviderName {
		return nil, fmt.Errorf("Expected Openstack cloud, found %s", cloud.ProviderName())
	}

	openstackCloud, ok := cloud.(*openstack.OpenStack)
	if ok && openstackCloud.NodeVolumeAttachLimit() > 0 {
		volumeLimits[util.CinderVolumeLimitKey] = int64(openstackCloud.NodeVolumeAttachLimit())
	}

	return volumeLimits, nil
}

func (plugin *cinderPlugin) VolumeLimitKey(spec *volume.Spec) string {
	return util.CinderVolumeLimitKey
}

func (plugin *cinderPlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
	}
}

func (plugin *cinderPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	return plugin.newMounterInternal(spec, pod.UID, &DiskUtil{}, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *cinderPlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, manager cdManager, mounter mount.Interface) (volume.Mounter, error) {
	pdName, fsType, readOnly, err := getVolumeInfo(spec)
	if err != nil {
		return nil, err
	}

	return &cinderVolumeMounter{
		cinderVolume: &cinderVolume{
			podUID:          podUID,
			volName:         spec.Name(),
			pdName:          pdName,
			mounter:         mounter,
			manager:         manager,
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, spec.Name(), plugin.host)),
		},
		fsType:             fsType,
		readOnly:           readOnly,
		blockDeviceMounter: util.NewSafeFormatAndMountFromHost(plugin.GetPluginName(), plugin.host),
		mountOptions:       util.MountOptionFromSpec(spec),
	}, nil
}

func (plugin *cinderPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, &DiskUtil{}, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *cinderPlugin) newUnmounterInternal(volName string, podUID types.UID, manager cdManager, mounter mount.Interface) (volume.Unmounter, error) {
	return &cinderVolumeUnmounter{
		&cinderVolume{
			podUID:          podUID,
			volName:         volName,
			manager:         manager,
			mounter:         mounter,
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, volName, plugin.host)),
		}}, nil
}

func (plugin *cinderPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	return plugin.newDeleterInternal(spec, &DiskUtil{})
}

func (plugin *cinderPlugin) newDeleterInternal(spec *volume.Spec, manager cdManager) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Cinder == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.Cinder is nil")
	}
	return &cinderVolumeDeleter{
		&cinderVolume{
			volName: spec.Name(),
			pdName:  spec.PersistentVolume.Spec.Cinder.VolumeID,
			manager: manager,
			plugin:  plugin,
		}}, nil
}

func (plugin *cinderPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	return plugin.newProvisionerInternal(options, &DiskUtil{})
}

func (plugin *cinderPlugin) newProvisionerInternal(options volume.VolumeOptions, manager cdManager) (volume.Provisioner, error) {
	return &cinderVolumeProvisioner{
		cinderVolume: &cinderVolume{
			manager: manager,
			plugin:  plugin,
		},
		options: options,
	}, nil
}

func (plugin *cinderPlugin) getCloudProvider() (BlockStorageProvider, error) {
	cloud := plugin.host.GetCloudProvider()
	if cloud == nil {
		if _, err := os.Stat(DefaultCloudConfigPath); err == nil {
			var config *os.File
			config, err = os.Open(DefaultCloudConfigPath)
			if err != nil {
				return nil, fmt.Errorf("unable to load OpenStack configuration from default path : %v", err)
			}
			defer config.Close()
			cloud, err = cloudprovider.GetCloudProvider(openstack.ProviderName, config)
			if err != nil {
				return nil, fmt.Errorf("unable to create OpenStack cloud provider from default path : %v", err)
			}
		} else {
			return nil, fmt.Errorf("OpenStack cloud provider was not initialized properly : %v", err)
		}
	}

	switch cloud := cloud.(type) {
	case *openstack.OpenStack:
		return cloud, nil
	default:
		return nil, errors.New("invalid cloud provider: expected OpenStack")
	}
}

func (plugin *cinderPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	mounter := plugin.host.GetMounter(plugin.GetPluginName())
	kvh, ok := plugin.host.(volume.KubeletVolumeHost)
	if !ok {
		return nil, fmt.Errorf("plugin volume host does not implement KubeletVolumeHost interface")
	}
	hu := kvh.GetHostUtil()
	pluginMntDir := util.GetPluginMountDir(plugin.host, plugin.GetPluginName())
	sourceName, err := hu.GetDeviceNameFromMount(mounter, mountPath, pluginMntDir)
	if err != nil {
		return nil, err
	}
	klog.V(4).Infof("Found volume %s mounted to %s", sourceName, mountPath)
	cinderVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			Cinder: &v1.CinderVolumeSource{
				VolumeID: sourceName,
			},
		},
	}
	return volume.NewSpecFromVolume(cinderVolume), nil
}

var _ volume.ExpandableVolumePlugin = &cinderPlugin{}

func (plugin *cinderPlugin) ExpandVolumeDevice(spec *volume.Spec, newSize resource.Quantity, oldSize resource.Quantity) (resource.Quantity, error) {
	volumeID, _, _, err := getVolumeInfo(spec)
	if err != nil {
		return oldSize, err
	}
	cloud, err := plugin.getCloudProvider()
	if err != nil {
		return oldSize, err
	}

	expandedSize, err := cloud.ExpandVolume(volumeID, oldSize, newSize)
	if err != nil {
		return oldSize, err
	}

	klog.V(2).Infof("volume %s expanded to new size %d successfully", volumeID, int(newSize.Value()))
	return expandedSize, nil
}

func (plugin *cinderPlugin) NodeExpand(resizeOptions volume.NodeResizeOptions) (bool, error) {
	fsVolume, err := util.CheckVolumeModeFilesystem(resizeOptions.VolumeSpec)
	if err != nil {
		return false, fmt.Errorf("error checking VolumeMode: %v", err)
	}
	// if volume is not a fs file system, there is nothing for us to do here.
	if !fsVolume {
		return true, nil
	}

	_, err = util.GenericResizeFS(plugin.host, plugin.GetPluginName(), resizeOptions.DevicePath, resizeOptions.DeviceMountPath)
	if err != nil {
		return false, err
	}
	return true, nil
}

var _ volume.NodeExpandableVolumePlugin = &cinderPlugin{}

func (plugin *cinderPlugin) RequiresFSResize() bool {
	return true
}

// Abstract interface to PD operations.
type cdManager interface {
	// Attaches the disk to the kubelet's host machine.
	AttachDisk(mounter *cinderVolumeMounter, globalPDPath string) error
	// Detaches the disk from the kubelet's host machine.
	DetachDisk(unmounter *cinderVolumeUnmounter) error
	// Creates a volume
	CreateVolume(provisioner *cinderVolumeProvisioner, node *v1.Node, allowedTopologies []v1.TopologySelectorTerm) (volumeID string, volumeSizeGB int, labels map[string]string, fstype string, err error)
	// Deletes a volume
	DeleteVolume(deleter *cinderVolumeDeleter) error
}

var _ volume.Mounter = &cinderVolumeMounter{}

type cinderVolumeMounter struct {
	*cinderVolume
	fsType             string
	readOnly           bool
	blockDeviceMounter *mount.SafeFormatAndMount
	mountOptions       []string
}

// cinderPersistentDisk volumes are disk resources provided by C3
// that are attached to the kubelet's host machine and exposed to the pod.
type cinderVolume struct {
	volName string
	podUID  types.UID
	// Unique identifier of the volume, used to find the disk resource in the provider.
	pdName string
	// Filesystem type, optional.
	fsType string
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager cdManager
	// Mounter interface that provides system calls to mount the global path to the pod local path.
	mounter mount.Interface
	plugin  *cinderPlugin
	volume.MetricsProvider
}

func (b *cinderVolumeMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         !b.readOnly,
		SupportsSELinux: true,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *cinderVolumeMounter) CanMount() error {
	return nil
}

func (b *cinderVolumeMounter) SetUp(mounterArgs volume.MounterArgs) error {
	return b.SetUpAt(b.GetPath(), mounterArgs)
}

// SetUp bind mounts to the volume path.
func (b *cinderVolumeMounter) SetUpAt(dir string, mounterArgs volume.MounterArgs) error {
	klog.V(5).Infof("Cinder SetUp %s to %s", b.pdName, dir)

	b.plugin.volumeLocks.LockKey(b.pdName)
	defer b.plugin.volumeLocks.UnlockKey(b.pdName)

	notmnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	if err != nil && !os.IsNotExist(err) {
		klog.Errorf("Cannot validate mount point: %s %v", dir, err)
		return err
	}
	if !notmnt {
		klog.V(4).Infof("Something is already mounted to target %s", dir)
		return nil
	}
	globalPDPath := makeGlobalPDName(b.plugin.host, b.pdName)

	options := []string{"bind"}
	if b.readOnly {
		options = append(options, "ro")
	}

	if err := os.MkdirAll(dir, 0750); err != nil {
		klog.V(4).Infof("Could not create directory %s: %v", dir, err)
		return err
	}

	mountOptions := util.JoinMountOptions(options, b.mountOptions)
	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	klog.V(4).Infof("Attempting to mount cinder volume %s to %s with options %v", b.pdName, dir, mountOptions)
	err = b.mounter.Mount(globalPDPath, dir, "", options)
	if err != nil {
		klog.V(4).Infof("Mount failed: %v", err)
		notmnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
		if mntErr != nil {
			klog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
			return err
		}
		if !notmnt {
			if mntErr = b.mounter.Unmount(dir); mntErr != nil {
				klog.Errorf("Failed to unmount: %v", mntErr)
				return err
			}
			notmnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
			if mntErr != nil {
				klog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
				return err
			}
			if !notmnt {
				// This is very odd, we don't expect it.  We'll try again next sync loop.
				klog.Errorf("%s is still mounted, despite call to unmount().  Will try again next sync loop.", b.GetPath())
				return err
			}
		}
		os.Remove(dir)
		klog.Errorf("Failed to mount %s: %v", dir, err)
		return err
	}

	if !b.readOnly {
		volume.SetVolumeOwnership(b, mounterArgs.FsGroup)
	}
	klog.V(3).Infof("Cinder volume %s mounted to %s", b.pdName, dir)

	return nil
}

func makeGlobalPDName(host volume.VolumeHost, devName string) string {
	return filepath.Join(host.GetPluginDir(cinderVolumePluginName), util.MountsInGlobalPDPath, devName)
}

func (cd *cinderVolume) GetPath() string {
	return getPath(cd.podUID, cd.volName, cd.plugin.host)
}

type cinderVolumeUnmounter struct {
	*cinderVolume
}

var _ volume.Unmounter = &cinderVolumeUnmounter{}

func (c *cinderVolumeUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (c *cinderVolumeUnmounter) TearDownAt(dir string) error {
	if pathExists, pathErr := mount.PathExists(dir); pathErr != nil {
		return fmt.Errorf("Error checking if path exists: %v", pathErr)
	} else if !pathExists {
		klog.Warningf("Warning: Unmount skipped because path does not exist: %v", dir)
		return nil
	}

	klog.V(5).Infof("Cinder TearDown of %s", dir)
	notmnt, err := c.mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		klog.V(4).Infof("IsLikelyNotMountPoint check failed: %v", err)
		return err
	}
	if notmnt {
		klog.V(4).Infof("Nothing is mounted to %s, ignoring", dir)
		return os.Remove(dir)
	}

	// Find Cinder volumeID to lock the right volume
	// TODO: refactor VolumePlugin.NewUnmounter to get full volume.Spec just like
	// NewMounter. We could then find volumeID there without probing MountRefs.
	refs, err := c.mounter.GetMountRefs(dir)
	if err != nil {
		klog.V(4).Infof("GetMountRefs failed: %v", err)
		return err
	}
	if len(refs) == 0 {
		klog.V(4).Infof("Directory %s is not mounted", dir)
		return fmt.Errorf("directory %s is not mounted", dir)
	}
	c.pdName = path.Base(refs[0])
	klog.V(4).Infof("Found volume %s mounted to %s", c.pdName, dir)

	// lock the volume (and thus wait for any concurrent SetUpAt to finish)
	c.plugin.volumeLocks.LockKey(c.pdName)
	defer c.plugin.volumeLocks.UnlockKey(c.pdName)

	// Reload list of references, there might be SetUpAt finished in the meantime
	_, err = c.mounter.GetMountRefs(dir)
	if err != nil {
		klog.V(4).Infof("GetMountRefs failed: %v", err)
		return err
	}
	if err := c.mounter.Unmount(dir); err != nil {
		klog.V(4).Infof("Unmount failed: %v", err)
		return err
	}
	klog.V(3).Infof("Successfully unmounted: %s\n", dir)

	notmnt, mntErr := c.mounter.IsLikelyNotMountPoint(dir)
	if mntErr != nil {
		klog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
		return err
	}
	if notmnt {
		if err := os.Remove(dir); err != nil {
			klog.V(4).Infof("Failed to remove directory after unmount: %v", err)
			return err
		}
	}
	return nil
}

type cinderVolumeDeleter struct {
	*cinderVolume
}

var _ volume.Deleter = &cinderVolumeDeleter{}

func (r *cinderVolumeDeleter) GetPath() string {
	return getPath(r.podUID, r.volName, r.plugin.host)
}

func (r *cinderVolumeDeleter) Delete() error {
	return r.manager.DeleteVolume(r)
}

type cinderVolumeProvisioner struct {
	*cinderVolume
	options volume.VolumeOptions
}

var _ volume.Provisioner = &cinderVolumeProvisioner{}

func (c *cinderVolumeProvisioner) Provision(selectedNode *v1.Node, allowedTopologies []v1.TopologySelectorTerm) (*v1.PersistentVolume, error) {
	if !util.AccessModesContainedInAll(c.plugin.GetAccessModes(), c.options.PVC.Spec.AccessModes) {
		return nil, fmt.Errorf("invalid AccessModes %v: only AccessModes %v are supported", c.options.PVC.Spec.AccessModes, c.plugin.GetAccessModes())
	}

	volumeID, sizeGB, labels, fstype, err := c.manager.CreateVolume(c, selectedNode, allowedTopologies)
	if err != nil {
		return nil, err
	}

	var volumeMode *v1.PersistentVolumeMode
	if utilfeature.DefaultFeatureGate.Enabled(features.BlockVolume) {
		volumeMode = c.options.PVC.Spec.VolumeMode
		if volumeMode != nil && *volumeMode == v1.PersistentVolumeBlock {
			// Block volumes should not have any FSType
			fstype = ""
		}
	}

	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:   c.options.PVName,
			Labels: labels,
			Annotations: map[string]string{
				util.VolumeDynamicallyCreatedByKey: "cinder-dynamic-provisioner",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: c.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   c.options.PVC.Spec.AccessModes,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGB)),
			},
			VolumeMode: volumeMode,
			PersistentVolumeSource: v1.PersistentVolumeSource{
				Cinder: &v1.CinderPersistentVolumeSource{
					VolumeID: volumeID,
					FSType:   fstype,
					ReadOnly: false,
				},
			},
			MountOptions: c.options.MountOptions,
		},
	}
	if len(c.options.PVC.Spec.AccessModes) == 0 {
		pv.Spec.AccessModes = c.plugin.GetAccessModes()
	}

	requirements := make([]v1.NodeSelectorRequirement, 0)
	for k, v := range labels {
		if v != "" {
			requirements = append(requirements, v1.NodeSelectorRequirement{Key: k, Operator: v1.NodeSelectorOpIn, Values: []string{v}})
		}
	}
	if len(requirements) > 0 {
		pv.Spec.NodeAffinity = new(v1.VolumeNodeAffinity)
		pv.Spec.NodeAffinity.Required = new(v1.NodeSelector)
		pv.Spec.NodeAffinity.Required.NodeSelectorTerms = make([]v1.NodeSelectorTerm, 1)
		pv.Spec.NodeAffinity.Required.NodeSelectorTerms[0].MatchExpressions = requirements
	}

	return pv, nil
}

func getVolumeInfo(spec *volume.Spec) (string, string, bool, error) {
	if spec.Volume != nil && spec.Volume.Cinder != nil {
		return spec.Volume.Cinder.VolumeID, spec.Volume.Cinder.FSType, spec.Volume.Cinder.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.Cinder != nil {
		return spec.PersistentVolume.Spec.Cinder.VolumeID, spec.PersistentVolume.Spec.Cinder.FSType, spec.ReadOnly, nil
	}

	return "", "", false, fmt.Errorf("Spec does not reference a Cinder volume type")
}
