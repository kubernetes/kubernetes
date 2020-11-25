// +build !providerless

/*
Copyright 2014 The Kubernetes Authors.

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

package gcepd

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"

	"k8s.io/klog/v2"
	"k8s.io/mount-utils"
	utilstrings "k8s.io/utils/strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	volumehelpers "k8s.io/cloud-provider/volume/helpers"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	gcecloud "k8s.io/legacy-cloud-providers/gce"
)

// ProbeVolumePlugins is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&gcePersistentDiskPlugin{nil}}
}

type gcePersistentDiskPlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &gcePersistentDiskPlugin{}
var _ volume.PersistentVolumePlugin = &gcePersistentDiskPlugin{}
var _ volume.DeletableVolumePlugin = &gcePersistentDiskPlugin{}
var _ volume.ProvisionableVolumePlugin = &gcePersistentDiskPlugin{}
var _ volume.ExpandableVolumePlugin = &gcePersistentDiskPlugin{}
var _ volume.VolumePluginWithAttachLimits = &gcePersistentDiskPlugin{}

const (
	gcePersistentDiskPluginName = "kubernetes.io/gce-pd"
)

// The constants are used to map from the machine type (number of CPUs) to the limit of
// persistent disks that can be attached to an instance. Please refer to gcloud doc
// https://cloud.google.com/compute/docs/machine-types
// These constants are all the documented attach limit minus one because the
// node boot disk is considered an attachable disk so effective attach limit is
// one less.
const (
	volumeLimitSmall = 15
	volumeLimitBig   = 127
)

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, utilstrings.EscapeQualifiedName(gcePersistentDiskPluginName), volName)
}

func (plugin *gcePersistentDiskPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *gcePersistentDiskPlugin) GetPluginName() string {
	return gcePersistentDiskPluginName
}

func (plugin *gcePersistentDiskPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return volumeSource.PDName, nil
}

func (plugin *gcePersistentDiskPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.GCEPersistentDisk != nil) ||
		(spec.Volume != nil && spec.Volume.GCEPersistentDisk != nil)
}

func (plugin *gcePersistentDiskPlugin) RequiresRemount(spec *volume.Spec) bool {
	return false
}

func (plugin *gcePersistentDiskPlugin) SupportsMountOption() bool {
	return true
}

func (plugin *gcePersistentDiskPlugin) SupportsBulkVolumeVerification() bool {
	return true
}

func (plugin *gcePersistentDiskPlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
		v1.ReadOnlyMany,
	}
}

func (plugin *gcePersistentDiskPlugin) GetVolumeLimits() (map[string]int64, error) {
	volumeLimits := map[string]int64{
		util.GCEVolumeLimitKey: volumeLimitSmall,
	}
	cloud := plugin.host.GetCloudProvider()

	// if we can't fetch cloudprovider we return an error
	// hoping external CCM or admin can set it. Returning
	// default values from here will mean, no one can
	// override them.
	if cloud == nil {
		return nil, fmt.Errorf("no cloudprovider present")
	}

	if cloud.ProviderName() != gcecloud.ProviderName {
		return nil, fmt.Errorf("expected gce cloud got %s", cloud.ProviderName())
	}

	instances, ok := cloud.Instances()
	if !ok {
		klog.Warning("Failed to get instances from cloud provider")
		return volumeLimits, nil
	}

	instanceType, err := instances.InstanceType(context.TODO(), plugin.host.GetNodeName())
	if err != nil {
		klog.Errorf("Failed to get instance type from GCE cloud provider")
		return volumeLimits, nil
	}
	smallMachineTypes := []string{"f1-micro", "g1-small", "e2-micro", "e2-small", "e2-medium"}
	for _, small := range smallMachineTypes {
		if instanceType == small {
			volumeLimits[util.GCEVolumeLimitKey] = volumeLimitSmall
			return volumeLimits, nil
		}
	}
	volumeLimits[util.GCEVolumeLimitKey] = volumeLimitBig
	return volumeLimits, nil
}

func (plugin *gcePersistentDiskPlugin) VolumeLimitKey(spec *volume.Spec) string {
	return util.GCEVolumeLimitKey
}

func (plugin *gcePersistentDiskPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newMounterInternal(spec, pod.UID, &GCEDiskUtil{}, plugin.host.GetMounter(plugin.GetPluginName()))
}

func getVolumeSource(
	spec *volume.Spec) (*v1.GCEPersistentDiskVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.GCEPersistentDisk != nil {
		return spec.Volume.GCEPersistentDisk, spec.Volume.GCEPersistentDisk.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.GCEPersistentDisk != nil {
		return spec.PersistentVolume.Spec.GCEPersistentDisk, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("spec does not reference a GCE volume type")
}

func (plugin *gcePersistentDiskPlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, manager pdManager, mounter mount.Interface) (volume.Mounter, error) {
	// GCEPDs used directly in a pod have a ReadOnly flag set by the pod author.
	// GCEPDs used as a PersistentVolume gets the ReadOnly flag indirectly through the persistent-claim volume used to mount the PV
	volumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	pdName := volumeSource.PDName
	partition := ""
	if volumeSource.Partition != 0 {
		partition = strconv.Itoa(int(volumeSource.Partition))
	}

	return &gcePersistentDiskMounter{
		gcePersistentDisk: &gcePersistentDisk{
			podUID:          podUID,
			volName:         spec.Name(),
			pdName:          pdName,
			partition:       partition,
			mounter:         mounter,
			manager:         manager,
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, spec.Name(), plugin.host)),
		},
		mountOptions: util.MountOptionFromSpec(spec),
		readOnly:     readOnly}, nil
}

func (plugin *gcePersistentDiskPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newUnmounterInternal(volName, podUID, &GCEDiskUtil{}, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *gcePersistentDiskPlugin) newUnmounterInternal(volName string, podUID types.UID, manager pdManager, mounter mount.Interface) (volume.Unmounter, error) {
	return &gcePersistentDiskUnmounter{&gcePersistentDisk{
		podUID:          podUID,
		volName:         volName,
		manager:         manager,
		mounter:         mounter,
		plugin:          plugin,
		MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, volName, plugin.host)),
	}}, nil
}

func (plugin *gcePersistentDiskPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	return plugin.newDeleterInternal(spec, &GCEDiskUtil{})
}

func (plugin *gcePersistentDiskPlugin) newDeleterInternal(spec *volume.Spec, manager pdManager) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.GCEPersistentDisk == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.GCEPersistentDisk is nil")
	}
	return &gcePersistentDiskDeleter{
		gcePersistentDisk: &gcePersistentDisk{
			volName: spec.Name(),
			pdName:  spec.PersistentVolume.Spec.GCEPersistentDisk.PDName,
			manager: manager,
			plugin:  plugin,
		}}, nil
}

func (plugin *gcePersistentDiskPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	return plugin.newProvisionerInternal(options, &GCEDiskUtil{})
}

func (plugin *gcePersistentDiskPlugin) newProvisionerInternal(options volume.VolumeOptions, manager pdManager) (volume.Provisioner, error) {
	return &gcePersistentDiskProvisioner{
		gcePersistentDisk: &gcePersistentDisk{
			manager: manager,
			plugin:  plugin,
		},
		options: options,
	}, nil
}

func (plugin *gcePersistentDiskPlugin) RequiresFSResize() bool {
	return true
}

func (plugin *gcePersistentDiskPlugin) ExpandVolumeDevice(
	spec *volume.Spec,
	newSize resource.Quantity,
	oldSize resource.Quantity) (resource.Quantity, error) {
	cloud, err := getCloudProvider(plugin.host.GetCloudProvider())

	if err != nil {
		return oldSize, err
	}
	pdName := spec.PersistentVolume.Spec.GCEPersistentDisk.PDName
	updatedQuantity, err := cloud.ResizeDisk(pdName, oldSize, newSize)

	if err != nil {
		return oldSize, err
	}
	return updatedQuantity, nil
}

func (plugin *gcePersistentDiskPlugin) NodeExpand(resizeOptions volume.NodeResizeOptions) (bool, error) {
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

var _ volume.NodeExpandableVolumePlugin = &gcePersistentDiskPlugin{}

func (plugin *gcePersistentDiskPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
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
	gceVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
				PDName: sourceName,
			},
		},
	}
	return volume.NewSpecFromVolume(gceVolume), nil
}

// Abstract interface to PD operations.
type pdManager interface {
	// Creates a volume
	CreateVolume(provisioner *gcePersistentDiskProvisioner, node *v1.Node, allowedTopologies []v1.TopologySelectorTerm) (volumeID string, volumeSizeGB int, labels map[string]string, fstype string, err error)
	// Deletes a volume
	DeleteVolume(deleter *gcePersistentDiskDeleter) error
}

// gcePersistentDisk volumes are disk resources provided by Google Compute Engine
// that are attached to the kubelet's host machine and exposed to the pod.
type gcePersistentDisk struct {
	volName string
	podUID  types.UID
	// Unique identifier of the PD, used to find the disk resource in the provider.
	pdName string
	// Specifies the partition to mount
	partition string
	// Utility interface to provision and delete disks
	manager pdManager
	// Mounter interface that provides system calls to mount the global path to the pod local path.
	mounter mount.Interface
	plugin  *gcePersistentDiskPlugin
	volume.MetricsProvider
}

type gcePersistentDiskMounter struct {
	*gcePersistentDisk
	// Specifies whether the disk will be mounted as read-only.
	readOnly     bool
	mountOptions []string
}

var _ volume.Mounter = &gcePersistentDiskMounter{}

func (b *gcePersistentDiskMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         !b.readOnly,
		SupportsSELinux: true,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *gcePersistentDiskMounter) CanMount() error {
	return nil
}

// SetUp bind mounts the disk global mount to the volume path.
func (b *gcePersistentDiskMounter) SetUp(mounterArgs volume.MounterArgs) error {
	return b.SetUpAt(b.GetPath(), mounterArgs)
}

// SetUp bind mounts the disk global mount to the give volume path.
func (b *gcePersistentDiskMounter) SetUpAt(dir string, mounterArgs volume.MounterArgs) error {
	// TODO: handle failed mounts here.
	notMnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	klog.V(4).Infof("GCE PersistentDisk set up: Dir (%s) PD name (%q) Mounted (%t) Error (%v), ReadOnly (%t)", dir, b.pdName, !notMnt, err, b.readOnly)
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("cannot validate mount point: %s %v", dir, err)
	}
	if !notMnt {
		return nil
	}

	if runtime.GOOS != "windows" {
		// in windows, we will use mklink to mount, will MkdirAll in Mount func
		if err := os.MkdirAll(dir, 0750); err != nil {
			return fmt.Errorf("mkdir failed on disk %s (%v)", dir, err)
		}
	}

	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	options := []string{"bind"}
	if b.readOnly {
		options = append(options, "ro")
	}

	globalPDPath := makeGlobalPDName(b.plugin.host, b.pdName)
	klog.V(4).Infof("attempting to mount %s", dir)

	mountOptions := util.JoinMountOptions(b.mountOptions, options)

	err = b.mounter.MountSensitiveWithoutSystemd(globalPDPath, dir, "", mountOptions, nil)
	if err != nil {
		notMnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
		if mntErr != nil {
			return fmt.Errorf("failed to mount: %v. Cleanup IsLikelyNotMountPoint check failed: %v", err, mntErr)
		}
		if !notMnt {
			if mntErr = b.mounter.Unmount(dir); mntErr != nil {
				return fmt.Errorf("failed to mount: %v. Cleanup failed to unmount: %v", err, mntErr)
			}
			notMnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
			if mntErr != nil {
				return fmt.Errorf("failed to mount: %v. Cleanup IsLikelyNotMountPoint check failed: %v", err, mntErr)
			}
			if !notMnt {
				// This is very odd, we don't expect it.  We'll try again next sync loop.
				return fmt.Errorf("%s is still mounted, despite call to unmount().  Will try again next sync loop", dir)
			}
		}
		mntErr = os.Remove(dir)
		if mntErr != nil {
			return fmt.Errorf("failed to mount: %v. Cleanup os Remove(%s) failed: %v", err, dir, mntErr)
		}

		return fmt.Errorf("mount of disk %s failed: %v", dir, err)
	}

	if !b.readOnly {
		volume.SetVolumeOwnership(b, mounterArgs.FsGroup, mounterArgs.FSGroupChangePolicy, util.FSGroupCompleteHook(b.plugin, nil))
	}
	return nil
}

func makeGlobalPDName(host volume.VolumeHost, devName string) string {
	return filepath.Join(host.GetPluginDir(gcePersistentDiskPluginName), util.MountsInGlobalPDPath, devName)
}

func (b *gcePersistentDiskMounter) GetPath() string {
	return getPath(b.podUID, b.volName, b.plugin.host)
}

type gcePersistentDiskUnmounter struct {
	*gcePersistentDisk
}

var _ volume.Unmounter = &gcePersistentDiskUnmounter{}

func (c *gcePersistentDiskUnmounter) GetPath() string {
	return getPath(c.podUID, c.volName, c.plugin.host)
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (c *gcePersistentDiskUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

// TearDownAt unmounts the bind mount
func (c *gcePersistentDiskUnmounter) TearDownAt(dir string) error {
	return mount.CleanupMountPoint(dir, c.mounter, false)
}

type gcePersistentDiskDeleter struct {
	*gcePersistentDisk
}

var _ volume.Deleter = &gcePersistentDiskDeleter{}

func (d *gcePersistentDiskDeleter) GetPath() string {
	return getPath(d.podUID, d.volName, d.plugin.host)
}

func (d *gcePersistentDiskDeleter) Delete() error {
	return d.manager.DeleteVolume(d)
}

type gcePersistentDiskProvisioner struct {
	*gcePersistentDisk
	options volume.VolumeOptions
}

var _ volume.Provisioner = &gcePersistentDiskProvisioner{}

func (c *gcePersistentDiskProvisioner) Provision(selectedNode *v1.Node, allowedTopologies []v1.TopologySelectorTerm) (*v1.PersistentVolume, error) {
	if !util.AccessModesContainedInAll(c.plugin.GetAccessModes(), c.options.PVC.Spec.AccessModes) {
		return nil, fmt.Errorf("invalid AccessModes %v: only AccessModes %v are supported", c.options.PVC.Spec.AccessModes, c.plugin.GetAccessModes())
	}

	volumeID, sizeGB, labels, fstype, err := c.manager.CreateVolume(c, selectedNode, allowedTopologies)
	if err != nil {
		return nil, err
	}

	if fstype == "" {
		fstype = "ext4"
	}

	volumeMode := c.options.PVC.Spec.VolumeMode
	if volumeMode != nil && *volumeMode == v1.PersistentVolumeBlock {
		// Block volumes should not have any FSType
		fstype = ""
	}

	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:   c.options.PVName,
			Labels: map[string]string{},
			Annotations: map[string]string{
				util.VolumeDynamicallyCreatedByKey: "gce-pd-dynamic-provisioner",
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
				GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
					PDName:    volumeID,
					Partition: 0,
					ReadOnly:  false,
					FSType:    fstype,
				},
			},
			MountOptions: c.options.MountOptions,
		},
	}
	if len(c.options.PVC.Spec.AccessModes) == 0 {
		pv.Spec.AccessModes = c.plugin.GetAccessModes()
	}

	requirements := make([]v1.NodeSelectorRequirement, 0)
	if len(labels) != 0 {
		if pv.Labels == nil {
			pv.Labels = make(map[string]string)
		}
		for k, v := range labels {
			pv.Labels[k] = v
			var values []string
			if k == v1.LabelFailureDomainBetaZone {
				values, err = volumehelpers.LabelZonesToList(v)
				if err != nil {
					return nil, fmt.Errorf("failed to convert label string for Zone: %s to a List: %v", v, err)
				}
			} else {
				values = []string{v}
			}
			requirements = append(requirements, v1.NodeSelectorRequirement{Key: k, Operator: v1.NodeSelectorOpIn, Values: values})
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
