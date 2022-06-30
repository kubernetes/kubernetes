//go:build !providerless
// +build !providerless

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

package vsphere_volume

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/mount-utils"
	utilstrings "k8s.io/utils/strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	volumehelpers "k8s.io/cloud-provider/volume/helpers"

	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&vsphereVolumePlugin{}}
}

type vsphereVolumePlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &vsphereVolumePlugin{}
var _ volume.PersistentVolumePlugin = &vsphereVolumePlugin{}
var _ volume.DeletableVolumePlugin = &vsphereVolumePlugin{}
var _ volume.ProvisionableVolumePlugin = &vsphereVolumePlugin{}

const (
	vsphereVolumePluginName = "kubernetes.io/vsphere-volume"
)

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, utilstrings.EscapeQualifiedName(vsphereVolumePluginName), volName)
}

// vSphere Volume Plugin
func (plugin *vsphereVolumePlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *vsphereVolumePlugin) GetPluginName() string {
	return vsphereVolumePluginName
}

func (plugin *vsphereVolumePlugin) IsMigratedToCSI() bool {
	return utilfeature.DefaultFeatureGate.Enabled(features.CSIMigrationvSphere)
}

func (plugin *vsphereVolumePlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return volumeSource.VolumePath, nil
}

func (plugin *vsphereVolumePlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.VsphereVolume != nil) ||
		(spec.Volume != nil && spec.Volume.VsphereVolume != nil)
}

func (plugin *vsphereVolumePlugin) RequiresRemount(spec *volume.Spec) bool {
	return false
}

func (plugin *vsphereVolumePlugin) SupportsMountOption() bool {
	return true
}

func (plugin *vsphereVolumePlugin) SupportsBulkVolumeVerification() bool {
	return true
}

func (plugin *vsphereVolumePlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	return plugin.newMounterInternal(spec, pod.UID, &VsphereDiskUtil{}, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *vsphereVolumePlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, &VsphereDiskUtil{}, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *vsphereVolumePlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, manager vdManager, mounter mount.Interface) (volume.Mounter, error) {
	vvol, _, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	volPath := vvol.VolumePath
	fsType := vvol.FSType

	return &vsphereVolumeMounter{
		vsphereVolume: &vsphereVolume{
			podUID:          podUID,
			volName:         spec.Name(),
			volPath:         volPath,
			manager:         manager,
			mounter:         mounter,
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, spec.Name(), plugin.host)),
		},
		fsType:       fsType,
		diskMounter:  util.NewSafeFormatAndMountFromHost(plugin.GetPluginName(), plugin.host),
		mountOptions: util.MountOptionFromSpec(spec),
	}, nil
}

func (plugin *vsphereVolumePlugin) newUnmounterInternal(volName string, podUID types.UID, manager vdManager, mounter mount.Interface) (volume.Unmounter, error) {
	return &vsphereVolumeUnmounter{
		&vsphereVolume{
			podUID:          podUID,
			volName:         volName,
			manager:         manager,
			mounter:         mounter,
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, volName, plugin.host)),
		}}, nil
}

func (plugin *vsphereVolumePlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	mounter := plugin.host.GetMounter(plugin.GetPluginName())
	kvh, ok := plugin.host.(volume.KubeletVolumeHost)
	if !ok {
		return nil, fmt.Errorf("plugin volume host does not implement KubeletVolumeHost interface")
	}
	hu := kvh.GetHostUtil()
	pluginMntDir := util.GetPluginMountDir(plugin.host, plugin.GetPluginName())
	volumePath, err := hu.GetDeviceNameFromMount(mounter, mountPath, pluginMntDir)
	if err != nil {
		return nil, err
	}
	volumePath = strings.Replace(volumePath, "\\040", " ", -1)
	klog.V(5).Infof("vSphere volume path is %q", volumePath)
	vsphereVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
				VolumePath: volumePath,
			},
		},
	}
	return volume.NewSpecFromVolume(vsphereVolume), nil
}

// Abstract interface to disk operations.
type vdManager interface {
	// Creates a volume
	CreateVolume(provisioner *vsphereVolumeProvisioner, selectedNode *v1.Node, selectedZone []string) (volSpec *VolumeSpec, err error)
	// Deletes a volume
	DeleteVolume(deleter *vsphereVolumeDeleter) error
}

// vspherePersistentDisk volumes are disk resources are attached to the kubelet's host machine and exposed to the pod.
type vsphereVolume struct {
	volName string
	podUID  types.UID
	// Unique identifier of the volume, used to find the disk resource in the provider.
	volPath string
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager vdManager
	// Mounter interface that provides system calls to mount the global path to the pod local path.
	mounter mount.Interface
	plugin  *vsphereVolumePlugin
	volume.MetricsProvider
}

var _ volume.Mounter = &vsphereVolumeMounter{}

type vsphereVolumeMounter struct {
	*vsphereVolume
	fsType       string
	diskMounter  *mount.SafeFormatAndMount
	mountOptions []string
}

func (b *vsphereVolumeMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		SELinuxRelabel: true,
		Managed:        true,
	}
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *vsphereVolumeMounter) SetUp(mounterArgs volume.MounterArgs) error {
	return b.SetUpAt(b.GetPath(), mounterArgs)
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *vsphereVolumeMounter) SetUpAt(dir string, mounterArgs volume.MounterArgs) error {
	klog.V(5).Infof("vSphere volume setup %s to %s", b.volPath, dir)

	// TODO: handle failed mounts here.
	notmnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	if err != nil && !os.IsNotExist(err) {
		klog.V(4).Infof("IsLikelyNotMountPoint failed: %v", err)
		return err
	}
	if !notmnt {
		klog.V(4).Infof("Something is already mounted to target %s", dir)
		return nil
	}

	if runtime.GOOS != "windows" {
		// On Windows, Mount will create the parent of dir and mklink (create a symbolic link) at dir later, so don't create a
		// directory at dir now. Otherwise mklink will error: "Cannot create a file when that file already exists".
		if err := os.MkdirAll(dir, 0750); err != nil {
			klog.Errorf("Could not create directory %s: %v", dir, err)
			return err
		}
	}

	options := []string{"bind"}

	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	globalPDPath := makeGlobalPDPath(b.plugin.host, b.volPath)
	mountOptions := util.JoinMountOptions(options, b.mountOptions)
	err = b.mounter.MountSensitiveWithoutSystemd(globalPDPath, dir, "", mountOptions, nil)
	if err != nil {
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
				klog.Errorf("%s is still mounted, despite call to unmount().  Will try again next sync loop.", b.GetPath())
				return err
			}
		}
		os.Remove(dir)
		return err
	}
	volume.SetVolumeOwnership(b, mounterArgs.FsGroup, mounterArgs.FSGroupChangePolicy, util.FSGroupCompleteHook(b.plugin, nil))
	klog.V(3).Infof("vSphere volume %s mounted to %s", b.volPath, dir)

	return nil
}

var _ volume.Unmounter = &vsphereVolumeUnmounter{}

type vsphereVolumeUnmounter struct {
	*vsphereVolume
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (v *vsphereVolumeUnmounter) TearDown() error {
	return v.TearDownAt(v.GetPath())
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (v *vsphereVolumeUnmounter) TearDownAt(dir string) error {
	return mount.CleanupMountPoint(dir, v.mounter, false)
}

func makeGlobalPDPath(host volume.VolumeHost, devName string) string {
	return filepath.Join(host.GetPluginDir(vsphereVolumePluginName), util.MountsInGlobalPDPath, devName)
}

func (vv *vsphereVolume) GetPath() string {
	name := vsphereVolumePluginName
	return vv.plugin.host.GetPodVolumeDir(vv.podUID, utilstrings.EscapeQualifiedName(name), vv.volName)
}

// vSphere Persistent Volume Plugin
func (plugin *vsphereVolumePlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
	}
}

// vSphere Deletable Volume Plugin
type vsphereVolumeDeleter struct {
	*vsphereVolume
}

var _ volume.Deleter = &vsphereVolumeDeleter{}

func (plugin *vsphereVolumePlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	return plugin.newDeleterInternal(spec, &VsphereDiskUtil{})
}

func (plugin *vsphereVolumePlugin) newDeleterInternal(spec *volume.Spec, manager vdManager) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.VsphereVolume == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.VsphereVolume is nil")
	}
	return &vsphereVolumeDeleter{
		&vsphereVolume{
			volName: spec.Name(),
			volPath: spec.PersistentVolume.Spec.VsphereVolume.VolumePath,
			manager: manager,
			plugin:  plugin,
		}}, nil
}

func (r *vsphereVolumeDeleter) Delete() error {
	return r.manager.DeleteVolume(r)
}

// vSphere Provisionable Volume Plugin
type vsphereVolumeProvisioner struct {
	*vsphereVolume
	options volume.VolumeOptions
}

var _ volume.Provisioner = &vsphereVolumeProvisioner{}

func (plugin *vsphereVolumePlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	return plugin.newProvisionerInternal(options, &VsphereDiskUtil{})
}

func (plugin *vsphereVolumePlugin) newProvisionerInternal(options volume.VolumeOptions, manager vdManager) (volume.Provisioner, error) {
	return &vsphereVolumeProvisioner{
		vsphereVolume: &vsphereVolume{
			manager: manager,
			plugin:  plugin,
		},
		options: options,
	}, nil
}

func (v *vsphereVolumeProvisioner) Provision(selectedNode *v1.Node, allowedTopologies []v1.TopologySelectorTerm) (*v1.PersistentVolume, error) {
	if !util.ContainsAllAccessModes(v.plugin.GetAccessModes(), v.options.PVC.Spec.AccessModes) {
		return nil, fmt.Errorf("invalid AccessModes %v: only AccessModes %v are supported", v.options.PVC.Spec.AccessModes, v.plugin.GetAccessModes())
	}
	klog.V(1).Infof("Provision with selectedNode: %s and allowedTopologies : %s", getNodeName(selectedNode), allowedTopologies)
	selectedZones, err := volumehelpers.ZonesFromAllowedTopologies(allowedTopologies)
	if err != nil {
		return nil, err
	}

	klog.V(4).Infof("Selected zones for volume : %s", selectedZones)
	volSpec, err := v.manager.CreateVolume(v, selectedNode, selectedZones.List())
	if err != nil {
		return nil, err
	}

	if volSpec.Fstype == "" {
		volSpec.Fstype = "ext4"
	}

	volumeMode := v.options.PVC.Spec.VolumeMode
	if volumeMode != nil && *volumeMode == v1.PersistentVolumeBlock {
		klog.V(5).Infof("vSphere block volume should not have any FSType")
		volSpec.Fstype = ""
	}

	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:   v.options.PVName,
			Labels: map[string]string{},
			Annotations: map[string]string{
				util.VolumeDynamicallyCreatedByKey: "vsphere-volume-dynamic-provisioner",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: v.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   v.options.PVC.Spec.AccessModes,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse(fmt.Sprintf("%dKi", volSpec.Size)),
			},
			VolumeMode: volumeMode,
			PersistentVolumeSource: v1.PersistentVolumeSource{
				VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
					VolumePath:        volSpec.Path,
					FSType:            volSpec.Fstype,
					StoragePolicyName: volSpec.StoragePolicyName,
					StoragePolicyID:   volSpec.StoragePolicyID,
				},
			},
			MountOptions: v.options.MountOptions,
		},
	}
	if len(v.options.PVC.Spec.AccessModes) == 0 {
		pv.Spec.AccessModes = v.plugin.GetAccessModes()
	}

	labels := volSpec.Labels
	requirements := make([]v1.NodeSelectorRequirement, 0)
	if len(labels) != 0 {
		if pv.Labels == nil {
			pv.Labels = make(map[string]string)
		}
		for k, v := range labels {
			pv.Labels[k] = v
			var values []string
			if k == v1.LabelTopologyZone || k == v1.LabelFailureDomainBetaZone {
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

func getVolumeSource(
	spec *volume.Spec) (*v1.VsphereVirtualDiskVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.VsphereVolume != nil {
		return spec.Volume.VsphereVolume, spec.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.VsphereVolume != nil {
		return spec.PersistentVolume.Spec.VsphereVolume, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("spec does not reference a VSphere volume type")
}

func getNodeName(node *v1.Node) string {
	if node == nil {
		return ""
	}
	return node.Name
}
