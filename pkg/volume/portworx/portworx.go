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

package portworx

import (
	"fmt"
	"os"

	volumeclient "github.com/libopenstorage/openstorage/api/client/volume"
	"k8s.io/klog"
	"k8s.io/utils/mount"
	utilstrings "k8s.io/utils/strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

const (
	attachContextKey = "context"
	attachHostKey    = "host"
)

// ProbeVolumePlugins is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&portworxVolumePlugin{nil, nil}}
}

type portworxVolumePlugin struct {
	host volume.VolumeHost
	util *portworxVolumeUtil
}

var _ volume.VolumePlugin = &portworxVolumePlugin{}
var _ volume.PersistentVolumePlugin = &portworxVolumePlugin{}
var _ volume.DeletableVolumePlugin = &portworxVolumePlugin{}
var _ volume.ProvisionableVolumePlugin = &portworxVolumePlugin{}
var _ volume.ExpandableVolumePlugin = &portworxVolumePlugin{}

const (
	portworxVolumePluginName = "kubernetes.io/portworx-volume"
)

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, utilstrings.EscapeQualifiedName(portworxVolumePluginName), volName)
}

func (plugin *portworxVolumePlugin) Init(host volume.VolumeHost) error {
	client, err := volumeclient.NewDriverClient(
		fmt.Sprintf("http://%s:%d", host.GetHostName(), osdMgmtDefaultPort),
		pxdDriverName, osdDriverVersion, pxDriverName)
	if err != nil {
		return err
	}

	plugin.host = host
	plugin.util = &portworxVolumeUtil{
		portworxClient: client,
	}

	return nil
}

func (plugin *portworxVolumePlugin) GetPluginName() string {
	return portworxVolumePluginName
}

func (plugin *portworxVolumePlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return volumeSource.VolumeID, nil
}

func (plugin *portworxVolumePlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.PortworxVolume != nil) ||
		(spec.Volume != nil && spec.Volume.PortworxVolume != nil)
}

func (plugin *portworxVolumePlugin) RequiresRemount() bool {
	return false
}

func (plugin *portworxVolumePlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
		v1.ReadWriteMany,
	}
}

func (plugin *portworxVolumePlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	return plugin.newMounterInternal(spec, pod.UID, plugin.util, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *portworxVolumePlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, manager portworxManager, mounter mount.Interface) (volume.Mounter, error) {
	pwx, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	volumeID := pwx.VolumeID
	fsType := pwx.FSType

	return &portworxVolumeMounter{
		portworxVolume: &portworxVolume{
			podUID:          podUID,
			volName:         spec.Name(),
			volumeID:        volumeID,
			manager:         manager,
			mounter:         mounter,
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, spec.Name(), plugin.host)),
		},
		fsType:      fsType,
		readOnly:    readOnly,
		diskMounter: util.NewSafeFormatAndMountFromHost(plugin.GetPluginName(), plugin.host)}, nil
}

func (plugin *portworxVolumePlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, plugin.util, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *portworxVolumePlugin) newUnmounterInternal(volName string, podUID types.UID, manager portworxManager,
	mounter mount.Interface) (volume.Unmounter, error) {
	return &portworxVolumeUnmounter{
		&portworxVolume{
			podUID:          podUID,
			volName:         volName,
			manager:         manager,
			mounter:         mounter,
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, volName, plugin.host)),
		}}, nil
}

func (plugin *portworxVolumePlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	return plugin.newDeleterInternal(spec, plugin.util)
}

func (plugin *portworxVolumePlugin) newDeleterInternal(spec *volume.Spec, manager portworxManager) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.PortworxVolume == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.PortworxVolume is nil")
	}

	return &portworxVolumeDeleter{
		portworxVolume: &portworxVolume{
			volName:  spec.Name(),
			volumeID: spec.PersistentVolume.Spec.PortworxVolume.VolumeID,
			manager:  manager,
			plugin:   plugin,
		}}, nil
}

func (plugin *portworxVolumePlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	return plugin.newProvisionerInternal(options, plugin.util)
}

func (plugin *portworxVolumePlugin) newProvisionerInternal(options volume.VolumeOptions, manager portworxManager) (volume.Provisioner, error) {
	return &portworxVolumeProvisioner{
		portworxVolume: &portworxVolume{
			manager: manager,
			plugin:  plugin,
		},
		options: options,
	}, nil
}

func (plugin *portworxVolumePlugin) RequiresFSResize() bool {
	return false
}

func (plugin *portworxVolumePlugin) ExpandVolumeDevice(
	spec *volume.Spec,
	newSize resource.Quantity,
	oldSize resource.Quantity) (resource.Quantity, error) {
	klog.V(4).Infof("Expanding: %s from %v to %v", spec.Name(), oldSize, newSize)
	err := plugin.util.ResizeVolume(spec, newSize, plugin.host)
	if err != nil {
		return oldSize, err
	}

	klog.V(4).Infof("Successfully resized %s to %v", spec.Name(), newSize)
	return newSize, nil
}

func (plugin *portworxVolumePlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	portworxVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			PortworxVolume: &v1.PortworxVolumeSource{
				VolumeID: volumeName,
			},
		},
	}
	return volume.NewSpecFromVolume(portworxVolume), nil
}

func (plugin *portworxVolumePlugin) SupportsMountOption() bool {
	return false
}

func (plugin *portworxVolumePlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func getVolumeSource(
	spec *volume.Spec) (*v1.PortworxVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.PortworxVolume != nil {
		return spec.Volume.PortworxVolume, spec.Volume.PortworxVolume.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.PortworxVolume != nil {
		return spec.PersistentVolume.Spec.PortworxVolume, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a Portworx Volume type")
}

// Abstract interface to PD operations.
type portworxManager interface {
	// Creates a volume
	CreateVolume(provisioner *portworxVolumeProvisioner) (volumeID string, volumeSizeGB int64, labels map[string]string, err error)
	// Deletes a volume
	DeleteVolume(deleter *portworxVolumeDeleter) error
	// Attach a volume
	AttachVolume(mounter *portworxVolumeMounter, attachOptions map[string]string) (string, error)
	// Detach a volume
	DetachVolume(unmounter *portworxVolumeUnmounter) error
	// Mount a volume
	MountVolume(mounter *portworxVolumeMounter, mountDir string) error
	// Unmount a volume
	UnmountVolume(unmounter *portworxVolumeUnmounter, mountDir string) error
	// Resize a volume
	ResizeVolume(spec *volume.Spec, newSize resource.Quantity, host volume.VolumeHost) error
}

// portworxVolume volumes are portworx block devices
// that are attached to the kubelet's host machine and exposed to the pod.
type portworxVolume struct {
	volName string
	podUID  types.UID
	// Unique id of the PD, used to find the disk resource in the provider.
	volumeID string
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager portworxManager
	// Mounter interface that provides system calls to mount the global path to the pod local path.
	mounter mount.Interface
	plugin  *portworxVolumePlugin
	volume.MetricsProvider
}

type portworxVolumeMounter struct {
	*portworxVolume
	// Filesystem type, optional.
	fsType string
	// Specifies whether the disk will be attached as read-only.
	readOnly bool
	// diskMounter provides the interface that is used to mount the actual block device.
	diskMounter *mount.SafeFormatAndMount
}

var _ volume.Mounter = &portworxVolumeMounter{}

func (b *portworxVolumeMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         !b.readOnly,
		SupportsSELinux: false,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *portworxVolumeMounter) CanMount() error {
	return nil
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *portworxVolumeMounter) SetUp(mounterArgs volume.MounterArgs) error {
	return b.SetUpAt(b.GetPath(), mounterArgs)
}

// SetUpAt attaches the disk and bind mounts to the volume path.
func (b *portworxVolumeMounter) SetUpAt(dir string, mounterArgs volume.MounterArgs) error {
	notMnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	klog.Infof("Portworx Volume set up. Dir: %s %v %v", dir, !notMnt, err)
	if err != nil && !os.IsNotExist(err) {
		klog.Errorf("Cannot validate mountpoint: %s", dir)
		return err
	}
	if !notMnt {
		return nil
	}

	attachOptions := make(map[string]string)
	attachOptions[attachContextKey] = dir
	attachOptions[attachHostKey] = b.plugin.host.GetHostName()
	if _, err := b.manager.AttachVolume(b, attachOptions); err != nil {
		return err
	}

	klog.V(4).Infof("Portworx Volume %s attached", b.volumeID)

	if err := os.MkdirAll(dir, 0750); err != nil {
		return err
	}

	if err := b.manager.MountVolume(b, dir); err != nil {
		return err
	}
	if !b.readOnly {
		volume.SetVolumeOwnership(b, mounterArgs.FsGroup)
	}
	klog.Infof("Portworx Volume %s setup at %s", b.volumeID, dir)
	return nil
}

func (pwx *portworxVolume) GetPath() string {
	return getPath(pwx.podUID, pwx.volName, pwx.plugin.host)
}

type portworxVolumeUnmounter struct {
	*portworxVolume
}

var _ volume.Unmounter = &portworxVolumeUnmounter{}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (c *portworxVolumeUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (c *portworxVolumeUnmounter) TearDownAt(dir string) error {
	klog.Infof("Portworx Volume TearDown of %s", dir)

	if err := c.manager.UnmountVolume(c, dir); err != nil {
		return err
	}

	// Call Portworx Detach Volume.
	if err := c.manager.DetachVolume(c); err != nil {
		return err
	}

	return nil
}

type portworxVolumeDeleter struct {
	*portworxVolume
}

var _ volume.Deleter = &portworxVolumeDeleter{}

func (d *portworxVolumeDeleter) GetPath() string {
	return getPath(d.podUID, d.volName, d.plugin.host)
}

func (d *portworxVolumeDeleter) Delete() error {
	return d.manager.DeleteVolume(d)
}

type portworxVolumeProvisioner struct {
	*portworxVolume
	options   volume.VolumeOptions
	namespace string
}

var _ volume.Provisioner = &portworxVolumeProvisioner{}

func (c *portworxVolumeProvisioner) Provision(selectedNode *v1.Node, allowedTopologies []v1.TopologySelectorTerm) (*v1.PersistentVolume, error) {
	if !util.AccessModesContainedInAll(c.plugin.GetAccessModes(), c.options.PVC.Spec.AccessModes) {
		return nil, fmt.Errorf("invalid AccessModes %v: only AccessModes %v are supported", c.options.PVC.Spec.AccessModes, c.plugin.GetAccessModes())
	}

	if util.CheckPersistentVolumeClaimModeBlock(c.options.PVC) {
		return nil, fmt.Errorf("%s does not support block volume provisioning", c.plugin.GetPluginName())
	}

	volumeID, sizeGiB, labels, err := c.manager.CreateVolume(c)
	if err != nil {
		return nil, err
	}

	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:   c.options.PVName,
			Labels: map[string]string{},
			Annotations: map[string]string{
				util.VolumeDynamicallyCreatedByKey: "portworx-volume-dynamic-provisioner",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: c.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   c.options.PVC.Spec.AccessModes,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGiB)),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				PortworxVolume: &v1.PortworxVolumeSource{
					VolumeID: volumeID,
				},
			},
		},
	}

	if len(labels) != 0 {
		if pv.Labels == nil {
			pv.Labels = make(map[string]string)
		}
		for k, v := range labels {
			pv.Labels[k] = v
		}
	}

	if len(c.options.PVC.Spec.AccessModes) == 0 {
		pv.Spec.AccessModes = c.plugin.GetAccessModes()
	}

	return pv, nil
}
