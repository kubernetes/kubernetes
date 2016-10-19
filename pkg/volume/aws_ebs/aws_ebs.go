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

package aws_ebs

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	kstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&awsElasticBlockStorePlugin{nil}}
}

type awsElasticBlockStorePlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &awsElasticBlockStorePlugin{}
var _ volume.PersistentVolumePlugin = &awsElasticBlockStorePlugin{}
var _ volume.DeletableVolumePlugin = &awsElasticBlockStorePlugin{}
var _ volume.ProvisionableVolumePlugin = &awsElasticBlockStorePlugin{}

const (
	awsElasticBlockStorePluginName = "kubernetes.io/aws-ebs"
)

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, kstrings.EscapeQualifiedNameForDisk(awsElasticBlockStorePluginName), volName)
}

func (plugin *awsElasticBlockStorePlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *awsElasticBlockStorePlugin) GetPluginName() string {
	return awsElasticBlockStorePluginName
}

func (plugin *awsElasticBlockStorePlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return volumeSource.VolumeID, nil
}

func (plugin *awsElasticBlockStorePlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.AWSElasticBlockStore != nil) ||
		(spec.Volume != nil && spec.Volume.AWSElasticBlockStore != nil)
}

func (plugin *awsElasticBlockStorePlugin) RequiresRemount() bool {
	return false
}

func (plugin *awsElasticBlockStorePlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
	}
}

func (plugin *awsElasticBlockStorePlugin) NewMounter(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newMounterInternal(spec, pod.UID, &AWSDiskUtil{}, plugin.host.GetMounter())
}

func (plugin *awsElasticBlockStorePlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, manager ebsManager, mounter mount.Interface) (volume.Mounter, error) {
	// EBSs used directly in a pod have a ReadOnly flag set by the pod author.
	// EBSs used as a PersistentVolume gets the ReadOnly flag indirectly through the persistent-claim volume used to mount the PV
	ebs, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	volumeID := ebs.VolumeID
	fsType := ebs.FSType
	partition := ""
	if ebs.Partition != 0 {
		partition = strconv.Itoa(int(ebs.Partition))
	}

	return &awsElasticBlockStoreMounter{
		awsElasticBlockStore: &awsElasticBlockStore{
			podUID:          podUID,
			volName:         spec.Name(),
			volumeID:        volumeID,
			partition:       partition,
			manager:         manager,
			mounter:         mounter,
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, spec.Name(), plugin.host)),
		},
		fsType:      fsType,
		readOnly:    readOnly,
		diskMounter: &mount.SafeFormatAndMount{Interface: plugin.host.GetMounter(), Runner: exec.New()}}, nil
}

func (plugin *awsElasticBlockStorePlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newUnmounterInternal(volName, podUID, &AWSDiskUtil{}, plugin.host.GetMounter())
}

func (plugin *awsElasticBlockStorePlugin) newUnmounterInternal(volName string, podUID types.UID, manager ebsManager, mounter mount.Interface) (volume.Unmounter, error) {
	return &awsElasticBlockStoreUnmounter{&awsElasticBlockStore{
		podUID:          podUID,
		volName:         volName,
		manager:         manager,
		mounter:         mounter,
		plugin:          plugin,
		MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, volName, plugin.host)),
	}}, nil
}

func (plugin *awsElasticBlockStorePlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	return plugin.newDeleterInternal(spec, &AWSDiskUtil{})
}

func (plugin *awsElasticBlockStorePlugin) newDeleterInternal(spec *volume.Spec, manager ebsManager) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.AWSElasticBlockStore == nil {
		glog.Errorf("spec.PersistentVolumeSource.AWSElasticBlockStore is nil")
		return nil, fmt.Errorf("spec.PersistentVolumeSource.AWSElasticBlockStore is nil")
	}
	return &awsElasticBlockStoreDeleter{
		awsElasticBlockStore: &awsElasticBlockStore{
			volName:  spec.Name(),
			volumeID: spec.PersistentVolume.Spec.AWSElasticBlockStore.VolumeID,
			manager:  manager,
			plugin:   plugin,
		}}, nil
}

func (plugin *awsElasticBlockStorePlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	return plugin.newProvisionerInternal(options, &AWSDiskUtil{})
}

func (plugin *awsElasticBlockStorePlugin) newProvisionerInternal(options volume.VolumeOptions, manager ebsManager) (volume.Provisioner, error) {
	return &awsElasticBlockStoreProvisioner{
		awsElasticBlockStore: &awsElasticBlockStore{
			manager: manager,
			plugin:  plugin,
		},
		options: options,
	}, nil
}

func getVolumeSource(
	spec *volume.Spec) (*api.AWSElasticBlockStoreVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.AWSElasticBlockStore != nil {
		return spec.Volume.AWSElasticBlockStore, spec.Volume.AWSElasticBlockStore.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.AWSElasticBlockStore != nil {
		return spec.PersistentVolume.Spec.AWSElasticBlockStore, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference an AWS EBS volume type")
}

func (plugin *awsElasticBlockStorePlugin) ConstructVolumeSpec(volName, mountPath string) (*volume.Spec, error) {
	mounter := plugin.host.GetMounter()
	pluginDir := plugin.host.GetPluginDir(plugin.GetPluginName())
	sourceName, err := mounter.GetDeviceNameFromMount(mountPath, pluginDir)
	if err != nil {
		return nil, err
	}
	awsVolume := &api.Volume{
		Name: volName,
		VolumeSource: api.VolumeSource{
			AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
				VolumeID: sourceName,
			},
		},
	}
	return volume.NewSpecFromVolume(awsVolume), nil
}

// Abstract interface to PD operations.
type ebsManager interface {
	CreateVolume(provisioner *awsElasticBlockStoreProvisioner) (volumeID string, volumeSizeGB int, labels map[string]string, err error)
	// Deletes a volume
	DeleteVolume(deleter *awsElasticBlockStoreDeleter) error
}

// awsElasticBlockStore volumes are disk resources provided by Amazon Web Services
// that are attached to the kubelet's host machine and exposed to the pod.
type awsElasticBlockStore struct {
	volName string
	podUID  types.UID
	// Unique id of the PD, used to find the disk resource in the provider.
	volumeID string
	// Specifies the partition to mount
	partition string
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager ebsManager
	// Mounter interface that provides system calls to mount the global path to the pod local path.
	mounter mount.Interface
	plugin  *awsElasticBlockStorePlugin
	volume.MetricsProvider
}

type awsElasticBlockStoreMounter struct {
	*awsElasticBlockStore
	// Filesystem type, optional.
	fsType string
	// Specifies whether the disk will be attached as read-only.
	readOnly bool
	// diskMounter provides the interface that is used to mount the actual block device.
	diskMounter *mount.SafeFormatAndMount
}

var _ volume.Mounter = &awsElasticBlockStoreMounter{}

func (b *awsElasticBlockStoreMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         !b.readOnly,
		SupportsSELinux: true,
	}
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *awsElasticBlockStoreMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

// SetUpAt attaches the disk and bind mounts to the volume path.
func (b *awsElasticBlockStoreMounter) SetUpAt(dir string, fsGroup *int64) error {
	// TODO: handle failed mounts here.
	notMnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	glog.V(4).Infof("PersistentDisk set up: %s %v %v", dir, !notMnt, err)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("cannot validate mount point: %s %v", dir, err)
		return err
	}
	if !notMnt {
		return nil
	}

	globalPDPath := makeGlobalPDPath(b.plugin.host, b.volumeID)

	if err := os.MkdirAll(dir, 0750); err != nil {
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
			glog.Errorf("IsLikelyNotMountPoint check failed for %s: %v", dir, mntErr)
			return err
		}
		if !notMnt {
			if mntErr = b.mounter.Unmount(dir); mntErr != nil {
				glog.Errorf("failed to unmount %s: %v", dir, mntErr)
				return err
			}
			notMnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
			if mntErr != nil {
				glog.Errorf("IsLikelyNotMountPoint check failed for %s: %v", dir, mntErr)
				return err
			}
			if !notMnt {
				// This is very odd, we don't expect it.  We'll try again next sync loop.
				glog.Errorf("%s is still mounted, despite call to unmount().  Will try again next sync loop.", dir)
				return err
			}
		}
		os.Remove(dir)
		glog.Errorf("Mount of disk %s failed: %v", dir, err)
		return err
	}

	if !b.readOnly {
		volume.SetVolumeOwnership(b, fsGroup)
	}

	glog.V(4).Infof("Successfully mounted %s", dir)
	return nil
}

func makeGlobalPDPath(host volume.VolumeHost, volumeID string) string {
	// Clean up the URI to be more fs-friendly
	name := volumeID
	name = strings.Replace(name, "://", "/", -1)
	return path.Join(host.GetPluginDir(awsElasticBlockStorePluginName), "mounts", name)
}

// Reverses the mapping done in makeGlobalPDPath
func getVolumeIDFromGlobalMount(host volume.VolumeHost, globalPath string) (string, error) {
	basePath := path.Join(host.GetPluginDir(awsElasticBlockStorePluginName), "mounts")
	rel, err := filepath.Rel(basePath, globalPath)
	if err != nil {
		glog.Errorf("Failed to get volume id from global mount %s - %v", globalPath, err)
		return "", err
	}
	if strings.Contains(rel, "../") {
		glog.Errorf("Unexpected mount path: %s", globalPath)
		return "", fmt.Errorf("unexpected mount path: " + globalPath)
	}
	// Reverse the :// replacement done in makeGlobalPDPath
	volumeID := rel
	if strings.HasPrefix(volumeID, "aws/") {
		volumeID = strings.Replace(volumeID, "aws/", "aws://", 1)
	}
	glog.V(2).Info("Mapping mount dir ", globalPath, " to volumeID ", volumeID)
	return volumeID, nil
}

func (ebs *awsElasticBlockStore) GetPath() string {
	return getPath(ebs.podUID, ebs.volName, ebs.plugin.host)
}

type awsElasticBlockStoreUnmounter struct {
	*awsElasticBlockStore
}

var _ volume.Unmounter = &awsElasticBlockStoreUnmounter{}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (c *awsElasticBlockStoreUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

// Unmounts the bind mount
func (c *awsElasticBlockStoreUnmounter) TearDownAt(dir string) error {
	notMnt, err := c.mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		glog.V(2).Info("Error checking if mountpoint ", dir, ": ", err)
		return err
	}
	if notMnt {
		glog.V(2).Info("Not mountpoint, deleting")
		return os.Remove(dir)
	}

	// Unmount the bind-mount inside this pod
	if err := c.mounter.Unmount(dir); err != nil {
		glog.V(2).Info("Error unmounting dir ", dir, ": ", err)
		return err
	}
	notMnt, mntErr := c.mounter.IsLikelyNotMountPoint(dir)
	if mntErr != nil {
		glog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
		return err
	}
	if notMnt {
		if err := os.Remove(dir); err != nil {
			glog.V(2).Info("Error removing mountpoint ", dir, ": ", err)
			return err
		}
	}
	return nil
}

type awsElasticBlockStoreDeleter struct {
	*awsElasticBlockStore
}

var _ volume.Deleter = &awsElasticBlockStoreDeleter{}

func (d *awsElasticBlockStoreDeleter) GetPath() string {
	return getPath(d.podUID, d.volName, d.plugin.host)
}

func (d *awsElasticBlockStoreDeleter) Delete() error {
	return d.manager.DeleteVolume(d)
}

type awsElasticBlockStoreProvisioner struct {
	*awsElasticBlockStore
	options   volume.VolumeOptions
	namespace string
}

var _ volume.Provisioner = &awsElasticBlockStoreProvisioner{}

func (c *awsElasticBlockStoreProvisioner) Provision() (*api.PersistentVolume, error) {
	volumeID, sizeGB, labels, err := c.manager.CreateVolume(c)
	if err != nil {
		glog.Errorf("Provision failed: %v", err)
		return nil, err
	}

	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name:   c.options.PVName,
			Labels: map[string]string{},
			Annotations: map[string]string{
				"kubernetes.io/createdby": "aws-ebs-dynamic-provisioner",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: c.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   c.options.PVC.Spec.AccessModes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGB)),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
					VolumeID:  volumeID,
					FSType:    "ext4",
					Partition: 0,
					ReadOnly:  false,
				},
			},
		},
	}

	if len(c.options.PVC.Spec.AccessModes) == 0 {
		pv.Spec.AccessModes = c.plugin.GetAccessModes()
	}

	if len(labels) != 0 {
		if pv.Labels == nil {
			pv.Labels = make(map[string]string)
		}
		for k, v := range labels {
			pv.Labels[k] = v
		}
	}

	return pv, nil
}
