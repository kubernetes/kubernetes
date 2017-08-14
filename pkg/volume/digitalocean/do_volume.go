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

package digitalocean

import (
	"fmt"
	"os"
	"path"
	"strings"

	"github.com/digitalocean/godo"
	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	"k8s.io/kubernetes/pkg/util/mount"
	kstrings "k8s.io/kubernetes/pkg/util/strings"

	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

const (
	volumeNameMaxLength = 64
)

type doVolume struct {
	volName  string
	podUID   types.UID
	volumeID string
	mounter  mount.Interface
	plugin   *doVolumePlugin
	manager  volManager
	volume.MetricsProvider
}

type volManager interface {
	DeleteVolume(volumeID string) error
	CreateVolume(name, description string, sizeGB int) (string, error)
	FindDropletForNode(node *v1.Node) (*godo.Droplet, error)
	AttachVolume(volumeID string, dropletID int) (string, error)
	GetDroplet(dropletID int) (*godo.Droplet, error)
	DisksAreAttached(volumeIDs []string, dropletID int) (map[string]bool, error)
	DetachVolume(volumeID string, dropletID int) error
}

var _ volume.Volume = &doVolume{}

func (v *doVolume) GetPath() string {
	return v.plugin.host.GetPodVolumeDir(
		v.podUID, kstrings.EscapeQualifiedNameForDisk(doVolumePluginName), v.volName)
}

type doVolumeMounter struct {
	*doVolume
	// Filesystem type, optional.
	fsType string
	// Specifies whether the disk will be attached as read-only.
	readOnly bool
	// diskMounter provides the interface that is used to mount the actual volume device.
	diskMounter *mount.SafeFormatAndMount
}

var _ volume.Mounter = &doVolumeMounter{}

// GetAttributes returns the attributes of the mounter
func (vm *doVolumeMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        vm.readOnly,
		Managed:         !vm.readOnly,
		SupportsSELinux: true,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (vm *doVolumeMounter) CanMount() error {
	return nil
}

// SetUp attaches the disk and bind mounts to the volume path
func (vm *doVolumeMounter) SetUp(fsGroup *int64) error {
	return vm.SetUpAt(vm.GetPath(), fsGroup)
}

// SetUpAt attaches the disk and bind mounts to the volume path.
func (vm *doVolumeMounter) SetUpAt(dir string, fsGroup *int64) error {

	notMnt, err := vm.mounter.IsLikelyNotMountPoint(dir)
	glog.V(4).Infof("Digital Ocean volume set up: %s %v %v", dir, !notMnt, err)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("IsLikelyNotMountPoint failed validating mount point: %s %v", dir, err)
		return err
	}
	if !notMnt {
		return nil
	}

	globalPDPath := makeGlobalPDPath(vm.plugin.host, vm.volumeID)

	if err = os.MkdirAll(dir, 0750); err != nil {
		glog.Errorf("failed creating mount point %q:  %v", dir, err)
		return err
	}

	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	options := []string{"bind"}
	if vm.readOnly {
		options = append(options, "ro")
	}
	err = vm.mounter.Mount(globalPDPath, dir, "", options)
	if err != nil {
		notMnt, mntErr := vm.mounter.IsLikelyNotMountPoint(dir)
		if mntErr != nil {
			glog.Errorf("IsLikelyNotMountPoint failed validating mount point %s: %v", dir, mntErr)
			return err
		}
		if !notMnt {
			if mntErr = vm.mounter.Unmount(dir); mntErr != nil {
				glog.Errorf("failed to unmount %s: %v", dir, mntErr)
				return err
			}
			notMnt, mntErr := vm.mounter.IsLikelyNotMountPoint(dir)
			if mntErr != nil {
				glog.Errorf("IsLikelyNotMountPoint failed validating mount point %s: %v", dir, mntErr)
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

	if !vm.readOnly {
		volume.SetVolumeOwnership(vm, fsGroup)
	}

	glog.V(4).Infof("Digital Ocean volume %s successfully mounted to %s", vm.volumeID, dir)
	return nil
}

type doVolumeUnmounter struct {
	*doVolume
}

var _ volume.Unmounter = &doVolumeUnmounter{}

// TearDown unmounts the bind mount
func (vu *doVolumeUnmounter) TearDown() error {
	return vu.TearDownAt(vu.GetPath())
}

// TearDownAt unmounts the volume from the specified directory and
// removes traces of the SetUp procedure.
func (vu *doVolumeUnmounter) TearDownAt(dir string) error {
	return util.UnmountPath(dir, vu.mounter)
}

type doVolumeDeleter struct {
	*doVolume
}

var _ volume.Deleter = &doVolumeDeleter{}

func (vd *doVolumeDeleter) GetPath() string {
	return vd.plugin.host.GetPodVolumeDir(
		vd.podUID,
		kstrings.EscapeQualifiedNameForDisk(doVolumePluginName),
		vd.volName)
}

func (vd *doVolumeDeleter) Delete() error {
	err := vd.manager.DeleteVolume(vd.volumeID)
	if err != nil {
		glog.V(2).Infof("Error deleting Digital Ocean volume %s: %v", vd.volumeID, err)
		return err
	}
	return nil
}

type doVolumeProvisioner struct {
	*doVolume
	options volume.VolumeOptions
}

var _ volume.Provisioner = &doVolumeProvisioner{}

// Provision creates the resource at Digital Ocean and waits for it to be available
func (vp *doVolumeProvisioner) Provision() (*v1.PersistentVolume, error) {
	if !volume.AccessModesContainedInAll(vp.plugin.GetAccessModes(), vp.options.PVC.Spec.AccessModes) {
		return nil, fmt.Errorf("invalid AccessModes %v: only AccessModes %v are supported",
			vp.options.PVC.Spec.AccessModes, vp.plugin.GetAccessModes())
	}

	name := strings.ToLower(volume.GenerateVolumeName(vp.options.ClusterName, vp.options.PVName, volumeNameMaxLength))
	description := fmt.Sprintf("kubernetes volume for cluster %s", vp.options.ClusterName)
	capacity := vp.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	sizeGB := int(volume.RoundUpSize(capacity.Value(), 1024*1024*1024))

	volumeID, err := vp.manager.CreateVolume(name, description, sizeGB)
	if err != nil {
		glog.V(2).Infof("Error creating Digital Ocean volume: %v", err)
		return nil, err
	}

	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:   vp.options.PVName,
			Labels: map[string]string{},
			Annotations: map[string]string{
				"kubernetes.io/createdby": "do-dynamic-provisioner",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: vp.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   vp.options.PVC.Spec.AccessModes,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGB)),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				DOVolume: &v1.DOVolumeSource{
					VolumeID: volumeID,
					FSType:   "ext4",
					ReadOnly: false,
				},
			},
		},
	}

	if len(vp.options.PVC.Spec.AccessModes) == 0 {
		pv.Spec.AccessModes = vp.plugin.GetAccessModes()
	}

	return pv, nil
}

func makeGlobalPDPath(host volume.VolumeHost, volume string) string {
	return path.Join(host.GetPluginDir(doVolumePluginName), mount.MountsInGlobalPDPath, volume)
}
