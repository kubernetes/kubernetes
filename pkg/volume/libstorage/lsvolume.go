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

package libstorage

import (
	"fmt"
	"os"
	"path"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/keymutex"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"

	lstypes "github.com/emccode/libstorage/api/types"
	lsutils "github.com/emccode/libstorage/api/utils"

	"github.com/golang/glog"
)

var _ volume.Volume = &lsVolume{}
var _ volume.Provisioner = &lsVolume{}
var _ volume.Deleter = &lsVolume{}
var _ volume.Mounter = &lsVolume{}
var _ volume.Unmounter = &lsVolume{}
var _ volume.Provisioner = &lsVolume{}
var _ volume.Deleter = &lsVolume{}

type lsVolume struct {
	podUID    types.UID
	volume    *lstypes.Volume
	mountPath string
	mounter   mount.Interface
	plugin    *lsPlugin
	readOnly  bool
	options   volume.VolumeOptions
	k8mtx     keymutex.KeyMutex
	volume.MetricsNil
}

func (m *lsVolume) GetPath() string {
	return m.plugin.host.GetPodVolumeDir(
		m.podUID,
		strings.EscapeQualifiedNameForDisk(lsPluginName),
		m.volume.Name)
}

// SetUp prepares and mounts/unpacks the volume to a
// self-determined directory path. The mount point and its
// content should be owned by 'fsGroup' so that it can be
// accessed by the pod. This may be called more than once, so
// implementations must be idempotent.
func (m *lsVolume) SetUp(fsGroup *int64) error {
	return m.SetUpAt(m.GetPath(), fsGroup)
}

// SetUpAt prepares and mounts/unpacks the volume to the
// specified directory path, which may or may not exist yet.
// The mount point and its content should be owned by
// 'fsGroup' so that it can be accessed by the pod. This may
// be called more than once, so implementations must be
// idempotent.
func (m *lsVolume) SetUpAt(dir string, fsGroup *int64) error {
	// make sure we can bind-mount before even continuing
	notMntPoint, err := m.mounter.IsLikelyNotMountPoint(dir)
	if err != nil && !os.IsNotExist(err) {
		glog.V(4).Infof("libStorage: IsLikelyNotMountPoint call failed: %s", err)
		return err
	}
	if !notMntPoint {
		glog.Warningf("libStorage: mount failed, device already mounted at %s", dir)
		return nil
	}

	// mount volume to host, get mountPoint
	mountPoint, err := m.mountVolume()
	if err != nil {
		return err
	}
	if mountPoint == "" {
		return fmt.Errorf("Volume %s not attached to a mountpoint", m.volume.VolumeName())
	}
	m.mountPath = mountPoint

	// attempt to bind-mount volume's mountPoint -> pod's dir
	// if any of this fails, unmount the mountPoint
	_, err = m.mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		if os.IsNotExist(err) { //create the target dir
			if err := os.MkdirAll(dir, 0750); err != nil {
				if unMntErr := m.unmountVolume(); unMntErr != nil {
					return fmt.Errorf("Unmount failed after failed MkdirAll: %s : %s", err, unMntErr)
				}
				return err
			}
		} else {
			if unMntErr := m.unmountVolume(); unMntErr != nil {
				return fmt.Errorf("Unmount after IsLikelyNotMountPoint failed: %s : %s", err, unMntErr)
			}
			return err
		}
	}

	options := []string{"bind"}
	if m.readOnly {
		options = append(options, "ro")
	}
	// bind-mount
	err = m.mounter.Mount(mountPoint, dir, "", options)
	if err != nil {
		// attempt to clen up
		notMntPoint, mntErr := m.mounter.IsLikelyNotMountPoint(dir)
		if mntErr != nil {
			glog.Errorf("libStorage: IsLikelyNotMountPoint failed after bind-mount failure: %s : %s", mntErr, err)
			return mntErr
		}
		// remove bind-mount if there
		if !notMntPoint {
			if mntErr := m.mounter.Unmount(dir); mntErr != nil {
				return fmt.Errorf("Unmount failed after bind-mount err: %s: %s", mntErr, err)
			}
		}

		if rmErr := os.Remove(dir); rmErr != nil {
			return fmt.Errorf("RemoveAll failed after bind-mount err: %s: %s", rmErr, err)
		}

		if unMntErr := m.unmountVolume(); unMntErr != nil {
			return fmt.Errorf("Unmount failed after bind-mount error: %s : %s", unMntErr, err)
		}
		return err
	}
	glog.V(3).Infof("libStorage: volume %s mounted to %s", m.mountPath, dir)
	return nil
}

func (m *lsVolume) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        m.readOnly,
		Managed:         false,
		SupportsSELinux: true,
	}
}

// TearDown unmounts the volume from a self-determined directory and
// removes traces of the SetUp procedure.
func (m *lsVolume) TearDown() error {
	return m.TearDownAt(m.GetPath())
}

// Unmounts the bind mount, and remove the volume only if the libstorage
// resource was the last reference to that volume on the kubelet.
func (m *lsVolume) TearDownAt(dir string) error {
	glog.V(5).Infof("libStorage: teardown volume %s", dir)

	// try removing dir if not mountpoint
	notMntPoint, err := m.mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		glog.V(4).Infof("libStorage: IsLikelyNotMountPoint failed %s: %s ", dir, err)
		return err
	}
	if notMntPoint {
		glog.V(4).Infof("libStorage: delete mount point %s", dir)
		return os.Remove(dir)
	}

	// if mounted, find reference for dir
	refs, err := mount.GetMountRefs(m.mounter, dir)
	if err != nil {
		glog.V(4).Infof("libStorage: error getting mount refs %s: %s ", dir, err)
		return err
	}
	if len(refs) == 0 {
		msg := fmt.Sprintf("libStorage: no mount reference found for dir %s", dir)
		glog.V(4).Info(msg)
		return fmt.Errorf(msg)
	}

	// unmount the bind mount point for dir
	mountPath := path.Base(refs[0])
	glog.V(4).Infof("libStorage: found volume %s mounted to %s", mountPath, dir)

	if err := m.mounter.Unmount(dir); err != nil {
		glog.V(4).Infof("libStorage: unmount  failed for %s", err)
		return err
	}
	glog.V(3).Infof("libStorage: unmount OK for %s\n", dir)

	// any ref left is for the kublet path that was bound mouted. Remove it.
	refs, err = mount.GetMountRefs(m.mounter, dir)
	if err != nil {
		glog.V(4).Infof("libStorage: GetMountRefs failed: %v", err)
		return err
	}
	if len(refs) == 1 {
		if err := m.unmountVolume(); err != nil {
			glog.V(4).Infof("libStorage: unmount volume %s failed: %s", m.volume.VolumeName(), err)
			return err
		}
	}

	// remove the directory for dir
	notMntPoint, err = m.mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		glog.V(4).Infof("libStorage: IsLikelyNotMountPoint failed for %s: %s ", dir, err)
		return err
	}
	if notMntPoint {
		glog.V(4).Infof("libStorage: deleting bound mount point %s", dir)
		return os.Remove(dir)
	}

	return nil
}

// Provision creates the resource by allocating the underlying volume in a
// storage system. This method should block until completion and returns
// PersistentVolume representing the created storage resource.
func (m *lsVolume) Provision() (*api.PersistentVolume, error) {
	glog.V(4).Info("libStorage: provisioning volume")
	vol, err := m.createVol()
	if err != nil {
		return nil, err
	}
	m.volume = vol

	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name:   m.options.PVName,
			Labels: map[string]string{},
			Annotations: map[string]string{
				"kubernetes.io/createdby": "libstorage-dynamic-provisioner",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: m.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   m.options.AccessModes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse(
					fmt.Sprintf("%dGi", m.volume.Size),
				),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				LibStorage: &api.LibStorageVolumeSource{
					VolumeID:   m.volume.ID,
					VolumeName: m.volume.VolumeName(),
				},
			},
		},
	}

	glog.V(4).Infof("libStorage: provisioned PV: %#v", pv)
	return pv, nil
}

func (m *lsVolume) Delete() error {
	glog.V(4).Infof("libStorage: deleting disk: %+v", m.volume)
	return m.deleteVol()
}

// ---------------- Helpers --------------------- //

// creteVol uses libstorage to create a volume, reports error if any
func (m *lsVolume) createVol() (*lstypes.Volume, error) {
	libClient, err := m.plugin.getClient()
	if err != nil {
		return nil, err
	}

	name := volume.GenerateVolumeName(m.options.ClusterName, m.options.PVName, 255)
	glog.V(4).Infof("libStorage: creating volume %v\n", name)

	volSizeBytes := m.options.Capacity.Value()
	volSizeGB := int64(volume.RoundUpSize(volSizeBytes, 1024*1024*1024))

	opts := &lstypes.VolumeCreateOpts{
		Size: &volSizeGB,
		Opts: lsutils.NewStore(),
	}

	vol, err := libClient.Storage().VolumeCreate(m.plugin.ctx, name, opts)
	if err != nil {
		glog.Errorf("libStorage: unable to create volume %s: %s", name, err)
		return nil, err
	}

	glog.V(4).Infof("libStorage: created volume %+v\n", vol)

	return vol, nil
}

// mountVolume mounts vol to host kublet and returns the mountPath.
func (m *lsVolume) mountVolume() (string, error) {
	volName := m.volume.VolumeName()
	m.k8mtx.LockKey(volName)
	defer m.k8mtx.UnlockKey(volName)

	glog.V(4).Infof("libStorage: attaching volume %q\n", volName)

	libClient, err := m.plugin.getClient()
	if err != nil {
		return "", err
	}

	// ensure volume already created in storage system
	vols, err := libClient.Storage().Volumes(
		m.plugin.ctx, &lstypes.VolumesOpts{Attachments: false})

	if err != nil {
		glog.Error("libStorage Volumes() failed: ", err)
		return "", err
	}
	// get existing known vols
	if len(vols) < 1 {
		glog.V(4).Info("libStorage: No existing volumes found")
		return "", fmt.Errorf("no existing volumes found")
	}

	// validate existing volumes
	var vol *lstypes.Volume
	for _, vol = range vols {
		if vol.VolumeName() == volName {
			break
		}
	}
	if vol == nil {
		glog.Error("libStorage: volume not found: ", volName)
		return "", fmt.Errorf("no matching volume found")
	}

	// mount vol to host kublet and return mountPath
	mountPath, _, err := libClient.Integration().Mount(
		m.plugin.ctx, "", vol.VolumeName(),
		&lstypes.VolumeMountOpts{
			OverwriteFS: false,
		},
	)
	if err != nil {
		glog.Error("libStorage: mount operation failed: ", err)
		return "", err
	}

	return mountPath, nil
}

// unmount ls volume
func (m *lsVolume) unmountVolume() error {
	libClient, err := m.plugin.getClient()
	if err != nil {
		return err
	}

	glog.V(4).Infof("libStorage: Unmounting volume %q\r\n", m.volume.VolumeName())

	err = libClient.Integration().Unmount(
		m.plugin.ctx, "", m.volume.VolumeName(), lsutils.NewStore())

	if err != nil {
		glog.Error("libStorage: unable to unmount volume ", m.volume.VolumeName())
		return err
	}

	return nil
}

func (m *lsVolume) deleteVol() error {
	libClient, err := m.plugin.getClient()
	if err != nil {
		return err
	}

	glog.V(4).Infof("libStorage: deleting volume %q\n", m.volume.VolumeName())
	err = libClient.Storage().VolumeRemove(
		m.plugin.ctx, m.volume.ID, lsutils.NewStore())
	if err != nil {
		return err
	}

	return nil
}
