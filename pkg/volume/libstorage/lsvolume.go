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

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/keymutex"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

type lsVolume struct {
	volName  string
	readOnly bool
	fsType   string

	plugin  *lsPlugin
	options volume.VolumeOptions

	podUID  types.UID
	mounter mount.Interface
	k8mtx   keymutex.KeyMutex

	volume.MetricsNil
}

// *******************
// volume.Volume Impl
// *******************
var _ volume.Volume = &lsVolume{}

func (m *lsVolume) GetPath() string {
	return m.plugin.host.GetPodVolumeDir(
		m.podUID,
		strings.EscapeQualifiedNameForDisk(lsPluginName),
		m.volName)
}

// *************
// Mounter Impl
// *************
var _ volume.Mounter = &lsVolume{}

func (m *lsVolume) SetUp(fsGroup *int64) error {
	return m.SetUpAt(m.GetPath(), fsGroup)
}

// SetUp bind mounts the disk global mount to the volume path.
func (m *lsVolume) SetUpAt(dir string, fsGroup *int64) error {
	pdPath := m.makePDPath(m.plugin.lsMgr.getService(), m.volName)
	glog.Infof(
		"libStorage: setting up bind-mount for %s:%s to %s",
		m.volName, pdPath, dir,
	)

	// make sure we can bind-mount before even continuing
	notMntPoint, err := m.mounter.IsLikelyNotMountPoint(dir)
	if err != nil && !os.IsNotExist(err) {
		glog.V(4).Infof("libStorage: IsLikelyNotMountPoint failed: %s", err)
		return err
	}
	if !notMntPoint {
		glog.Warningf("libStorage: volume %s already mounted at %s", m.volName, dir)
		return nil
	}

	glog.V(4).Infof("libStorage: mkdir %s", dir)
	if err := os.MkdirAll(dir, 0750); err != nil {
		glog.Errorf("libStorage: mkdir failed: %v", err)
		return err
	}

	options := []string{"bind"}
	if m.readOnly {
		options = append(options, "ro")
	}

	// bind-mount libstorage mountpoint to k8s dir
	glog.V(4).Infof("libStorage: bind-mounting %s:%s to %s", m.volName, pdPath, dir)
	err = m.mounter.Mount(pdPath, dir, "", options)
	if err != nil {
		notMnt, mntErr := m.mounter.IsLikelyNotMountPoint(dir)
		if mntErr != nil {
			glog.Errorf("libStorage: IsLikelyNotMountPoint failed: %v", mntErr)
			return err
		}
		if !notMnt {
			if mntErr = m.mounter.Unmount(dir); mntErr != nil {
				glog.Errorf("libStoage: failed to unmount: %v", mntErr)
				return err
			}
			notMnt, mntErr := m.mounter.IsLikelyNotMountPoint(dir)
			if mntErr != nil {
				glog.Errorf("libStorage: IsLikelyNotMountPoint failed: %v", mntErr)
				return err
			}
			if !notMnt {
				// This is very odd, we don't expect it.  We'll try again next sync loop.
				glog.Errorf("libStorage: %s is still mounted.  Will try again next sync loop.", dir)
				return err
			}
		}
		os.Remove(dir)
		glog.Errorf("libStorage: bind-mount %s failed: %v", dir, err)
		return err
	}

	glog.Infof("libStorage: successfully bind-mounted %s:%s as %s",
		m.volName, pdPath, dir)
	return nil
}

func (m *lsVolume) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        m.readOnly,
		Managed:         false,
		SupportsSELinux: true,
	}
}

// **********************
// volume.Unmounter Impl
// *********************
var _ volume.Unmounter = &lsVolume{}

// TearDownAt unmounts the bind mount
func (m *lsVolume) TearDown() error {
	return m.TearDownAt(m.GetPath())
}

// Unmounts the bind mount, and remove the volume only if the libstorage
// resource was the last reference to that volume on the kubelet.
func (m *lsVolume) TearDownAt(dir string) error {
	glog.Infof("libStorage: tearing down bind-mount to dir %s", dir)
	notMnt, err := m.mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		if os.IsNotExist(err) {
			glog.V(4).Infof("libStorage: dir does not exist, skipping unmount: %s: %v", dir, err)
			return nil
		}
		glog.V(2).Infof("libStorage: IsLikelyNotMountPoint failed for %s: %v ", dir, err)
		return err
	}
	if notMnt {
		glog.V(2).Infof("libStorage: %s is no longer a mountpoint, deleting", dir)
		return os.Remove(dir)
	}

	// Unmount the bind-mount inside this pod
	if err := m.mounter.Unmount(dir); err != nil {
		glog.V(2).Infof("libStorage: error unmounting dir %s: %v ", dir, err)
		return err
	}
	notMnt, mntErr := m.mounter.IsLikelyNotMountPoint(dir)
	if mntErr != nil {
		glog.Errorf("libStorage: IsLikelyNotMountPoint failed: %v", mntErr)
		return err
	}
	if notMnt {
		if err := os.Remove(dir); err != nil {
			glog.V(2).Infof("libStorage: error removing bind-mount dir %s: %v", dir, err)
			return err
		}
	}

	return fmt.Errorf("libStorage: failed to unmount dir %s", dir)
}

// ********************
// volume.Deleter Impl
// ********************
var _ volume.Deleter = &lsVolume{}

func (m *lsVolume) Delete() error {
	err := m.plugin.lsMgr.deleteVolume(m)
	if err != nil {
		glog.Errorf("libStorage: failed to delete volume %s: %v", m.volName, err)
		return err
	}

	glog.V(4).Infof("libStorage: successfully deleted %s", m.volName)
	return nil
}
