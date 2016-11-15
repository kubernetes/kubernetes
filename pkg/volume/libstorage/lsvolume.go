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
	"os"
	"sync"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
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
	k8mtx   sync.Mutex

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

// CanMount checks to verify that the volume can be mounted prior to Setup.
// A nil error indicates that the volume is ready for mounitnig.
func (m *lsVolume) CanMount() error {
	return nil
}

func (m *lsVolume) SetUp(fsGroup *int64) error {
	return m.SetUpAt(m.GetPath(), fsGroup)
}

// SetUp bind mounts the disk global mount to the volume path.
func (m *lsVolume) SetUpAt(dir string, fsGroup *int64) error {
	glog.Info("libStorage: setting up volume")

	options := []string{}
	if m.readOnly {
		options = append(options, "ro")
	}

	pdPath := m.makePDPath(m.plugin.lsMgr.getService(), m.volName)

	// attach the volume and mount
	glog.V(4).Infof("libStorage: attaching volume %s", m.volName)
	devicePath, err := m.plugin.lsMgr.attachVolume(m.volName)
	if err != nil {
		glog.Errorf("libStorage: failed to attach volume:  %v", err)
		return err
	}

	notDevMnt, err := m.mounter.IsLikelyNotMountPoint(pdPath)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("libStorage: IsLikelyNotMountPoint test failed for dir %v", pdPath)
		return err
	}

	if notDevMnt {
		glog.V(4).Infof("libStorage: mounting device %s -> %s", devicePath, pdPath)

		if err := os.MkdirAll(pdPath, 0750); err != nil {
			glog.Errorf("libStorage: failed to create dir %#v:  %v", pdPath, err)
			return err
		}
		glog.V(4).Infof("libStorage: created directory %s", pdPath)

		diskMounter := &mount.SafeFormatAndMount{
			Interface: m.mounter,
			Runner:    exec.New(),
		}
		err = diskMounter.FormatAndMount(
			devicePath,
			pdPath,
			m.fsType,
			options,
		)

		if err != nil {
			os.Remove(pdPath)
			return err
		}
		glog.V(4).Infof(
			"libStorage: formatted %s:%s [%s,%+v], mounted as %s",
			m.volName, devicePath, m.fsType, options, pdPath)
	} else {
		glog.Warningf("libStorage: already mounted: %s", pdPath)
	}

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

	// bind-mount for pod
	options = append(options, "bind")
	glog.V(4).Infof("libStorage: bind-mount %s ->  %s", pdPath, dir)
	if err := os.MkdirAll(dir, 0750); err != nil {
		glog.Errorf("libStorage: mkdir failed: %v", err)
		return err
	}
	glog.V(4).Infof("libStorage: created bind-mout target dir %s", dir)

	if _, err := os.Stat(dir); err != nil {
		glog.Errorf("libStorage Error creating dir %v: %v", dir, err)
	} else {
		glog.V(4).Infof("libStorage: mount dir created ok %v", dir)
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
	m.k8mtx.Lock()
	defer m.k8mtx.Unlock()

	glog.Infof("libStorage: tearing down bind-mount to dir %s", dir)
	notMnt, err := m.mounter.IsLikelyNotMountPoint(dir)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("libStorage: checking mount point %s failed: %v", dir, err)
		return err
	}

	dirRemoved := false
	if notMnt {
		if err := os.Remove(dir); err != nil && !os.IsNotExist(err) {
			return err
		}
		dirRemoved = true
		glog.V(4).Infof("libStorage: dir %s removed", dir)
	}

	// Unmount the bind-mount inside this pod
	// only do this if dir is still around and it's a mnt point
	if !dirRemoved {
		if err := m.mounter.Unmount(dir); err != nil {
			glog.V(2).Infof("libStorage: error unmounting dir %s: %v ", dir, err)
			return err
		}
		glog.V(4).Infof("libStorage: dir %s unmounted successfully", dir)

		// check again on dir
		notMnt, mntErr := m.mounter.IsLikelyNotMountPoint(dir)
		if mntErr != nil && !os.IsNotExist(mntErr) {
			glog.Errorf("libStorage: mount point check failed for %s: %v", dir, mntErr)
			return err
		}
		if notMnt {
			if err := os.Remove(dir); err != nil && !os.IsNotExist(err) {
				glog.V(2).Infof("libStorage: error removing bind-mount dir %s: %v", dir, err)
				return err
			}
			glog.V(4).Infof("libStorage: removed dir %s", dir)
		}
	}

	//unmount device
	if m.plugin.lsMgr != nil && m.plugin.lsMgr.getHost() != "" {
		pdPathRemoved := false
		pdPath := m.makePDPath(m.plugin.lsMgr.getService(), m.volName)
		glog.V(4).Info("libStorage: attempting to unmout device diretory %s", pdPath)

		notMnt, err = m.mounter.IsLikelyNotMountPoint(pdPath)
		if err != nil && !os.IsNotExist(err) {
			glog.Errorf("libStorage: checking mount point %s failed: %v", dir, err)
			return err
		}

		if notMnt {
			if err := os.Remove(pdPath); err != nil && !os.IsNotExist(err) {
				return err
			}
			pdPathRemoved = true
			glog.V(4).Infof("libStorage: dir %s removed", pdPath)
		}

		if !pdPathRemoved {
			if err := m.mounter.Unmount(pdPath); err != nil {
				glog.V(2).Infof("libStorage: error unmounting dir %s: %v ", dir, err)
				return err
			}
			glog.V(4).Infof("libStorage: dir %s unmounted successfully", dir)

			// check mount point again
			notMnt, err = m.mounter.IsLikelyNotMountPoint(pdPath)
			if err != nil && !os.IsNotExist(err) {
				glog.Errorf("libStorage: checking mount point %s failed: %v", dir, err)
				return err
			}

			if notMnt {
				if err := os.Remove(pdPath); err != nil && !os.IsNotExist(err) {
					return err
				}
				pdPathRemoved = true
				glog.V(4).Infof("libStorage: dir %s removed", pdPath)
			}
		}

		if pdPathRemoved {
			if err := m.plugin.lsMgr.detachVolume(m.volName); err != nil {
				glog.Errorf("libStorage: failed detaching volume %s  %v", m.volName, err)
				return err
			}
			glog.V(4).Infof("libStorage: volume %v detached successfully", m.volName)
		}
	} else {
		glog.Warningf("libStorage: did not receive lsclient settings, volume %s may not have been detached", m.volName)
	}

	glog.V(4).Infof("libStorage: teardown successful")
	return nil
}

// ********************
// volume.Deleter Impl
// ********************
var _ volume.Deleter = &lsVolume{}

func (m *lsVolume) Delete() error {
	err := m.plugin.lsMgr.deleteVolume(m.volName)
	if err != nil {
		glog.Errorf("libStorage: failed to delete volume %s: %v", m.volName, err)
		return err
	}

	glog.V(4).Infof("libStorage: successfully deleted %s", m.volName)
	return nil
}
