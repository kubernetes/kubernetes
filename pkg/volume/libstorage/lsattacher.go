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
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/keymutex"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volutil "k8s.io/kubernetes/pkg/volume/util"
)

var (
	attachTimeout = 10 * time.Second
	attachSleep   = time.Second
	attachMtx     = keymutex.NewKeyMutex()
)

// ***************
// Attacher Impl
// ***************
var _ volume.Attacher = &lsVolume{}

func (m *lsVolume) Attach(spec *volume.Spec, node types.NodeName) (string, error) {
	attachMtx.LockKey(string(node))
	defer attachMtx.UnlockKey(string(node))
	glog.V(4).Infof("libStorage: attaching volume to host %v", node)

	lsSource, err := getLibStorageSource(spec)
	if err != nil {
		return "", err
	}

	m.volName = lsSource.VolumeName
	m.plugin.resetMgr(lsSource.Host, lsSource.Service)
	path, err := m.plugin.lsMgr.attachVolume(m.volName)
	if err != nil {
		glog.Errorf(
			"libStorage: failed to attach volume to host %v: %v",
			node, err,
		)
		return "", err
	}

	glog.V(4).Infof(
		"libStorage: successfully attached device %s to host %s",
		path, node,
	)

	return path, nil
}

func (m *lsVolume) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	result := make(map[*volume.Spec]bool)
	for _, spec := range specs {
		vol, err := getLibStorageSource(spec)
		if err != nil {
			glog.Errorf("libStorage: error getting volume source %v: %v", spec.Name(), err)
			continue
		}
		attached, err := m.plugin.lsMgr.isAttached(vol.VolumeName)
		if err != nil {
			glog.Errorf("libStorage: failed to get attachment status for volume %s: %v", vol.VolumeName, err)
			continue
		}
		result[spec] = attached
	}
	return result, nil
}

func (m *lsVolume) WaitForAttach(spec *volume.Spec, devicePath string, timeout time.Duration) (string, error) {
	lsSource, err := getLibStorageSource(spec)
	if err != nil {
		return "", err
	}

	if devicePath == "" {
		glog.Errorf("libStorage: failed to WaitForAttach for "+
			"volume %v, missing device path",
			lsSource.VolumeName,
		)
		return "", fmt.Errorf("missing device path for WaitForWatch")
	}

	ticker := time.NewTicker(attachSleep)
	defer ticker.Stop()

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			glog.V(5).Infof(
				"libStorage: verifying attachment of %v", lsSource.VolumeName)
			path, err := m.verifyDevicePath(devicePath)
			if err != nil {
				glog.Warningf("libStorage: failed to find device in list: %v", err)
			}
			if path != "" {
				glog.Infof("libStorage: volume %v attached as device %v",
					lsSource.VolumeName, path)
				return path, nil
			}
		case <-timer.C:
			glog.Errorf("libStorage: timed out waiting for device path for %v", m.volName)
			return "", fmt.Errorf("WaitForAttach timeout")
		}
	}
}

func (m *lsVolume) GetDeviceMountPath(spec *volume.Spec) (string, error) {
	lsSource, err := getLibStorageSource(spec)
	if err != nil {
		return "", err
	}
	return m.makePDPath(lsSource.Service, lsSource.VolumeName), nil
}

// MountDevice mounts device to global mount point.
func (m *lsVolume) MountDevice(
	spec *volume.Spec, devicePath string, deviceMountPath string) error {

	mounter := m.plugin.host.GetMounter()
	notMnt, err := mounter.IsLikelyNotMountPoint(deviceMountPath)
	if err != nil {
		if os.IsNotExist(err) {
			glog.V(4).Infof("libStorage: mkdir %v", deviceMountPath)
			if err := os.MkdirAll(deviceMountPath, 0750); err != nil {
				glog.Errorf(
					"libStorage: failed to create dir %#v:  %v",
					deviceMountPath, err,
				)
				return err
			}
			notMnt = true
		} else {
			return err
		}
	}

	lsSource, err := getLibStorageSource(spec)
	if err != nil {
		return err
	}

	glog.V(4).Infof(
		"libStorage: attempting to mount %s:%s as %s",
		lsSource.VolumeName, devicePath, deviceMountPath,
	)

	options := []string{}
	if m.readOnly {
		options = append(options, "ro")
	}

	if notMnt {
		diskMounter := &mount.SafeFormatAndMount{
			Interface: mounter,
			Runner:    exec.New(),
		}
		err = diskMounter.FormatAndMount(
			devicePath,
			deviceMountPath,
			lsSource.FSType,
			options,
		)

		if err != nil {
			os.Remove(deviceMountPath)
			return err
		}
		glog.V(4).Infof(
			"libStorage: formatted %s:%s [%s,%+v], mounted as %s",
			lsSource.VolumeName,
			devicePath,
			lsSource.FSType,
			options,
			deviceMountPath,
		)
	}
	if !notMnt {
		glog.V(4).Infof(
			"libStorage: %v already formatted/mounted at %v",
			devicePath,
			deviceMountPath,
		)
	}
	return nil
}

// ***************
// Detacher Impl
// ***************
var _ volume.Detacher = &lsVolume{}

func (m *lsVolume) Detach(deviceMountPath string, node types.NodeName) error {
	volName := path.Base(deviceMountPath)
	m.volName = volName
	glog.V(4).Infof("libStorage: detaching %v from host %v", deviceMountPath, node)

	attachMtx.LockKey(string(node))
	defer attachMtx.UnlockKey(string(node))

	if err := m.plugin.lsMgr.detachVolume(volName); err != nil {
		glog.Errorf("libStorage: failed detaching volume %s from host %s: %v", volName, node, err)
		return err
	}

	glog.V(4).Infof("libStorage: detached volume %v from host %v", volName, node)
	return nil
}

func (m *lsVolume) WaitForDetach(devicePath string, timeout time.Duration) error {
	ticker := time.NewTicker(attachSleep)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			glog.V(5).Infof("libStorage: verifying %v is detached",
				devicePath)
			pathExists, err := volutil.PathExists(devicePath)
			if err != nil {
				glog.Errorf("libStorage: error on detaching %v: %v",
					devicePath, err)
				return err
			}
			if !pathExists {
				glog.V(5).Infof("libStorage: device %v does not exist, "+
					"probably detached", devicePath)
				return nil
			}
		case <-timer.C:
			glog.Errorf("libStorage: timeout waiting to detach %v", devicePath)
			return fmt.Errorf("WaitForDetach timeout")
		}
	}
}

func (m *lsVolume) UnmountDevice(deviceMountPath string) error {
	err := volutil.UnmountPath(deviceMountPath, m.plugin.host.GetMounter())
	if err != nil {
		glog.V(4).Infof("libStorage: failed to unmount %v: %v",
			deviceMountPath, err)
		return err
	}
	glog.V(4).Infof("libStorage: successfully unmounted %v", deviceMountPath)
	return nil
}
