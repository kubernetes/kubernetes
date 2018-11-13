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

package azure_dd

import (
	"fmt"
	"os"
	"runtime"

	"k8s.io/api/core/v1"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

type azureDiskMounter struct {
	*dataDisk
	spec    *volume.Spec
	plugin  *azureDataDiskPlugin
	options volume.VolumeOptions
}

type azureDiskUnmounter struct {
	*dataDisk
	plugin *azureDataDiskPlugin
}

var _ volume.Unmounter = &azureDiskUnmounter{}
var _ volume.Mounter = &azureDiskMounter{}

func (m *azureDiskMounter) GetAttributes() volume.Attributes {
	readOnly := false
	volumeSource, _, err := getVolumeSource(m.spec)
	if err != nil {
		klog.Infof("azureDisk - mounter failed to get volume source for spec %s %v", m.spec.Name(), err)
	} else if volumeSource.ReadOnly != nil {
		readOnly = *volumeSource.ReadOnly
	}
	return volume.Attributes{
		ReadOnly:        readOnly,
		Managed:         !readOnly,
		SupportsSELinux: true,
	}
}

func (m *azureDiskMounter) CanMount() error {
	return nil
}

func (m *azureDiskMounter) SetUp(fsGroup *int64) error {
	return m.SetUpAt(m.GetPath(), fsGroup)
}

func (m *azureDiskMounter) GetPath() string {
	return getPath(m.dataDisk.podUID, m.dataDisk.volumeName, m.plugin.host)
}

func (m *azureDiskMounter) SetUpAt(dir string, fsGroup *int64) error {
	mounter := m.plugin.host.GetMounter(m.plugin.GetPluginName())
	volumeSource, _, err := getVolumeSource(m.spec)

	if err != nil {
		klog.Infof("azureDisk - mounter failed to get volume source for spec %s", m.spec.Name())
		return err
	}

	diskName := volumeSource.DiskName
	mountPoint, err := mounter.IsLikelyNotMountPoint(dir)

	if err != nil && !os.IsNotExist(err) {
		klog.Infof("azureDisk - cannot validate mount point for disk %s on  %s %v", diskName, dir, err)
		return err
	}
	if !mountPoint {
		// testing original mount point, make sure the mount link is valid
		_, err := (&osIOHandler{}).ReadDir(dir)
		if err == nil {
			klog.V(4).Infof("azureDisk - already mounted to target %s", dir)
			return nil
		}
		// mount link is invalid, now unmount and remount later
		klog.Warningf("azureDisk - ReadDir %s failed with %v, unmount this directory", dir, err)
		if err := mounter.Unmount(dir); err != nil {
			klog.Errorf("azureDisk - Unmount directory %s failed with %v", dir, err)
			return err
		}
		mountPoint = true
	}

	if runtime.GOOS != "windows" {
		// in windows, we will use mklink to mount, will MkdirAll in Mount func
		if err := os.MkdirAll(dir, 0750); err != nil {
			klog.Errorf("azureDisk - mkdir failed on disk %s on dir: %s (%v)", diskName, dir, err)
			return err
		}
	}

	options := []string{"bind"}

	if volumeSource.ReadOnly != nil && *volumeSource.ReadOnly {
		options = append(options, "ro")
	}

	if m.options.MountOptions != nil {
		options = util.JoinMountOptions(m.options.MountOptions, options)
	}

	klog.V(4).Infof("azureDisk - Attempting to mount %s on %s", diskName, dir)
	isManagedDisk := (*volumeSource.Kind == v1.AzureManagedDisk)
	globalPDPath, err := makeGlobalPDPath(m.plugin.host, volumeSource.DataDiskURI, isManagedDisk)

	if err != nil {
		return err
	}

	mountErr := mounter.Mount(globalPDPath, dir, *volumeSource.FSType, options)
	// Everything in the following control flow is meant as an
	// attempt cleanup a failed setupAt (bind mount)
	if mountErr != nil {
		klog.Infof("azureDisk - SetupAt:Mount disk:%s at dir:%s failed during mounting with error:%v, will attempt to clean up", diskName, dir, mountErr)
		mountPoint, err := mounter.IsLikelyNotMountPoint(dir)
		if err != nil {
			return fmt.Errorf("azureDisk - SetupAt:Mount:Failure:cleanup IsLikelyNotMountPoint check failed for disk:%s on dir:%s with error %v original-mountErr:%v", diskName, dir, err, mountErr)
		}

		if !mountPoint {
			if err = mounter.Unmount(dir); err != nil {
				return fmt.Errorf("azureDisk - SetupAt:Mount:Failure:cleanup failed to unmount disk:%s on dir:%s with error:%v original-mountErr:%v", diskName, dir, err, mountErr)
			}
			mountPoint, err := mounter.IsLikelyNotMountPoint(dir)
			if err != nil {
				return fmt.Errorf("azureDisk - SetupAt:Mount:Failure:cleanup IsLikelyNotMountPoint for disk:%s on dir:%s check failed with error:%v original-mountErr:%v", diskName, dir, err, mountErr)
			}
			if !mountPoint {
				// not cool. leave for next sync loop.
				return fmt.Errorf("azureDisk - SetupAt:Mount:Failure:cleanup disk %s is still mounted on %s during cleanup original-mountErr:%v, despite call to unmount(). Will try again next sync loop.", diskName, dir, mountErr)
			}
		}

		if err = os.Remove(dir); err != nil {
			return fmt.Errorf("azureDisk - SetupAt:Mount:Failure error cleaning up (removing dir:%s) with error:%v original-mountErr:%v", dir, err, mountErr)
		}

		klog.V(2).Infof("azureDisk - Mount of disk:%s on dir:%s failed with mount error:%v post failure clean up was completed", diskName, dir, mountErr)
		return mountErr
	}

	if volumeSource.ReadOnly == nil || !*volumeSource.ReadOnly {
		volume.SetVolumeOwnership(m, fsGroup)
	}

	klog.V(2).Infof("azureDisk - successfully mounted disk %s on %s", diskName, dir)
	return nil
}

func (u *azureDiskUnmounter) TearDown() error {
	return u.TearDownAt(u.GetPath())
}

func (u *azureDiskUnmounter) TearDownAt(dir string) error {
	if pathExists, pathErr := util.PathExists(dir); pathErr != nil {
		return fmt.Errorf("Error checking if path exists: %v", pathErr)
	} else if !pathExists {
		klog.Warningf("Warning: Unmount skipped because path does not exist: %v", dir)
		return nil
	}

	klog.V(4).Infof("azureDisk - TearDownAt: %s", dir)
	mounter := u.plugin.host.GetMounter(u.plugin.GetPluginName())
	mountPoint, err := mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		return fmt.Errorf("azureDisk - TearDownAt: %s failed to do IsLikelyNotMountPoint %s", dir, err)
	}
	if mountPoint {
		if err := os.Remove(dir); err != nil {
			return fmt.Errorf("azureDisk - TearDownAt: %s failed to do os.Remove %s", dir, err)
		}
	}
	if err := mounter.Unmount(dir); err != nil {
		return fmt.Errorf("azureDisk - TearDownAt: %s failed to do mounter.Unmount %s", dir, err)
	}
	mountPoint, err = mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		return fmt.Errorf("azureDisk - TearTownAt:IsLikelyNotMountPoint check failed: %v", err)
	}

	if mountPoint {
		return os.Remove(dir)
	}

	return fmt.Errorf("azureDisk - failed to un-bind-mount volume dir")
}

func (u *azureDiskUnmounter) GetPath() string {
	return getPath(u.dataDisk.podUID, u.dataDisk.volumeName, u.plugin.host)
}
