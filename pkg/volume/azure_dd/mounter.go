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

package azure_dd

import (
	"fmt"
	"os"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/volume"
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
	volumeSource, _ := getVolumeSource(m.spec)
	return volume.Attributes{
		ReadOnly:        *volumeSource.ReadOnly,
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
	mounter := m.plugin.host.GetMounter()
	volumeSource, err := getVolumeSource(m.spec)

	if err != nil {
		glog.Infof("azureDisk - mounter failed to get volume source for spec %s", m.spec.Name())
		return err
	}

	diskName := volumeSource.DiskName
	mountPoint, err := mounter.IsLikelyNotMountPoint(dir)

	if err != nil && !os.IsNotExist(err) {
		glog.Infof("azureDisk - cannot validate mount point for disk %s on  %s %v", diskName, dir, err)
		return err
	}
	if !mountPoint {
		glog.Infof("azureDisk - Not a mounting point for disk %s on %s %v", diskName, dir, err)
		return nil
	}

	if err := os.MkdirAll(dir, 0750); err != nil {
		glog.Infof("azureDisk - mkdir failed on disk %s on dir: %s (%v)", diskName, dir, err)
		return err
	}

	options := []string{"bind"}

	if *volumeSource.ReadOnly {
		options = append(options, "ro")
	}

	glog.V(4).Infof("azureDisk - Attempting to mount %s on %s", diskName, dir)
	isManagedDisk := (*volumeSource.Kind == v1.AzureManagedDisk)
	globalPDPath, err := makeGlobalPDPath(m.plugin.host, volumeSource.DataDiskURI, isManagedDisk)

	if err != nil {
		return err
	}

	err = mounter.Mount(globalPDPath, dir, *volumeSource.FSType, options)
	if err != nil {
		mountPoint, err := mounter.IsLikelyNotMountPoint(dir)
		if err != nil {
			glog.Infof("azureDisk - IsLikelyNotMountPoint check failed: %v", err)
			return err
		}
		if !mountPoint {
			if err = mounter.Unmount(dir); err != nil {
				glog.Infof("azureDisk - failed to unmount: %v", err)
				return err
			}
			mountPoint, err := mounter.IsLikelyNotMountPoint(dir)
			if err != nil {
				glog.Infof("azureDisk - IsLikelyNotMountPoint check failed: %v", err)
				return err
			}
			if !mountPoint {
				// not cool. leave for next sync loop.
				glog.Infof("azureDisk - %s is still mounted, despite call to unmount().  Will try again next sync loop.", dir)
				return err
			}
		}
		_ = os.Remove(dir)
		glog.Errorf("azureDisk - Mount of disk %s on %s failed: %v", diskName, dir, err)
		return err
	}
	//TODO: do we need to do this? the bind-mount performs ro mount
	if !*volumeSource.ReadOnly {
		volume.SetVolumeOwnership(m, fsGroup)
	}

	glog.V(2).Infof("azureDisk - successfully mounted disk %s on %s", diskName, dir)
	return nil

}

func (u *azureDiskUnmounter) TearDown() error {
	return u.TearDownAt(u.GetPath())
}

func (u *azureDiskUnmounter) TearDownAt(dir string) error {
	glog.V(4).Infof("azureDisk - TearDownAt: %s", dir)

	mounter := u.plugin.host.GetMounter()
	mountPoint, err := mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		glog.Infof("azureDisk - TearDownAt: %s failed to do IsLikelyNotMountPoint %s", dir, err)
		return err
	}
	if mountPoint {
		err := os.Remove(dir)
		if err != nil {
			glog.Infof("azureDisk - TearDownAt: %s failed to do os.Remove %s", dir, err)
		}

		return err
	}
	if err := mounter.Unmount(dir); err != nil {
		if err != nil {
			glog.Infof("azureDisk - TearDownAt: %s failed to do mounter.Unmount %s", dir, err)
		}
		return err
	}
	mountPoint, err = mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		glog.Infof("azureDisk - TearTownAt:IsLikelyNotMountPoint check failed: %v", err)
		return err
	}
	if mountPoint {
		return os.Remove(dir)
	}
	return fmt.Errorf("azureDisk - failed to un-bind-mount volume dir")

}

func (u *azureDiskUnmounter) GetPath() string {
	return getPath(u.dataDisk.podUID, u.dataDisk.volumeName, u.plugin.host)
}
