/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package network_volume

import (
	"os"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/golang/glog"
)

type networkVolumePluginInterface interface {
	Name() string
	GetHost() volume.VolumeHost
}

type NetworkVolumePlugin struct {
	PluginName string
	Host       volume.VolumeHost
}

func (plugin *NetworkVolumePlugin) Name() string {
	return plugin.PluginName
}

func (plugin *NetworkVolumePlugin) GetHost() volume.VolumeHost {
	return plugin.Host
}

func (plugin *NetworkVolumePlugin) Init(host volume.VolumeHost) {
	plugin.Host = host
}

type networkVolumeInterface interface {
	// SetUp attaches the disk and bind mounts to the volume path.
	SetUp() error
	SetUpAt(path string) error

	// Unmounts the bind mount, and detaches the disk only if the PD
	// resource was the last reference to that disk on the kubelet.
	TearDown()
	TearDownAt() error

	GetPath() string
}

type NetworkVolume struct {
	Plugin  networkVolumePluginInterface
	PodUID  types.UID
	VolName string

	Mounter networkVolumeMounter
}

func (volume *NetworkVolume) GetPath() string {
	name := volume.Plugin.Name()
	return volume.Plugin.GetHost().GetPodVolumeDir(volume.PodUID, util.EscapeQualifiedNameForDisk(name), volume.VolName)
}

// SetUp attaches the disk and bind mounts to the volume path.
func (volume *NetworkVolume) SetUp() error {
	return volume.SetUpAt(volume.GetPath())
}

func (volume *NetworkVolume) SetUpAt(dir string) error {
	// Check if something is already mounted at the given dir.
	mountpoint, err := volume.Mounter.IsMountPoint(dir)
	glog.V(4).Infof("NetworkVolume set up: %s %v %v", dir, mountpoint, err)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if mountpoint {
		return nil
	}

	// Attach the volume
	err = volume.Mounter.AttachVolume()
	if err != nil {
		glog.Errorf("Failed to attach volume: %v", err)
		return err
	}

	// Create mount dir
	if err := os.MkdirAll(dir, 0750); err != nil {
		return err
	}

	// Use the volume's mounter to perform the actual mount
	err = volume.Mounter.MountVolume(dir)
	if err != nil {
		mountpoint, mntErr := volume.Mounter.IsMountPoint(dir)
		if mntErr != nil {
			glog.Errorf("isMountpoint check failed: %v", mntErr)
			return err
		}
		if mountpoint {
			if mntErr = volume.Mounter.UnmountVolume(dir); mntErr != nil {
				glog.Errorf("Failed to unmount: %v", mntErr)
				return err
			}
			mountpoint, mntErr := volume.Mounter.IsMountPoint(dir)
			if mntErr != nil {
				glog.Errorf("isMountpoint check failed: %v", mntErr)
				return err
			}
			if mountpoint {
				// This is very odd, we don't expect it.  We'll try again next sync loop.
				glog.Errorf("%s is still mounted, despite call to unmount().  Will try again next sync loop.", dir)
				return err
			}
		}
		//TODO: Check that there are no other mounts before detaching the volume.
		volume.Mounter.DetachVolume(volume.Mounter.GetGlobalPath())
		return err
	}
	return nil
}

// Unmounts the bind mount, and detaches the disk only if the volume
// resource was the last reference to that disk on the kubelet.
func (volume *NetworkVolume) TearDown() error {
	return volume.TearDownAt(volume.GetPath())
}

// Unmounts the bind mount, and detaches the disk only if the VOLUME
// resource was the last reference to that disk on the kubelet.
func (volume *NetworkVolume) TearDownAt(dir string) error {
	// Check the given dir and remove it
	mountpoint, err := volume.Mounter.IsMountPoint(dir)
	if err != nil {
		glog.Errorf("Error checking IsMountPoint: %v", err)
		return err
	}
	if !mountpoint {
		return os.Remove(dir)
	}

	// get a list of bind mounts referring to dir
	refs, err := mount.GetMountRefs(volume.Mounter, dir)
	if err != nil {
		return err
	}
	// Unmount the bind-mount inside this pod
	if err := volume.Mounter.UnmountVolume(dir); err != nil {
		glog.Errorf("Unmount failed: %v", err)
		return err
	}

	// If len(refs) is 1, then all bind mounts have been removed, and the
	// remaining reference is the global mount. It is safe to detach.
	if len(refs) == 1 {
		err = volume.Mounter.DetachVolume(refs[0])
		if err != nil {
			glog.Warningf("Failed to detach disk: %v", err)
		}
	}

	// Ensure unmount succeeded and remove dir
	mountpoint, mntErr := volume.Mounter.IsMountPoint(dir)
	if mntErr != nil {
		glog.Errorf("IsMountpoint check failed: %v", mntErr)
		return mntErr
	}
	if !mountpoint {
		if err := os.Remove(dir); err != nil {
			return err
		}
	}
	return nil
}

type networkVolumeMounter interface {
	mount.Interface
	// Perform special disk prep oprations and actual mount
	MountVolume(dest string) error
	// Unmount and do any special disk cleanup
	UnmountVolume(dir string) error
	// Called if the call to Mount fails to do any special cleanup required.
	AttachVolume() error
	// Detach the volume from the Node if needed.
	DetachVolume(path string) error
	// Returns the global mount path. For network volumes which
	// require a global mount after an attach this is where the
	// volume should be mounted.
	GetGlobalPath() string
}
