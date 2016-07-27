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
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/keymutex"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

type azureDiskAttacher struct {
	host          volume.VolumeHost
	azureProvider azureCloudProvider
}

var _ volume.Attacher = &azureDiskAttacher{}

var _ volume.AttachableVolumePlugin = &azureDataDiskPlugin{}

const (
	checkSleepDuration = time.Second
)

// acquire lock to get an lun number
var getLunMutex = keymutex.NewKeyMutex()

// NewAttacher initializes an Attacher
func (plugin *azureDataDiskPlugin) NewAttacher() (volume.Attacher, error) {
	azure, err := getAzureCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		glog.V(4).Infof("failed to get azure provider")
		return nil, err
	}

	return &azureDiskAttacher{
		host:          plugin.host,
		azureProvider: azure,
	}, nil
}

// Attach attaches a volume.Spec to a Azure VM referenced by hostname, returning the disk's LUN
func (attacher *azureDiskAttacher) Attach(spec *volume.Spec, hostName string) (string, error) {
	volumeSource, err := getVolumeSource(spec)
	if err != nil {
		glog.Warningf("failed to get azure disk spec")
		return "", err
	}
	instanceid, err := attacher.azureProvider.InstanceID(hostName)
	if err != nil {
		glog.Warningf("failed to get azure instance id")
		return "", fmt.Errorf("failed to get azure instance id for host %q", hostName)
	}
	if ind := strings.LastIndex(instanceid, "/"); ind >= 0 {
		instanceid = instanceid[(ind + 1):]
	}

	lun, err := attacher.azureProvider.GetDiskLun(volumeSource.DiskName, volumeSource.DataDiskURI, instanceid)
	if err == cloudprovider.InstanceNotFound {
		// Log error and continue with attach
		glog.Warningf(
			"Error checking if volume is already attached to current node (%q). Will continue and try attach anyway. err=%v",
			instanceid, err)
	}

	if err == nil {
		// Volume is already attached to node.
		glog.V(4).Infof("Attach operation is successful. volume %q is already attached to node %q at lun %d.", volumeSource.DiskName, instanceid, lun)
	} else {
		getLunMutex.LockKey(instanceid)
		defer getLunMutex.UnlockKey(instanceid)

		lun, err = attacher.azureProvider.GetNextDiskLun(instanceid)
		if err != nil {
			glog.Warningf("no LUN available for instance %q", instanceid)
			return "", fmt.Errorf("all LUNs are used, cannot attach volume %q to instance %q", volumeSource.DiskName, instanceid)
		}

		err = attacher.azureProvider.AttachDisk(volumeSource.DiskName, volumeSource.DataDiskURI, instanceid, lun, compute.CachingTypes(*volumeSource.CachingMode))
		if err == nil {
			glog.V(4).Infof("Attach operation successful: volume %q attached to node %q.", volumeSource.DataDiskURI, instanceid)
		} else {
			glog.V(2).Infof("Attach volume %q to instance %q failed with %v", volumeSource.DataDiskURI, instanceid, err)
			return "", fmt.Errorf("Attach volume %q to instance %q failed with %v", volumeSource.DiskName, instanceid, err)
		}
	}

	return strconv.Itoa(int(lun)), err
}

// WaitForAttach runs on the node to detect if the volume (referenced by LUN) is attached. If attached, the device path is returned
func (attacher *azureDiskAttacher) WaitForAttach(spec *volume.Spec, lunStr string, timeout time.Duration) (string, error) {
	volumeSource, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	if len(lunStr) == 0 {
		return "", fmt.Errorf("WaitForAttach failed for Azure disk %q: lun is empty.", volumeSource.DiskName)
	}

	lun, err := strconv.Atoi(lunStr)
	if err != nil {
		return "", fmt.Errorf("WaitForAttach: wrong lun %q, err: %v", lunStr, err)
	}
	scsiHostRescan(&osIOHandler{})
	exe := exec.New()
	devicePath := ""

	err = wait.Poll(checkSleepDuration, timeout, func() (bool, error) {
		glog.V(4).Infof("Checking Azure disk %q(lun %s) is attached.", volumeSource.DiskName, lunStr)
		if devicePath, err = findDiskByLun(lun, &osIOHandler{}, exe); err == nil {
			glog.V(4).Infof("Successfully found attached Azure disk %q(lun %s, device path %s).", volumeSource.DiskName, lunStr, devicePath)
			return true, nil
		} else {
			//Log error, if any, and continue checking periodically
			glog.V(4).Infof("Error Stat Azure disk (%q) is attached: %v", volumeSource.DiskName, err)
			return false, nil
		}
	})
	return devicePath, err
}

// GetDeviceMountPath finds the volume's mount path on the node
func (attacher *azureDiskAttacher) GetDeviceMountPath(spec *volume.Spec) (string, error) {
	volumeSource, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return makeGlobalPDPath(attacher.host, volumeSource.DiskName), nil
}

// MountDevice runs mount command on the node to mount the volume
func (attacher *azureDiskAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	mounter := attacher.host.GetMounter()
	notMnt, err := mounter.IsLikelyNotMountPoint(deviceMountPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(deviceMountPath, 0750); err != nil {
				return err
			}
			notMnt = true
		} else {
			return err
		}
	}

	volumeSource, err := getVolumeSource(spec)
	if err != nil {
		return err
	}

	options := []string{}
	if spec.ReadOnly {
		options = append(options, "ro")
	}
	if notMnt {
		diskMounter := &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()}
		err = diskMounter.FormatAndMount(devicePath, deviceMountPath, *volumeSource.FSType, options)
		if err != nil {
			os.Remove(deviceMountPath)
			return err
		}
	}
	return nil
}

type azureDiskDetacher struct {
	mounter       mount.Interface
	azureProvider azureCloudProvider
}

var _ volume.Detacher = &azureDiskDetacher{}

// NewDetacher initializes a volume Detacher
func (plugin *azureDataDiskPlugin) NewDetacher() (volume.Detacher, error) {
	azure, err := getAzureCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		return nil, err
	}

	return &azureDiskDetacher{
		mounter:       plugin.host.GetMounter(),
		azureProvider: azure,
	}, nil
}

// Detach detaches disk from Azure VM.
func (detacher *azureDiskDetacher) Detach(diskName string, hostName string) error {
	if diskName == "" {
		return fmt.Errorf("invalid disk to detach: %q", diskName)
	}
	instanceid, err := detacher.azureProvider.InstanceID(hostName)
	if err != nil {
		glog.Warningf("no instance id for host %q, skip detaching", hostName)
		return nil
	}
	if ind := strings.LastIndex(instanceid, "/"); ind >= 0 {
		instanceid = instanceid[(ind + 1):]
	}

	glog.V(4).Infof("detach %v from host %q", diskName, instanceid)
	err = detacher.azureProvider.DetachDiskByName(diskName, "" /* diskURI */, instanceid)
	if err != nil {
		glog.Errorf("failed to detach azure disk %q, err %v", diskName, err)
	}

	return err
}

// WaitForDetach detects if the disk is detached on the node
func (detacher *azureDiskDetacher) WaitForDetach(devicePath string, timeout time.Duration) error {
	return wait.Poll(checkSleepDuration, timeout, func() (bool, error) {
		glog.V(4).Infof("Checking device %q is detached.", devicePath)
		if pathExists, err := util.PathExists(devicePath); err != nil {
			return false, fmt.Errorf("Error checking if device path exists: %v", err)
		} else if !pathExists {
			return true, nil
		} else {
			return false, nil
		}
	})
}

// UnmountDevice unmounts the volume on the node
func (detacher *azureDiskDetacher) UnmountDevice(deviceMountPath string) error {
	volume := path.Base(deviceMountPath)
	if err := util.UnmountPath(deviceMountPath, detacher.mounter); err != nil {
		glog.Errorf("Error unmounting %q: %v", volume, err)
		return err
	} else {
		return nil
	}
}
