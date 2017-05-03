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

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

type azureDiskDetacher struct {
	plugin *azureDataDiskPlugin
}

type azureDiskAttacher struct {
	plugin *azureDataDiskPlugin
}

var _ volume.Attacher = &azureDiskAttacher{}
var _ volume.Detacher = &azureDiskDetacher{}

func (a *azureDiskAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	var lun int
	var err error
	volumeSource, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	diskName := volumeSource.DiskName
	hashedDiskUri := makeCRC32(strings.ToLower(volumeSource.DataDiskURI))
	glog.V(4).Infof("azureDisk - attempting to check if disk %s attached to node %s", diskName, nodeName)

	isManagedDisk := (*volumeSource.Kind == v1.AzureManagedDisk)
	attached, lun, err := a.plugin.commonController.isDiskAttached(hashedDiskUri, string(nodeName), isManagedDisk)

	if err != nil {
		glog.Infof("azureDisk - error checking if Azure Disk  (%s) is already attached to  node (%s). Will continue and try attach anyway. err:%v", diskName, nodeName, err)
	}

	if err == nil && attached {
		glog.Warningf("azureDisk - disk already attach: (%s) is already attached to node (%s)", diskName, nodeName)
		return strconv.Itoa(lun), nil
	} else {
		glog.V(4).Infof("azureDisk - Disk(%s) is not  attached to node (%s), will attach", diskName, nodeName)
	}

	err = nil // reset just in case the check for attached earlier failed

	cachingMode := string(*volumeSource.CachingMode)
	managed := (*volumeSource.Kind == v1.AzureManagedDisk)
	if managed {
		lun, err = a.plugin.managedDiskController.AttachDisk(string(nodeName), volumeSource.DataDiskURI, cachingMode)
	} else {
		lun, err = a.plugin.blobDiskController.AttachDisk(string(nodeName), volumeSource.DataDiskURI, cachingMode)
	}

	if err != nil {
		glog.Warningf("azureDisk - error attaching disk:%s (Managed:%v) to node:%v error:%v", diskName, managed, nodeName, err)
		return "", err
	}

	return strconv.Itoa(lun), nil
}

func (a *azureDiskAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	attachedDisks, err := a.plugin.commonController.getAttachedDisks(string(nodeName))
	if err != nil {
		return nil, err
	}

	specsMap := make(map[*volume.Spec]bool)

	for _, s := range specs {
		azureSpec, err := getVolumeSource(s)
		if err != nil {
			glog.Warningf("azureDisk - failed to get volume source for a spec during VolumesAreAttached, err: %s", err.Error())
		}
		for _, d := range attachedDisks {
			attachedDisk := strings.ToLower(d)
			specDisk := strings.ToLower(azureSpec.DataDiskURI)
			if attachedDisk == specDisk {
				specsMap[s] = true
				break
			}
		}
	}
	return specsMap, nil
}

func (a *azureDiskAttacher) WaitForAttach(spec *volume.Spec, devicePath string, timeout time.Duration) (string, error) {
	var err error
	lun, err := strconv.Atoi(devicePath)
	if err != nil {
		return "", fmt.Errorf("azureDisk - Wait for attach expect device path as a lun number, instead got: %s", devicePath)
	}

	volumeSource, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	io := &osIOHandler{}
	scsiHostRescan(io)

	diskName := volumeSource.DiskName
	nodeName := a.plugin.host.GetHostName()
	newDevicePath := ""

	err = wait.Poll(1*time.Second, timeout, func() (bool, error) {
		exe := exec.New()

		if newDevicePath, err = findDiskByLun(lun, io, exe); err != nil {
			return false, fmt.Errorf("azureDisk - WaitForAttach ticker failed node (%s) disk (%s) lun(%v) err(%s)", nodeName, diskName, lun, err)
		}

		// did we find it?
		if newDevicePath != "" {
			// the curent sequence k8s uses for unformated disk (check-disk, mount, fail, mkfs.extX) hangs on
			// Azure Managed disk scsi interface. this is a hack and will be replaced once we identify and solve
			// the root case on Azure.
			formatIfNotFormatted(newDevicePath, *volumeSource.FSType)
			return true, nil
		}

		return false, fmt.Errorf("azureDisk - WaitForAttach failed within timeout node (%s) diskId:(%s) lun:(%v)", nodeName, diskName, lun)
	})

	return newDevicePath, err
}

// to avoid name conflicts (similar *.vhd name)
// we use hash diskUri and we use it as device mount target.
// this is generalized for both managed and blob disks
// we also prefix the hash with m/b based on disk kind
func (a *azureDiskAttacher) GetDeviceMountPath(spec *volume.Spec) (string, error) {
	volumeSource, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	if volumeSource.Kind == nil { // this spec was constructed from info on the node
		pdPath := path.Join(a.plugin.host.GetPluginDir(azureDataDiskPluginName), mount.MountsInGlobalPDPath, volumeSource.DataDiskURI)
		return pdPath, nil
	}

	isManagedDisk := (*volumeSource.Kind == v1.AzureManagedDisk)
	return makeGlobalPDPath(a.plugin.host, volumeSource.DataDiskURI, isManagedDisk)
}

func (attacher *azureDiskAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	mounter := attacher.plugin.host.GetMounter()
	notMnt, err := mounter.IsLikelyNotMountPoint(deviceMountPath)

	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(deviceMountPath, 0750); err != nil {
				return fmt.Errorf("azureDisk - mountDevice:CreateDirectory failed with %s", err)
			}
			notMnt = true
		} else {
			return fmt.Errorf("azureDisk - mountDevice:IsLikelyNotMountPoint failed with %s", err)
		}
	}

	volumeSource, err := getVolumeSource(spec)
	if err != nil {
		return err
	}

	options := []string{}
	if notMnt {
		diskMounter := &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()}
		mountOptions := volume.MountOptionFromSpec(spec, options...)
		err = diskMounter.FormatAndMount(devicePath, deviceMountPath, *volumeSource.FSType, mountOptions)
		if err != nil {
			if cleanErr := os.Remove(deviceMountPath); cleanErr != nil {
				return fmt.Errorf("azureDisk - mountDevice:FormatAndMount failed with %s and clean up failed with :%v", err, cleanErr)
			}
			return fmt.Errorf("azureDisk - mountDevice:FormatAndMount failed with %s", err)
		}
	}
	return nil
}

func (d *azureDiskDetacher) Detach(deviceName string, nodeName types.NodeName) error {
	isManagedDisk, diskHash := diskKindHashfromPDName(deviceName)

	attached, _, err := d.plugin.commonController.isDiskAttached(diskHash, string(nodeName), isManagedDisk)
	if err != nil {
		// Log error and continue with detach
		glog.Warningf("azureDisk - error checking if Azure (%v) is already attached to current node (%v). Will continue and try detach anyway. err=%v", deviceName, nodeName, err)
	}

	if err == nil && !attached {
		// Volume is not attached to node. Success!
		glog.Warningf("azureDisk -Detach: disk %s was not attached to node %v.", deviceName, nodeName)
		return nil
	}

	if isManagedDisk {
		if err := d.plugin.managedDiskController.DetachDisk(string(nodeName), diskHash); err != nil {
			glog.Infof("azureDisk - error detaching  managed disk (%s) from node %q. error:%s", deviceName, nodeName, err.Error())
			return err
		}
	} else {
		if err := d.plugin.blobDiskController.DetachDisk(string(nodeName), diskHash); err != nil {
			glog.Infof("azureDisk -error detaching  blob  disk (%s) from node %q. error:%s", deviceName, nodeName, err.Error())
			return err
		}
	}

	glog.V(2).Infof("azureDisk - disk:%s managed:%v was detached from node:%v", deviceName, isManagedDisk, nodeName)
	return nil
}

// UnmountDevice unmounts the volume on the node
func (detacher *azureDiskDetacher) UnmountDevice(deviceMountPath string) error {
	err := volumeutil.UnmountPath(deviceMountPath, detacher.plugin.host.GetMounter())
	if err == nil {
		glog.V(4).Infof("azureDisk - Device %s was unmounted", deviceMountPath)
	} else {
		glog.Infof("azureDisk - Device %s failed to unmount with error: %s", deviceMountPath, err.Error())
	}
	return err
}
