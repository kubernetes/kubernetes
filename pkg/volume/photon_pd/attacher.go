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

package photon_pd

import (
	"context"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/photon"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

type photonPersistentDiskAttacher struct {
	host        volume.VolumeHost
	photonDisks photon.Disks
}

var _ volume.Attacher = &photonPersistentDiskAttacher{}

var _ volume.DeviceMounter = &photonPersistentDiskAttacher{}

var _ volume.AttachableVolumePlugin = &photonPersistentDiskPlugin{}

var _ volume.DeviceMountableVolumePlugin = &photonPersistentDiskPlugin{}

func (plugin *photonPersistentDiskPlugin) NewAttacher() (volume.Attacher, error) {
	photonCloud, err := getCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		klog.Errorf("Photon Controller attacher: NewAttacher failed to get cloud provider")
		return nil, err
	}

	return &photonPersistentDiskAttacher{
		host:        plugin.host,
		photonDisks: photonCloud,
	}, nil
}

func (plugin *photonPersistentDiskPlugin) NewDeviceMounter() (volume.DeviceMounter, error) {
	return plugin.NewAttacher()
}

// Attaches the volume specified by the given spec to the given host.
// On success, returns the device path where the device was attached on the
// node.
// Callers are responsible for retryinging on failure.
// Callers are responsible for thread safety between concurrent attach and
// detach operations.
func (attacher *photonPersistentDiskAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	hostName := string(nodeName)
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		klog.Errorf("Photon Controller attacher: Attach failed to get volume source")
		return "", err
	}
	attached, err := attacher.photonDisks.DiskIsAttached(context.TODO(), volumeSource.PdID, nodeName)

	if err != nil {
		klog.Warningf("Photon Controller: couldn't check if disk is Attached for host %s, will try attach disk: %+v", hostName, err)
		attached = false
	}

	if !attached {
		klog.V(4).Infof("Photon Controller: Attach disk called for host %s", hostName)

		err = attacher.photonDisks.AttachDisk(context.TODO(), volumeSource.PdID, nodeName)
		if err != nil {
			klog.Errorf("Error attaching volume %q to node %q: %+v", volumeSource.PdID, nodeName, err)
			return "", err
		}
	}

	PdidWithNoHypens := strings.Replace(volumeSource.PdID, "-", "", -1)
	return filepath.Join(diskByIDPath, diskPhotonPrefix+PdidWithNoHypens), nil
}

func (attacher *photonPersistentDiskAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	volumesAttachedCheck := make(map[*volume.Spec]bool)
	volumeSpecMap := make(map[string]*volume.Spec)
	pdIDList := []string{}
	for _, spec := range specs {
		volumeSource, _, err := getVolumeSource(spec)
		if err != nil {
			klog.Errorf("Error getting volume (%q) source : %v", spec.Name(), err)
			continue
		}

		pdIDList = append(pdIDList, volumeSource.PdID)
		volumesAttachedCheck[spec] = true
		volumeSpecMap[volumeSource.PdID] = spec
	}
	attachedResult, err := attacher.photonDisks.DisksAreAttached(context.TODO(), pdIDList, nodeName)
	if err != nil {
		klog.Errorf(
			"Error checking if volumes (%v) are attached to current node (%q). err=%v",
			pdIDList, nodeName, err)
		return volumesAttachedCheck, err
	}

	for pdID, attached := range attachedResult {
		if !attached {
			spec := volumeSpecMap[pdID]
			volumesAttachedCheck[spec] = false
			klog.V(2).Infof("VolumesAreAttached: check volume %q (specName: %q) is no longer attached", pdID, spec.Name())
		}
	}
	return volumesAttachedCheck, nil
}

func (attacher *photonPersistentDiskAttacher) WaitForAttach(spec *volume.Spec, devicePath string, _ *v1.Pod, timeout time.Duration) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		klog.Errorf("Photon Controller attacher: WaitForAttach failed to get volume source")
		return "", err
	}

	if devicePath == "" {
		return "", fmt.Errorf("WaitForAttach failed for PD %s: devicePath is empty.", volumeSource.PdID)
	}

	// scan scsi path to discover the new disk
	scsiHostScan()

	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			klog.V(4).Infof("Checking PD %s is attached", volumeSource.PdID)
			checkPath, err := verifyDevicePath(devicePath)
			if err != nil {
				// Log error, if any, and continue checking periodically. See issue #11321
				klog.Warningf("Photon Controller attacher: WaitForAttach with devicePath %s Checking PD %s Error verify path", devicePath, volumeSource.PdID)
			} else if checkPath != "" {
				// A device path has successfully been created for the VMDK
				klog.V(4).Infof("Successfully found attached PD %s.", volumeSource.PdID)
				// map path with spec.Name()
				volName := spec.Name()
				realPath, _ := filepath.EvalSymlinks(devicePath)
				deviceName := path.Base(realPath)
				volNameToDeviceName[volName] = deviceName
				return devicePath, nil
			}
		case <-timer.C:
			return "", fmt.Errorf("Could not find attached PD %s. Timeout waiting for mount paths to be created.", volumeSource.PdID)
		}
	}
}

// GetDeviceMountPath returns a path where the device should
// point which should be bind mounted for individual volumes.
func (attacher *photonPersistentDiskAttacher) GetDeviceMountPath(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		klog.Errorf("Photon Controller attacher: GetDeviceMountPath failed to get volume source")
		return "", err
	}

	return makeGlobalPDPath(attacher.host, volumeSource.PdID), nil
}

// GetMountDeviceRefs finds all other references to the device referenced
// by deviceMountPath; returns a list of paths.
func (plugin *photonPersistentDiskPlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	mounter := plugin.host.GetMounter(plugin.GetPluginName())
	return mounter.GetMountRefs(deviceMountPath)
}

// MountDevice mounts device to global mount point.
func (attacher *photonPersistentDiskAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	mounter := attacher.host.GetMounter(photonPersistentDiskPluginName)
	notMnt, err := mounter.IsLikelyNotMountPoint(deviceMountPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(deviceMountPath, 0750); err != nil {
				klog.Errorf("Failed to create directory at %#v. err: %s", deviceMountPath, err)
				return err
			}
			notMnt = true
		} else {
			return err
		}
	}

	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		klog.Errorf("Photon Controller attacher: MountDevice failed to get volume source. err: %s", err)
		return err
	}

	options := []string{}

	if notMnt {
		diskMounter := volumeutil.NewSafeFormatAndMountFromHost(photonPersistentDiskPluginName, attacher.host)
		mountOptions := volumeutil.MountOptionFromSpec(spec)
		err = diskMounter.FormatAndMount(devicePath, deviceMountPath, volumeSource.FSType, mountOptions)
		if err != nil {
			os.Remove(deviceMountPath)
			return err
		}
		klog.V(4).Infof("formatting spec %v devicePath %v deviceMountPath %v fs %v with options %+v", spec.Name(), devicePath, deviceMountPath, volumeSource.FSType, options)
	}
	return nil
}

type photonPersistentDiskDetacher struct {
	mounter     mount.Interface
	photonDisks photon.Disks
}

var _ volume.Detacher = &photonPersistentDiskDetacher{}

var _ volume.DeviceUnmounter = &photonPersistentDiskDetacher{}

func (plugin *photonPersistentDiskPlugin) NewDetacher() (volume.Detacher, error) {
	photonCloud, err := getCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		klog.Errorf("Photon Controller attacher: NewDetacher failed to get cloud provider. err: %s", err)
		return nil, err
	}

	return &photonPersistentDiskDetacher{
		mounter:     plugin.host.GetMounter(plugin.GetPluginName()),
		photonDisks: photonCloud,
	}, nil
}

func (plugin *photonPersistentDiskPlugin) NewDeviceUnmounter() (volume.DeviceUnmounter, error) {
	return plugin.NewDetacher()
}

// Detach the given device from the given host.
func (detacher *photonPersistentDiskDetacher) Detach(volumeName string, nodeName types.NodeName) error {

	hostName := string(nodeName)
	pdID := volumeName
	attached, err := detacher.photonDisks.DiskIsAttached(context.TODO(), pdID, nodeName)
	if err != nil {
		// Log error and continue with detach
		klog.Errorf(
			"Error checking if persistent disk (%q) is already attached to current node (%q). Will continue and try detach anyway. err=%v",
			pdID, hostName, err)
	}

	if err == nil && !attached {
		// Volume is already detached from node.
		klog.V(4).Infof("detach operation was successful. persistent disk %q is already detached from node %q.", pdID, hostName)
		return nil
	}

	if err := detacher.photonDisks.DetachDisk(context.TODO(), pdID, nodeName); err != nil {
		klog.Errorf("Error detaching volume %q: %v", pdID, err)
		return err
	}
	return nil
}

func (detacher *photonPersistentDiskDetacher) WaitForDetach(devicePath string, timeout time.Duration) error {
	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			klog.V(4).Infof("Checking device %q is detached.", devicePath)
			if pathExists, err := mount.PathExists(devicePath); err != nil {
				return fmt.Errorf("Error checking if device path exists: %v", err)
			} else if !pathExists {
				return nil
			}
		case <-timer.C:
			return fmt.Errorf("Timeout reached; Device %v is still attached", devicePath)
		}
	}
}

func (detacher *photonPersistentDiskDetacher) UnmountDevice(deviceMountPath string) error {
	return mount.CleanupMountPoint(deviceMountPath, detacher.mounter, false)
}

func (plugin *photonPersistentDiskPlugin) CanAttach(spec *volume.Spec) (bool, error) {
	return true, nil
}

func (plugin *photonPersistentDiskPlugin) CanDeviceMount(spec *volume.Spec) (bool, error) {
	return true, nil
}
