// +build !providerless

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

package vsphere_volume

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"k8s.io/klog/v2"
	"k8s.io/mount-utils"
	"k8s.io/utils/keymutex"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/legacy-cloud-providers/vsphere"
)

type vsphereVMDKAttacher struct {
	host           volume.VolumeHost
	vsphereVolumes vsphere.Volumes
}

var _ volume.Attacher = &vsphereVMDKAttacher{}

var _ volume.DeviceMounter = &vsphereVMDKAttacher{}

var _ volume.AttachableVolumePlugin = &vsphereVolumePlugin{}

var _ volume.DeviceMountableVolumePlugin = &vsphereVolumePlugin{}

// Singleton key mutex for keeping attach operations for the same host atomic
var attachdetachMutex = keymutex.NewHashed(0)

func (plugin *vsphereVolumePlugin) NewAttacher() (volume.Attacher, error) {
	vsphereCloud, err := getCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		return nil, err
	}

	return &vsphereVMDKAttacher{
		host:           plugin.host,
		vsphereVolumes: vsphereCloud,
	}, nil
}

func (plugin *vsphereVolumePlugin) NewDeviceMounter() (volume.DeviceMounter, error) {
	return plugin.NewAttacher()
}

// Attaches the volume specified by the given spec to the given host.
// On success, returns the device path where the device was attached on the
// node.
// Callers are responsible for retryinging on failure.
// Callers are responsible for thread safety between concurrent attach and
// detach operations.
func (attacher *vsphereVMDKAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	klog.V(4).InfoS("vSphere: Attach disk called for node", "node", klog.KRef("", string(nodeName)))

	// Keeps concurrent attach operations to same host atomic
	attachdetachMutex.LockKey(string(nodeName))
	defer attachdetachMutex.UnlockKey(string(nodeName))

	// vsphereCloud.AttachDisk checks if disk is already attached to host and
	// succeeds in that case, so no need to do that separately.
	diskUUID, err := attacher.vsphereVolumes.AttachDisk(volumeSource.VolumePath, volumeSource.StoragePolicyName, nodeName)
	if err != nil {
		klog.ErrorS(err, "Error attaching volume to node", "path", volumeSource.VolumePath, "node", klog.KRef("", string(nodeName)))
		return "", err
	}

	return filepath.Join(diskByIDPath, diskSCSIPrefix+diskUUID), nil
}

func (attacher *vsphereVMDKAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	klog.InfoS("Attacher.VolumesAreAttached called for node - Please use BulkVerifyVolumes for vSphere", "node", klog.KRef("", string(nodeName)))
	volumeNodeMap := map[types.NodeName][]*volume.Spec{
		nodeName: specs,
	}
	nodeVolumesResult := make(map[*volume.Spec]bool)
	nodesVerificationMap, err := attacher.BulkVerifyVolumes(volumeNodeMap)
	if err != nil {
		klog.ErrorS(err, "Attacher.VolumesAreAttached - error checking volumes for node", "node", klog.KRef("", string(nodeName)))
		return nodeVolumesResult, err
	}
	if result, ok := nodesVerificationMap[nodeName]; ok {
		return result, nil
	}
	return nodeVolumesResult, nil
}

func (attacher *vsphereVMDKAttacher) BulkVerifyVolumes(volumesByNode map[types.NodeName][]*volume.Spec) (map[types.NodeName]map[*volume.Spec]bool, error) {
	volumesAttachedCheck := make(map[types.NodeName]map[*volume.Spec]bool)
	volumePathsByNode := make(map[types.NodeName][]string)
	volumeSpecMap := make(map[string]*volume.Spec)

	for nodeName, volumeSpecs := range volumesByNode {
		for _, volumeSpec := range volumeSpecs {
			volumeSource, _, err := getVolumeSource(volumeSpec)
			if err != nil {
				klog.ErrorS(err, "Error getting volume source", "volumeSpec", volumeSpec.Name())
				continue
			}
			volPath := volumeSource.VolumePath
			volumePathsByNode[nodeName] = append(volumePathsByNode[nodeName], volPath)
			nodeVolume, nodeVolumeExists := volumesAttachedCheck[nodeName]
			if !nodeVolumeExists {
				nodeVolume = make(map[*volume.Spec]bool)
			}
			nodeVolume[volumeSpec] = true
			volumeSpecMap[volPath] = volumeSpec
			volumesAttachedCheck[nodeName] = nodeVolume
		}
	}
	attachedResult, err := attacher.vsphereVolumes.DisksAreAttached(volumePathsByNode)
	if err != nil {
		klog.ErrorS(err, "Error checking if volumes are attached to nodes", "volumePathsByNode", volumePathsByNode)
		return volumesAttachedCheck, err
	}

	for nodeName, nodeVolumes := range attachedResult {
		for volumePath, attached := range nodeVolumes {
			if !attached {
				spec := volumeSpecMap[volumePath]
				setNodeVolume(volumesAttachedCheck, spec, nodeName, false)
			}
		}
	}
	return volumesAttachedCheck, nil
}

func (attacher *vsphereVMDKAttacher) WaitForAttach(spec *volume.Spec, devicePath string, _ *v1.Pod, timeout time.Duration) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	if devicePath == "" {
		return "", fmt.Errorf("WaitForAttach failed for VMDK %q: devicePath is empty", volumeSource.VolumePath)
	}

	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			klog.V(5).InfoS("Checking VMDK is attached", "path", volumeSource.VolumePath)
			path, err := verifyDevicePath(devicePath)
			if err != nil {
				// Log error, if any, and continue checking periodically. See issue #11321
				klog.InfoS("Error verifying VMDK is attached", "path", volumeSource.VolumePath, "err", err)
			} else if path != "" {
				// A device path has successfully been created for the VMDK
				klog.InfoS("Successfully found attached VMDK.", "path", volumeSource.VolumePath)
				return path, nil
			}
		case <-timer.C:
			return "", fmt.Errorf("could not find attached VMDK %q. Timeout waiting for mount paths to be created", volumeSource.VolumePath)
		}
	}
}

// GetDeviceMountPath returns a path where the device should
// point which should be bind mounted for individual volumes.
func (attacher *vsphereVMDKAttacher) GetDeviceMountPath(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return makeGlobalPDPath(attacher.host, volumeSource.VolumePath), nil
}

// GetMountDeviceRefs finds all other references to the device referenced
// by deviceMountPath; returns a list of paths.
func (plugin *vsphereVolumePlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	mounter := plugin.host.GetMounter(plugin.GetPluginName())
	return mounter.GetMountRefs(deviceMountPath)
}

// MountDevice mounts device to global mount point.
func (attacher *vsphereVMDKAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	klog.InfoS("vSphere MountDevice do mount", "fromPath", devicePath, "mountPath", deviceMountPath)
	mounter := attacher.host.GetMounter(vsphereVolumePluginName)
	notMnt, err := mounter.IsLikelyNotMountPoint(deviceMountPath)
	if err != nil {
		if os.IsNotExist(err) {
			dir := deviceMountPath
			if runtime.GOOS == "windows" {
				dir = filepath.Dir(deviceMountPath)
			}
			if err := os.MkdirAll(dir, 0750); err != nil {
				klog.ErrorS(err, "Failed to create directory", "path", dir)
				return err
			}
			notMnt = true
		} else {
			return err
		}
	}

	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return err
	}

	options := []string{}

	if notMnt {
		diskMounter := volumeutil.NewSafeFormatAndMountFromHost(vsphereVolumePluginName, attacher.host)
		mountOptions := volumeutil.MountOptionFromSpec(spec, options...)
		err = diskMounter.FormatAndMount(devicePath, deviceMountPath, volumeSource.FSType, mountOptions)
		if err != nil {
			os.Remove(deviceMountPath)
			return err
		}
		klog.V(4).InfoS("Formatting spec, devicePath, deviceMountPath, fs with options", "specName", spec.Name(), "devicePath", devicePath, "deviceMountPath", deviceMountPath, "FSType", volumeSource.FSType, "options", options)
	}
	return nil
}

type vsphereVMDKDetacher struct {
	mounter        mount.Interface
	vsphereVolumes vsphere.Volumes
}

var _ volume.Detacher = &vsphereVMDKDetacher{}

var _ volume.DeviceUnmounter = &vsphereVMDKDetacher{}

func (plugin *vsphereVolumePlugin) NewDetacher() (volume.Detacher, error) {
	vsphereCloud, err := getCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		return nil, err
	}

	return &vsphereVMDKDetacher{
		mounter:        plugin.host.GetMounter(plugin.GetPluginName()),
		vsphereVolumes: vsphereCloud,
	}, nil
}

func (plugin *vsphereVolumePlugin) NewDeviceUnmounter() (volume.DeviceUnmounter, error) {
	return plugin.NewDetacher()
}

// Detach the given device from the given node.
func (detacher *vsphereVMDKDetacher) Detach(volumeName string, nodeName types.NodeName) error {
	volPath := getVolPathfromVolumeName(volumeName)
	attached, newVolumePath, err := detacher.vsphereVolumes.DiskIsAttached(volPath, nodeName)
	if err != nil {
		// Log error and continue with detach
		klog.ErrorS(err,
			"Error checking if volume is already attached to current node. Will continue and try detach anyway.",
			"path", volPath, "nodeName", klog.KRef("", string(nodeName)))
	}

	if err == nil && !attached {
		// Volume is already detached from node.
		klog.InfoS("Detach operation was successful. volume is already detached from node.", "path", volPath, "node", klog.KRef("", string(nodeName)))
		return nil
	}

	attachdetachMutex.LockKey(string(nodeName))
	defer attachdetachMutex.UnlockKey(string(nodeName))
	if err := detacher.vsphereVolumes.DetachDisk(newVolumePath, nodeName); err != nil {
		klog.ErrorS(err, "Error detaching volume", "path", volPath)
		return err
	}
	return nil
}

func (detacher *vsphereVMDKDetacher) UnmountDevice(deviceMountPath string) error {
	return mount.CleanupMountPoint(deviceMountPath, detacher.mounter, false)
}

func (plugin *vsphereVolumePlugin) CanAttach(spec *volume.Spec) (bool, error) {
	return true, nil
}

func (plugin *vsphereVolumePlugin) CanDeviceMount(spec *volume.Spec) (bool, error) {
	return true, nil
}

func setNodeVolume(
	nodeVolumeMap map[types.NodeName]map[*volume.Spec]bool,
	volumeSpec *volume.Spec,
	nodeName types.NodeName,
	check bool) {

	volumeMap := nodeVolumeMap[nodeName]
	if volumeMap == nil {
		volumeMap = make(map[*volume.Spec]bool)
		nodeVolumeMap[nodeName] = volumeMap
	}
	volumeMap[volumeSpec] = check
}
