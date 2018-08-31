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

package gce_pd

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

type gcePersistentDiskAttacher struct {
	host     volume.VolumeHost
	gceDisks gce.Disks
}

var _ volume.Attacher = &gcePersistentDiskAttacher{}

var _ volume.DeviceMounter = &gcePersistentDiskAttacher{}

var _ volume.AttachableVolumePlugin = &gcePersistentDiskPlugin{}

var _ volume.DeviceMountableVolumePlugin = &gcePersistentDiskPlugin{}

func (plugin *gcePersistentDiskPlugin) NewAttacher() (volume.Attacher, error) {
	gceCloud, err := getCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		return nil, err
	}

	return &gcePersistentDiskAttacher{
		host:     plugin.host,
		gceDisks: gceCloud,
	}, nil
}

func (plugin *gcePersistentDiskPlugin) NewDeviceMounter() (volume.DeviceMounter, error) {
	return plugin.NewAttacher()
}

func (plugin *gcePersistentDiskPlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	mounter := plugin.host.GetMounter(plugin.GetPluginName())
	return mounter.GetMountRefs(deviceMountPath)
}

// Attach checks with the GCE cloud provider if the specified volume is already
// attached to the node with the specified Name.
// If the volume is attached, it succeeds (returns nil).
// If it is not, Attach issues a call to the GCE cloud provider to attach it.
// Callers are responsible for retrying on failure.
// Callers are responsible for thread safety between concurrent attach and
// detach operations.
func (attacher *gcePersistentDiskAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	volumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	pdName := volumeSource.PDName

	deviceName, err := getDeviceName(spec)
	if err != nil {
		return "", err
	}

	attached, err := attacher.gceDisks.DiskIsAttached(deviceName, nodeName)
	if err != nil {
		// Log error and continue with attach
		glog.Errorf(
			"Error checking if PD (%q) is already attached to current node (%q). Will continue and try attach anyway. err=%v",
			pdName, nodeName, err)
	}

	if err == nil && attached {
		// Volume is already attached to node.
		glog.Infof("Attach operation is successful. PD %q is already attached to node %q.", pdName, nodeName)
	} else {
		if err := attacher.gceDisks.AttachDisk(pdName, nodeName, deviceName, readOnly, isRegionalPD(spec)); err != nil {
			glog.Errorf("Error attaching PD %q to node %q: %+v", pdName, nodeName, err)
			return "", err
		}
	}

	return path.Join(diskByIdPath, diskGooglePrefix+deviceName), nil
}

func (attacher *gcePersistentDiskAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	volumesAttachedCheck := make(map[*volume.Spec]bool)
	volumePdNameMap := make(map[string]*volume.Spec)
	deviceNameList := []string{}
	for _, spec := range specs {
		deviceName, err := getDeviceName(spec)
		// If error is occurred, skip this volume and move to the next one
		if err != nil {
			glog.Errorf("Error computing device name for volume (specName: %q) source : %v", spec.Name(), err)
			continue
		}
		deviceNameList = append(deviceNameList, deviceName)
		volumesAttachedCheck[spec] = true
		volumePdNameMap[deviceName] = spec
	}
	attachedResult, err := attacher.gceDisks.DisksAreAttached(deviceNameList, nodeName)
	if err != nil {
		// Log error and continue with attach
		glog.Errorf(
			"Error checking if PDs (with device names: %v) are already attached to current node (%q). err=%v",
			deviceNameList, nodeName, err)
		return volumesAttachedCheck, err
	}

	for deviceName, attached := range attachedResult {
		if !attached {
			spec := volumePdNameMap[deviceName]
			volumesAttachedCheck[spec] = false
			glog.V(2).Infof("VolumesAreAttached: check volume (specName: %q; deviceName: %q) is no longer attached", spec.Name(), deviceName)
		}
	}
	return volumesAttachedCheck, nil
}

func (attacher *gcePersistentDiskAttacher) WaitForAttach(spec *volume.Spec, devicePath string, _ *v1.Pod, timeout time.Duration) (string, error) {
	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	deviceName, err := getDeviceName(spec)
	if err != nil {
		return "", err
	}

	pdName := volumeSource.PDName
	partition := ""
	if volumeSource.Partition != 0 {
		partition = strconv.Itoa(int(volumeSource.Partition))
	}

	sdBefore, err := filepath.Glob(diskSDPattern)
	if err != nil {
		glog.Errorf("Error filepath.Glob(\"%s\"): %v\r\n", diskSDPattern, err)
	}
	sdBeforeSet := sets.NewString(sdBefore...)

	devicePaths := getDiskByIdPaths(deviceName, partition)
	for {
		select {
		case <-ticker.C:
			glog.V(5).Infof("Checking GCE PD %q is attached.", pdName)
			path, err := verifyDevicePath(devicePaths, sdBeforeSet, deviceName)
			if err != nil {
				// Log error, if any, and continue checking periodically. See issue #11321
				glog.Errorf("Error verifying GCE PD (%q) is attached: %v", pdName, err)
			} else if path != "" {
				// A device path has successfully been created for the PD
				glog.Infof("Successfully found attached GCE PD %q.", pdName)
				return path, nil
			}
		case <-timer.C:
			return "", fmt.Errorf("Could not find attached GCE PD %q. Timeout waiting for mount paths to be created.", pdName)
		}
	}
}

func (attacher *gcePersistentDiskAttacher) GetDeviceMountPath(
	spec *volume.Spec) (string, error) {
	deviceName, err := getDeviceName(spec)
	if err != nil {
		return "", err
	}

	return makeGlobalPDName(attacher.host, deviceName), nil
}

func (attacher *gcePersistentDiskAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	// Only mount the PD globally once.
	mounter := attacher.host.GetMounter(gcePersistentDiskPluginName)
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

	volumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return err
	}

	options := []string{}
	if readOnly {
		options = append(options, "ro")
	}
	if notMnt {
		diskMounter := volumeutil.NewSafeFormatAndMountFromHost(gcePersistentDiskPluginName, attacher.host)
		mountOptions := volumeutil.MountOptionFromSpec(spec, options...)
		err = diskMounter.FormatAndMount(devicePath, deviceMountPath, volumeSource.FSType, mountOptions)
		if err != nil {
			os.Remove(deviceMountPath)
			return err
		}
		glog.V(4).Infof("formatting spec %v devicePath %v deviceMountPath %v fs %v with options %+v", spec.Name(), devicePath, deviceMountPath, volumeSource.FSType, options)
	}
	return nil
}

type gcePersistentDiskDetacher struct {
	host     volume.VolumeHost
	gceDisks gce.Disks
}

var _ volume.Detacher = &gcePersistentDiskDetacher{}

var _ volume.DeviceUnmounter = &gcePersistentDiskDetacher{}

func (plugin *gcePersistentDiskPlugin) NewDetacher() (volume.Detacher, error) {
	gceCloud, err := getCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		return nil, err
	}

	return &gcePersistentDiskDetacher{
		host:     plugin.host,
		gceDisks: gceCloud,
	}, nil
}

func (plugin *gcePersistentDiskPlugin) NewDeviceUnmounter() (volume.DeviceUnmounter, error) {
	return plugin.NewDetacher()
}

// Detach checks with the GCE cloud provider if the specified volume is already
// attached to the specified node. If the volume is not attached, it succeeds
// (returns nil). If it is attached, Detach issues a call to the GCE cloud
// provider to attach it.
// Callers are responsible for retrying on failure.
// Callers are responsible for thread safety between concurrent attach and detach
// operations.
func (detacher *gcePersistentDiskDetacher) Detach(volumeName string, nodeName types.NodeName) error {
	deviceName := path.Base(volumeName)

	attached, err := detacher.gceDisks.DiskIsAttached(deviceName, nodeName)
	if err != nil {
		// Log error and continue with detach
		glog.Errorf(
			"Error checking if PD (deviceName: %q) is already attached to current node (%q). Will continue and try detach anyway. err=%v",
			deviceName, nodeName, err)
	}

	if err == nil && !attached {
		// Volume is not attached to node. Success!
		glog.Infof("Detach operation is successful. PD (deviceName: %q) was not attached to node %q.", deviceName, nodeName)
		return nil
	}

	if err = detacher.gceDisks.DetachDisk(deviceName, nodeName); err != nil {
		glog.Errorf("Error detaching PD (deviceName: %q) from node %q: %v", deviceName, nodeName, err)
		return err
	}

	return nil
}

func (detacher *gcePersistentDiskDetacher) UnmountDevice(deviceMountPath string) error {
	return volumeutil.UnmountPath(deviceMountPath, detacher.host.GetMounter(gcePersistentDiskPluginName))
}
