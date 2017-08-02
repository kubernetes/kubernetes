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

package digitalocean

import (
	"fmt"
	"os"
	"time"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/utils/exec"
)

const (
	checkSleepDuration = time.Second
)

type doVolumeAttacher struct {
	// plugin     *doVolumePlugin
	host    volume.VolumeHost
	manager volManager
}

var _ volume.Attacher = &doVolumeAttacher{}

// Attaches the volume specified by the given spec to the node
func (va *doVolumeAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	volumeSource, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	node, err := nodeFromName(va.host, nodeName)
	if err != nil {
		return "", err
	}

	found, err := va.manager.FindDropletForNode(node)
	if err != nil {
		return "", err
	}

	// FIXME currently droplet lists don't fill volumes but they will
	// For the time being, we nned to retrieve the droplet
	droplet, err := va.manager.GetDroplet(found.ID)
	if err != nil {
		return "", err
	}

	devicePath, err := va.manager.AttachVolume(volumeSource.VolumeID, droplet.ID)
	if err != nil {
		return "", err
	}

	return devicePath, nil
}

// // VolumesAreAttached checks whether the list of volumes still attached to the specified node
func (va *doVolumeAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	volumesAttachedCheck := make(map[*volume.Spec]bool)
	volumeSpecMap := make(map[string]*volume.Spec)
	volumeIDList := []string{}
	for _, spec := range specs {
		volumeSource, err := getVolumeSource(spec)
		if err != nil {
			glog.Errorf("Error getting volume (%q) source : %v", spec.Name(), err)
			continue
		}

		volumeIDList = append(volumeIDList, volumeSource.VolumeID)
		volumesAttachedCheck[spec] = true
		volumeSpecMap[volumeSource.VolumeID] = spec
	}

	node, err := nodeFromName(va.host, nodeName)
	if err != nil {
		return nil, err
	}

	droplet, err := va.manager.FindDropletForNode(node)
	if err != nil {
		return nil, err
	}

	attachedResult, err := va.manager.DisksAreAttached(volumeIDList, droplet.ID)
	if err != nil {
		// Log error and continue with attach
		glog.Errorf(
			"Error checking if volumes (%v) are attached to current node (%q). err=%v",
			volumeIDList, nodeName, err)
		return volumesAttachedCheck, err
	}

	for volumeID, attached := range attachedResult {
		if !attached {
			spec := volumeSpecMap[volumeID]
			volumesAttachedCheck[spec] = false
			glog.V(2).Infof("VolumesAreAttached: check volume %q (specName: %q) is no longer attached", volumeID, spec.Name())
		}
	}
	return volumesAttachedCheck, nil
}

// // WaitForAttach blocks until the device is attached to this node
func (va *doVolumeAttacher) WaitForAttach(spec *volume.Spec, devicePath string, timeout time.Duration) (string, error) {
	volumeSource, err := getVolumeSource(spec)

	if err != nil {
		return "", err
	}

	if len(devicePath) == 0 {
		return "", fmt.Errorf("WaitForAttach failed for Digital Ocean volume %q: devicePath is empty.",
			volumeSource.VolumeID)
	}

	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			glog.V(5).Infof("Checking if Digital Ocean Volume %q is attached at %s", volumeSource.VolumeID, devicePath)

			if pathExists, err := volumeutil.PathExists(devicePath); err != nil {
				return "", fmt.Errorf("Error checking if path exists: %v", err)
			} else if pathExists {
				glog.Infof("Successfully found attached Digital Ocean Volume %q.", volumeSource.VolumeID)
				return devicePath, nil
			}
			glog.V(5).Infof("Digital Ocean Volume (%q) is not attached yet", volumeSource.VolumeID)
		case <-timer.C:
			return "", fmt.Errorf("Could not find attached Digital Ocean Volume %q. Timeout waiting for mount paths to be created.", volumeSource.VolumeID)
		}
	}
}

// GetDeviceMountPath returns a path where the device should
// be mounted after it is attached. This is a global mount
// point which should be bind mounted for individual volumes.
func (va *doVolumeAttacher) GetDeviceMountPath(spec *volume.Spec) (string, error) {
	volumeSource, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}
	return makeGlobalPDPath(va.host, volumeSource.VolumeID), nil
}

// // MountDevice mounts the disk to a global path which
func (va *doVolumeAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	mounter := va.host.GetMounter()

	notMnt, err := mounter.IsLikelyNotMountPoint(deviceMountPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err = os.MkdirAll(deviceMountPath, 0750); err != nil {
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

	if notMnt {
		diskMounter := &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()}
		mountOptions := volume.MountOptionFromSpec(spec)
		err = diskMounter.FormatAndMount(devicePath, deviceMountPath, volumeSource.FSType, mountOptions)
		if err != nil {
			os.Remove(deviceMountPath)
			return err
		}
	}

	return nil
}

type doVolumeDetacher struct {
	host    volume.VolumeHost
	mounter mount.Interface
	manager volManager
}

var _ volume.Detacher = &doVolumeDetacher{}

// Detach the given device from the node with the given Name.
func (vd *doVolumeDetacher) Detach(volumeID string, nodeName types.NodeName) error {

	if volumeID == "" {
		return fmt.Errorf("Cannot detach empty volume name")
	}

	node, err := nodeFromName(vd.host, nodeName)
	if err != nil {
		glog.Warningf("couln't find the kubernetes node by name %q, skip detaching", nodeName)
		return err
	}

	droplet, err := vd.manager.FindDropletForNode(node)
	if err != nil {
		glog.Warningf("no droplet id for node %q, skip detaching", nodeName)
		return err
	}

	glog.V(4).Infof("detaching %q from node %q", volumeID, nodeName)
	if err = vd.manager.DetachVolume(volumeID, droplet.ID); err != nil {
		glog.Errorf("failed to detach Digital Ocean disk %q, err %v", volumeID, err)
	}

	return nil
}

// UnmountDevice unmounts the global mount of the disk.
func (vd *doVolumeDetacher) UnmountDevice(deviceMountPath string) error {
	return volumeutil.UnmountPath(deviceMountPath, vd.mounter)
}
