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

package aws_ebs

import (
	"fmt"
	"os"
	"path"
	"strconv"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

type awsElasticBlockStoreAttacher struct {
	host       volume.VolumeHost
	awsVolumes aws.Volumes
}

var _ volume.Attacher = &awsElasticBlockStoreAttacher{}

var _ volume.AttachableVolumePlugin = &awsElasticBlockStorePlugin{}

func (plugin *awsElasticBlockStorePlugin) NewAttacher() (volume.Attacher, error) {
	awsCloud, err := getCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		return nil, err
	}

	return &awsElasticBlockStoreAttacher{
		host:       plugin.host,
		awsVolumes: awsCloud,
	}, nil
}

func (plugin *awsElasticBlockStorePlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	mounter := plugin.host.GetMounter()
	return mount.GetMountRefs(mounter, deviceMountPath)
}

func (attacher *awsElasticBlockStoreAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	volumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	volumeID := aws.KubernetesVolumeID(volumeSource.VolumeID)

	// awsCloud.AttachDisk checks if disk is already attached to node and
	// succeeds in that case, so no need to do that separately.
	devicePath, err := attacher.awsVolumes.AttachDisk(volumeID, nodeName, readOnly)
	if err != nil {
		glog.Errorf("Error attaching volume %q: %+v", volumeID, err)
		return "", err
	}

	return devicePath, nil
}

func (attacher *awsElasticBlockStoreAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	volumesAttachedCheck := make(map[*volume.Spec]bool)
	volumeSpecMap := make(map[aws.KubernetesVolumeID]*volume.Spec)
	volumeIDList := []aws.KubernetesVolumeID{}
	for _, spec := range specs {
		volumeSource, _, err := getVolumeSource(spec)
		if err != nil {
			glog.Errorf("Error getting volume (%q) source : %v", spec.Name(), err)
			continue
		}

		name := aws.KubernetesVolumeID(volumeSource.VolumeID)
		volumeIDList = append(volumeIDList, name)
		volumesAttachedCheck[spec] = true
		volumeSpecMap[name] = spec
	}
	attachedResult, err := attacher.awsVolumes.DisksAreAttached(volumeIDList, nodeName)
	if err != nil {
		// Log error and continue with attach
		glog.Errorf(
			"Error checking if volumes (%v) is already attached to current node (%q). err=%v",
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

func (attacher *awsElasticBlockStoreAttacher) WaitForAttach(spec *volume.Spec, devicePath string, timeout time.Duration) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	volumeID := volumeSource.VolumeID
	partition := ""
	if volumeSource.Partition != 0 {
		partition = strconv.Itoa(int(volumeSource.Partition))
	}

	if devicePath == "" {
		return "", fmt.Errorf("WaitForAttach failed for AWS Volume %q: devicePath is empty.", volumeID)
	}

	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			glog.V(5).Infof("Checking AWS Volume %q is attached.", volumeID)
			if devicePath != "" {
				devicePaths := getDiskByIdPaths(partition, devicePath)
				path, err := verifyDevicePath(devicePaths)
				if err != nil {
					// Log error, if any, and continue checking periodically. See issue #11321
					glog.Errorf("Error verifying AWS Volume (%q) is attached: %v", volumeID, err)
				} else if path != "" {
					// A device path has successfully been created for the PD
					glog.Infof("Successfully found attached AWS Volume %q.", volumeID)
					return path, nil
				}
			} else {
				glog.V(5).Infof("AWS Volume (%q) is not attached yet", volumeID)
			}
		case <-timer.C:
			return "", fmt.Errorf("Could not find attached AWS Volume %q. Timeout waiting for mount paths to be created.", volumeID)
		}
	}
}

func (attacher *awsElasticBlockStoreAttacher) GetDeviceMountPath(
	spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return makeGlobalPDPath(attacher.host, aws.KubernetesVolumeID(volumeSource.VolumeID)), nil
}

// FIXME: this method can be further pruned.
func (attacher *awsElasticBlockStoreAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
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

	volumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return err
	}

	options := []string{}
	if readOnly {
		options = append(options, "ro")
	}
	if notMnt {
		diskMounter := &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()}
		err = diskMounter.FormatAndMount(devicePath, deviceMountPath, volumeSource.FSType, options)
		if err != nil {
			os.Remove(deviceMountPath)
			return err
		}
	}
	return nil
}

type awsElasticBlockStoreDetacher struct {
	mounter    mount.Interface
	awsVolumes aws.Volumes
}

var _ volume.Detacher = &awsElasticBlockStoreDetacher{}

func (plugin *awsElasticBlockStorePlugin) NewDetacher() (volume.Detacher, error) {
	awsCloud, err := getCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		return nil, err
	}

	return &awsElasticBlockStoreDetacher{
		mounter:    plugin.host.GetMounter(),
		awsVolumes: awsCloud,
	}, nil
}

func (detacher *awsElasticBlockStoreDetacher) Detach(deviceMountPath string, nodeName types.NodeName) error {
	volumeID := aws.KubernetesVolumeID(path.Base(deviceMountPath))

	attached, err := detacher.awsVolumes.DiskIsAttached(volumeID, nodeName)
	if err != nil {
		// Log error and continue with detach
		glog.Errorf(
			"Error checking if volume (%q) is already attached to current node (%q). Will continue and try detach anyway. err=%v",
			volumeID, nodeName, err)
	}

	if err == nil && !attached {
		// Volume is already detached from node.
		glog.Infof("detach operation was successful. volume %q is already detached from node %q.", volumeID, nodeName)
		return nil
	}

	if _, err = detacher.awsVolumes.DetachDisk(volumeID, nodeName); err != nil {
		glog.Errorf("Error detaching volumeID %q: %v", volumeID, err)
		return err
	}
	return nil
}

func (detacher *awsElasticBlockStoreDetacher) UnmountDevice(deviceMountPath string) error {
	return volumeutil.UnmountPath(deviceMountPath, detacher.mounter)
}
