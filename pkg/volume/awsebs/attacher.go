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

package awsebs

import (
	"fmt"
	"os"
	"path"
	"strconv"
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

type awsElasticBlockStoreAttacher struct {
	host       volume.VolumeHost
	awsVolumes aws.Volumes
}

var _ volume.Attacher = &awsElasticBlockStoreAttacher{}

var _ volume.DeviceMounter = &awsElasticBlockStoreAttacher{}

var _ volume.AttachableVolumePlugin = &awsElasticBlockStorePlugin{}

var _ volume.DeviceMountableVolumePlugin = &awsElasticBlockStorePlugin{}

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

func (plugin *awsElasticBlockStorePlugin) NewDeviceMounter() (volume.DeviceMounter, error) {
	return plugin.NewAttacher()
}

func (plugin *awsElasticBlockStorePlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	mounter := plugin.host.GetMounter(plugin.GetPluginName())
	return mounter.GetMountRefs(deviceMountPath)
}

func (attacher *awsElasticBlockStoreAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	volumeID := aws.KubernetesVolumeID(volumeSource.VolumeID)

	// awsCloud.AttachDisk checks if disk is already attached to node and
	// succeeds in that case, so no need to do that separately.
	devicePath, err := attacher.awsVolumes.AttachDisk(volumeID, nodeName)
	if err != nil {
		glog.Errorf("Error attaching volume %q to node %q: %+v", volumeID, nodeName, err)
		return "", err
	}

	return devicePath, nil
}

func (attacher *awsElasticBlockStoreAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {

	glog.Warningf("Attacher.VolumesAreAttached called for node %q - Please use BulkVerifyVolumes for AWS", nodeName)
	volumeNodeMap := map[types.NodeName][]*volume.Spec{
		nodeName: specs,
	}
	nodeVolumesResult := make(map[*volume.Spec]bool)
	nodesVerificationMap, err := attacher.BulkVerifyVolumes(volumeNodeMap)
	if err != nil {
		glog.Errorf("Attacher.VolumesAreAttached - error checking volumes for node %q with %v", nodeName, err)
		return nodeVolumesResult, err
	}

	if result, ok := nodesVerificationMap[nodeName]; ok {
		return result, nil
	}
	return nodeVolumesResult, nil
}

func (attacher *awsElasticBlockStoreAttacher) BulkVerifyVolumes(volumesByNode map[types.NodeName][]*volume.Spec) (map[types.NodeName]map[*volume.Spec]bool, error) {
	volumesAttachedCheck := make(map[types.NodeName]map[*volume.Spec]bool)
	diskNamesByNode := make(map[types.NodeName][]aws.KubernetesVolumeID)
	volumeSpecMap := make(map[aws.KubernetesVolumeID]*volume.Spec)

	for nodeName, volumeSpecs := range volumesByNode {
		for _, volumeSpec := range volumeSpecs {
			volumeSource, _, err := getVolumeSource(volumeSpec)

			if err != nil {
				glog.Errorf("Error getting volume (%q) source : %v", volumeSpec.Name(), err)
				continue
			}

			name := aws.KubernetesVolumeID(volumeSource.VolumeID)
			diskNamesByNode[nodeName] = append(diskNamesByNode[nodeName], name)

			nodeDisk, nodeDiskExists := volumesAttachedCheck[nodeName]

			if !nodeDiskExists {
				nodeDisk = make(map[*volume.Spec]bool)
			}
			nodeDisk[volumeSpec] = true
			volumeSpecMap[name] = volumeSpec
			volumesAttachedCheck[nodeName] = nodeDisk
		}
	}
	attachedResult, err := attacher.awsVolumes.DisksAreAttached(diskNamesByNode)

	if err != nil {
		glog.Errorf("Error checking if volumes are attached to nodes err = %v", err)
		return volumesAttachedCheck, err
	}

	for nodeName, nodeDisks := range attachedResult {
		for diskName, attached := range nodeDisks {
			if !attached {
				spec := volumeSpecMap[diskName]
				setNodeDisk(volumesAttachedCheck, spec, nodeName, false)
			}
		}
	}

	return volumesAttachedCheck, nil
}

func (attacher *awsElasticBlockStoreAttacher) WaitForAttach(spec *volume.Spec, devicePath string, _ *v1.Pod, timeout time.Duration) (string, error) {
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
		return "", fmt.Errorf("waitForAttach failed for AWS Volume %q: devicePath is empty", volumeID)
	}

	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			glog.V(5).Infof("Checking AWS Volume %q is attached.", volumeID)
			devicePaths := getDiskByIDPaths(aws.KubernetesVolumeID(volumeSource.VolumeID), partition, devicePath)
			path, err := verifyDevicePath(devicePaths)
			if err != nil {
				// Log error, if any, and continue checking periodically. See issue #11321
				glog.Errorf("Error verifying AWS Volume (%q) is attached: %v", volumeID, err)
			} else if path != "" {
				// A device path has successfully been created for the PD
				glog.Infof("Successfully found attached AWS Volume %q.", volumeID)
				return path, nil
			}
		case <-timer.C:
			return "", fmt.Errorf("could not find attached AWS Volume %q. Timeout waiting for mount paths to be created", volumeID)
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
	mounter := attacher.host.GetMounter(awsElasticBlockStorePluginName)
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
		diskMounter := volumeutil.NewSafeFormatAndMountFromHost(awsElasticBlockStorePluginName, attacher.host)
		mountOptions := volumeutil.MountOptionFromSpec(spec, options...)
		err = diskMounter.FormatAndMount(devicePath, deviceMountPath, volumeSource.FSType, mountOptions)
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

var _ volume.DeviceUnmounter = &awsElasticBlockStoreDetacher{}

func (plugin *awsElasticBlockStorePlugin) NewDetacher() (volume.Detacher, error) {
	awsCloud, err := getCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		return nil, err
	}

	return &awsElasticBlockStoreDetacher{
		mounter:    plugin.host.GetMounter(plugin.GetPluginName()),
		awsVolumes: awsCloud,
	}, nil
}

func (plugin *awsElasticBlockStorePlugin) NewDeviceUnmounter() (volume.DeviceUnmounter, error) {
	return plugin.NewDetacher()
}

func (detacher *awsElasticBlockStoreDetacher) Detach(volumeName string, nodeName types.NodeName) error {
	volumeID := aws.KubernetesVolumeID(path.Base(volumeName))

	if _, err := detacher.awsVolumes.DetachDisk(volumeID, nodeName); err != nil {
		glog.Errorf("Error detaching volumeID %q: %v", volumeID, err)
		return err
	}
	return nil
}

func (detacher *awsElasticBlockStoreDetacher) UnmountDevice(deviceMountPath string) error {
	return volumeutil.UnmountPath(deviceMountPath, detacher.mounter)
}

func setNodeDisk(
	nodeDiskMap map[types.NodeName]map[*volume.Spec]bool,
	volumeSpec *volume.Spec,
	nodeName types.NodeName,
	check bool) {

	volumeMap := nodeDiskMap[nodeName]
	if volumeMap == nil {
		volumeMap = make(map[*volume.Spec]bool)
		nodeDiskMap[nodeName] = volumeMap
	}
	volumeMap[volumeSpec] = check
}
