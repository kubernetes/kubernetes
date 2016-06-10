/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

type awsElasticBlockStoreAttacher struct {
	host volume.VolumeHost
}

var _ volume.Attacher = &awsElasticBlockStoreAttacher{}

var _ volume.AttachableVolumePlugin = &awsElasticBlockStorePlugin{}

func (plugin *awsElasticBlockStorePlugin) NewAttacher() (volume.Attacher, error) {
	return &awsElasticBlockStoreAttacher{host: plugin.host}, nil
}

func (plugin *awsElasticBlockStorePlugin) GetDeviceName(spec *volume.Spec) (string, error) {
	volumeSource, _ := getVolumeSource(spec)
	if volumeSource == nil {
		return "", fmt.Errorf("Spec does not reference an EBS volume type")
	}

	return volumeSource.VolumeID, nil
}

func (attacher *awsElasticBlockStoreAttacher) Attach(spec *volume.Spec, hostName string) error {
	volumeSource, readOnly := getVolumeSource(spec)
	volumeID := volumeSource.VolumeID

	awsCloud, err := getCloudProvider(attacher.host.GetCloudProvider())
	if err != nil {
		return err
	}

	attached, err := awsCloud.DiskIsAttached(volumeID, hostName)
	if err != nil {
		// Log error and continue with attach
		glog.Errorf(
			"Error checking if volume (%q) is already attached to current node (%q). Will continue and try attach anyway. err=%v",
			volumeID, hostName, err)
	}

	if err == nil && attached {
		// Volume is already attached to node.
		glog.Infof("Attach operation is successful. volume %q is already attached to node %q.", volumeID, hostName)
		return nil
	}

	if _, err = awsCloud.AttachDisk(volumeID, hostName, readOnly); err != nil {
		glog.Errorf("Error attaching volume %q: %+v", volumeID, err)
		return err
	}
	return nil
}

func (attacher *awsElasticBlockStoreAttacher) WaitForAttach(spec *volume.Spec, timeout time.Duration) (string, error) {
	awsCloud, err := getCloudProvider(attacher.host.GetCloudProvider())
	if err != nil {
		return "", err
	}
	volumeSource, _ := getVolumeSource(spec)
	volumeID := volumeSource.VolumeID
	partition := ""
	if volumeSource.Partition != 0 {
		partition = strconv.Itoa(int(volumeSource.Partition))
	}

	devicePath := ""
	if d, err := awsCloud.GetDiskPath(volumeID); err == nil {
		devicePath = d
	} else {
		glog.Errorf("GetDiskPath %q gets error %v", volumeID, err)
	}

	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			glog.V(5).Infof("Checking AWS Volume %q is attached.", volumeID)
			if devicePath == "" {
				if d, err := awsCloud.GetDiskPath(volumeID); err == nil {
					devicePath = d
				} else {
					glog.Errorf("GetDiskPath %q gets error %v", volumeID, err)
				}
			}
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

func (attacher *awsElasticBlockStoreAttacher) GetDeviceMountPath(spec *volume.Spec) string {
	volumeSource, _ := getVolumeSource(spec)
	return makeGlobalPDPath(attacher.host, volumeSource.VolumeID)
}

// FIXME: this method can be further pruned.
func (attacher *awsElasticBlockStoreAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string, mounter mount.Interface) error {
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

	volumeSource, readOnly := getVolumeSource(spec)

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
	host volume.VolumeHost
}

var _ volume.Detacher = &awsElasticBlockStoreDetacher{}

func (plugin *awsElasticBlockStorePlugin) NewDetacher() (volume.Detacher, error) {
	return &awsElasticBlockStoreDetacher{host: plugin.host}, nil
}

func (detacher *awsElasticBlockStoreDetacher) Detach(deviceMountPath string, hostName string) error {
	volumeID := path.Base(deviceMountPath)

	awsCloud, err := getCloudProvider(detacher.host.GetCloudProvider())
	if err != nil {
		return err
	}
	attached, err := awsCloud.DiskIsAttached(volumeID, hostName)
	if err != nil {
		// Log error and continue with detach
		glog.Errorf(
			"Error checking if volume (%q) is already attached to current node (%q). Will continue and try detach anyway. err=%v",
			volumeID, hostName, err)
	}

	if err == nil && !attached {
		// Volume is already detached from node.
		glog.Infof("detach operation was successful. volume %q is already detached from node %q.", volumeID, hostName)
		return nil
	}

	if _, err = awsCloud.DetachDisk(volumeID, hostName); err != nil {
		glog.Errorf("Error detaching volumeID %q: %v", volumeID, err)
		return err
	}
	return nil
}

func (detacher *awsElasticBlockStoreDetacher) WaitForDetach(devicePath string, timeout time.Duration) error {
	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			glog.V(5).Infof("Checking device %q is detached.", devicePath)
			if pathExists, err := pathExists(devicePath); err != nil {
				return fmt.Errorf("Error checking if device path exists: %v", err)
			} else if !pathExists {
				return nil
			}
		case <-timer.C:
			return fmt.Errorf("Timeout reached; PD Device %v is still attached", devicePath)
		}
	}
}

func (detacher *awsElasticBlockStoreDetacher) UnmountDevice(deviceMountPath string, mounter mount.Interface) error {
	volume := path.Base(deviceMountPath)
	if err := unmountPDAndRemoveGlobalPath(deviceMountPath, mounter); err != nil {
		glog.Errorf("Error unmounting %q: %v", volume, err)
	}

	return nil
}

func getVolumeSource(spec *volume.Spec) (*api.AWSElasticBlockStoreVolumeSource, bool) {
	var readOnly bool
	var volumeSource *api.AWSElasticBlockStoreVolumeSource

	if spec.Volume != nil && spec.Volume.AWSElasticBlockStore != nil {
		volumeSource = spec.Volume.AWSElasticBlockStore
		readOnly = volumeSource.ReadOnly
	} else {
		volumeSource = spec.PersistentVolume.Spec.AWSElasticBlockStore
		readOnly = spec.ReadOnly
	}

	return volumeSource, readOnly
}
