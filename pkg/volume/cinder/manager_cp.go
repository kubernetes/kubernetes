/*
Copyright 2015 The Kubernetes Authors.

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

package cinder

import (
	"errors"
	"fmt"
	"path"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/openstack"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/rackspace"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

const checkSleepDuration = 1 * time.Second

type CinderProvider interface {
	AttachDisk(instanceID string, diskName string) (string, error)
	DetachDisk(instanceID string, partialDiskId string) error
	DeleteVolume(volumeName string) error
	CreateVolume(name string, size int, vtype, availability string, tags *map[string]string) (volumeName string, err error)
	GetDevicePath(diskId string) string
	InstanceID() (string, error)
	GetAttachmentDiskPath(instanceID string, diskName string) (string, error)
	DiskIsAttached(diskName, instanceID string) (bool, error)
	DisksAreAttached(diskNames []string, instanceID string) (map[string]bool, error)
	ShouldTrustDevicePath() bool
	Instances() (cloudprovider.Instances, bool)
}

type cdManagerCloud struct {
	plugin *cinderPlugin
}

func (cdc *cdManagerCloud) GetName() string {
	return "cloud"
}

func (cdc *cdManagerCloud) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	cloud, err := getCloudProvider(cdc.plugin)
	if err != nil {
		return "", err
	}

	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	volumeID := volumeSource.VolumeID

	instances, res := cloud.Instances()
	if !res {
		return "", fmt.Errorf("failed to list openstack instances")
	}
	instanceid, err := instances.InstanceID(nodeName)
	if err != nil {
		return "", err
	}
	if ind := strings.LastIndex(instanceid, "/"); ind >= 0 {
		instanceid = instanceid[(ind + 1):]
	}
	attached, err := cloud.DiskIsAttached(volumeID, instanceid)
	if err != nil {
		// Log error and continue with attach
		glog.Warningf(
			"Error checking if volume (%q) is already attached to current instance (%q). Will continue and try attach anyway. err=%v",
			volumeID, instanceid, err)
	}

	if err == nil && attached {
		// Volume is already attached to instance.
		glog.Infof("Attach operation is successful. volume %q is already attached to instance %q.", volumeID, instanceid)
	} else {
		_, err = cloud.AttachDisk(instanceid, volumeID)
		if err == nil {
			glog.Infof("Attach operation successful: volume %q attached to instance %q.", volumeID, instanceid)
		} else {
			glog.Infof("Attach volume %q to instance %q failed with %v", volumeID, instanceid, err)
			return "", err
		}
	}

	devicePath, err := cloud.GetAttachmentDiskPath(instanceid, volumeID)
	if err != nil {
		glog.Infof("Attach volume %q to instance %q failed with %v", volumeID, instanceid, err)
		return "", err
	}

	return devicePath, err
}

func (cdc *cdManagerCloud) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	cloud, err := getCloudProvider(cdc.plugin)
	if err != nil {
		return nil, err
	}

	volumesAttachedCheck := make(map[*volume.Spec]bool)
	volumeSpecMap := make(map[string]*volume.Spec)
	volumeIDList := []string{}
	for _, spec := range specs {
		volumeSource, _, err := getVolumeSource(spec)
		if err != nil {
			glog.Errorf("Error getting volume (%q) source : %v", spec.Name(), err)
			continue
		}

		volumeIDList = append(volumeIDList, volumeSource.VolumeID)
		volumesAttachedCheck[spec] = true
		volumeSpecMap[volumeSource.VolumeID] = spec
	}
	attachedResult, err := cloud.DisksAreAttached(volumeIDList, string(nodeName))
	if err != nil {
		// Log error and continue with attach
		glog.Errorf(
			"Error checking if Volumes (%v) are already attached to current node (%q). Will continue and try attach anyway. err=%v",
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

func (cdc *cdManagerCloud) WaitForAttach(spec *volume.Spec, devicePath string, timeout time.Duration) (string, error) {
	// NOTE: devicePath is is path as reported by Cinder, which may be incorrect and should not be used. See Issue #33128
	cloud, err := getCloudProvider(cdc.plugin)
	if err != nil {
		return "", err
	}

	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	volumeID := volumeSource.VolumeID

	if devicePath == "" {
		return "", fmt.Errorf("WaitForAttach failed for Cinder disk %q: devicePath is empty.", volumeID)
	}

	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		probeAttachedVolume()
		select {
		case <-ticker.C:
			glog.V(5).Infof("Checking Cinder disk %q is attached.", volumeID)
			probeAttachedVolume()
			if !cloud.ShouldTrustDevicePath() {
				// Using the Cinder volume ID, find the real device path (See Issue #33128)
				devicePath = cloud.GetDevicePath(volumeID)
			}
			exists, err := volumeutil.PathExists(devicePath)
			if exists && err == nil {
				glog.Infof("Successfully found attached Cinder disk %q at %v.", volumeID, devicePath)
				return devicePath, nil
			} else {
				// Log an error, and continue checking periodically
				glog.Errorf("Error: could not find attached Cinder disk %q (path: %q): %v", volumeID, devicePath, err)
			}
		case <-timer.C:
			return "", fmt.Errorf("Could not find attached Cinder disk %q. Timeout waiting for mount paths to be created.", volumeID)
		}
	}
}

func (cdc *cdManagerCloud) Detach(_ *volume.Spec, deviceMountPath string, nodeName types.NodeName) error {
	cloud, err := getCloudProvider(cdc.plugin)
	if err != nil {
		return err
	}

	volumeID := path.Base(deviceMountPath)
	instances, res := cloud.Instances()
	if !res {
		return fmt.Errorf("failed to list openstack instances")
	}
	instanceid, err := instances.InstanceID(nodeName)
	if ind := strings.LastIndex(instanceid, "/"); ind >= 0 {
		instanceid = instanceid[(ind + 1):]
	}

	attached, err := cloud.DiskIsAttached(volumeID, instanceid)
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

	if err = cloud.DetachDisk(instanceid, volumeID); err != nil {
		glog.Errorf("Error detaching volume %q: %v", volumeID, err)
		return err
	}
	glog.Infof("detatached volume %q from instance %q", volumeID, instanceid)
	return nil
}

func (cdc *cdManagerCloud) WaitForDetach(_ *volume.Spec, devicePath string, timeout time.Duration) error {
	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			glog.V(5).Infof("Checking device %q is detached.", devicePath)
			if pathExists, err := volumeutil.PathExists(devicePath); err != nil {
				return fmt.Errorf("Error checking if device path exists: %v", err)
			} else if !pathExists {
				return nil
			}
		case <-timer.C:
			return fmt.Errorf("Timeout reached; PD Device %v is still attached", devicePath)
		}
	}
}

func (cdc *cdManagerCloud) UnmountDevice(_ *volume.Spec, deviceMountPath string, mounter mount.Interface) error {
	return volumeutil.UnmountPath(deviceMountPath, mounter)
}

func (cdc *cdManagerCloud) CreateVolume(c *cinderVolumeProvisioner) (volumeID string, volumeSizeGB int, secretRef *v1.LocalObjectReference, err error) {
	cloud, err := getCloudProvider(cdc.plugin)
	if err != nil {
		return "", 0, nil, err
	}

	capacity := c.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	volSizeBytes := capacity.Value()
	// Cinder works with gigabytes, convert to GiB with rounding up
	volSizeGB := int(volume.RoundUpSize(volSizeBytes, 1024*1024*1024))
	name := volume.GenerateVolumeName(c.options.ClusterName, c.options.PVName, 255) // Cinder volume name can have up to 255 characters
	vtype := ""
	availability := ""
	// Apply ProvisionerParameters (case-insensitive). We leave validation of
	// the values to the cloud provider.
	for k, v := range c.options.Parameters {
		switch strings.ToLower(k) {
		case "type":
			vtype = v
		case "availability":
			availability = v
		default:
			return "", 0, nil, fmt.Errorf("invalid option %q for volume plugin %s", k, c.plugin.GetPluginName())
		}
	}
	// TODO: implement PVC.Selector parsing
	if c.options.PVC.Spec.Selector != nil {
		return "", 0, nil, fmt.Errorf("claim.Spec.Selector is not supported for dynamic provisioning on Cinder")
	}

	name, err = cloud.CreateVolume(name, volSizeGB, vtype, availability, c.options.CloudTags)
	if err != nil {
		glog.V(2).Infof("Error creating cinder volume: %v", err)
		return "", 0, nil, err
	}
	glog.V(2).Infof("Successfully created cinder volume %s", name)
	return name, volSizeGB, nil, nil
}

func (cdc *cdManagerCloud) DeleteVolume(cd *cinderVolumeDeleter) error {
	cloud, err := getCloudProvider(cdc.plugin)
	if err != nil {
		return err
	}

	if err = cloud.DeleteVolume(cd.pdName); err != nil {
		// OpenStack cloud provider returns volume.tryAgainError when necessary,
		// no handling needed here.
		glog.V(2).Infof("Error deleting cinder volume %s: %v", cd.pdName, err)
		return err
	}
	glog.V(2).Infof("Successfully deleted cinder volume %s", cd.pdName)
	return nil
}

func probeAttachedVolume() error {
	executor := exec.New()
	args := []string{"trigger"}
	cmd := executor.Command("/usr/bin/udevadm", args...)
	_, err := cmd.CombinedOutput()
	if err != nil {
		glog.Errorf("error running udevadm trigger %v\n", err)
		return err
	}
	glog.V(4).Infof("Successfully probed all attachments")
	return nil
}

func getCloudProvider(plugin *cinderPlugin) (CinderProvider, error) {
	// For test purposes
	if plugin.cinderProvider != nil {
		return plugin.cinderProvider, nil
	}

	cloud := plugin.host.GetCloudProvider()
	if cloud == nil {
		glog.Errorf("Cloud provider not initialized properly")
		return nil, errors.New("Cloud provider not initialized properly")
	}

	switch cloud := cloud.(type) {
	case *rackspace.Rackspace:
		return cloud, nil
	case *openstack.OpenStack:
		return cloud, nil
	default:
		return nil, errors.New("Invalid cloud provider: expected OpenStack or Rackspace.")
	}
}
