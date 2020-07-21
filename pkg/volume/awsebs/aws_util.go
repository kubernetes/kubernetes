// +build !providerless

/*
Copyright 2014 The Kubernetes Authors.

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
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"k8s.io/klog/v2"
	"k8s.io/utils/mount"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	cloudprovider "k8s.io/cloud-provider"
	volumehelpers "k8s.io/cloud-provider/volume/helpers"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/legacy-cloud-providers/aws"
)

const (
	diskPartitionSuffix = ""
	checkSleepDuration  = time.Second
)

// AWSDiskUtil provides operations for EBS volume.
type AWSDiskUtil struct{}

// DeleteVolume deletes an AWS EBS volume.
func (util *AWSDiskUtil) DeleteVolume(d *awsElasticBlockStoreDeleter) error {
	cloud, err := getCloudProvider(d.awsElasticBlockStore.plugin.host.GetCloudProvider())
	if err != nil {
		return err
	}

	deleted, err := cloud.DeleteDisk(d.volumeID)
	if err != nil {
		// AWS cloud provider returns volume.deletedVolumeInUseError when
		// necessary, no handling needed here.
		klog.V(2).Infof("Error deleting EBS Disk volume %s: %v", d.volumeID, err)
		return err
	}
	if deleted {
		klog.V(2).Infof("Successfully deleted EBS Disk volume %s", d.volumeID)
	} else {
		klog.V(2).Infof("Successfully deleted EBS Disk volume %s (actually already deleted)", d.volumeID)
	}
	return nil
}

// CreateVolume creates an AWS EBS volume.
// Returns: volumeID, volumeSizeGB, labels, fstype, error
func (util *AWSDiskUtil) CreateVolume(c *awsElasticBlockStoreProvisioner, node *v1.Node, allowedTopologies []v1.TopologySelectorTerm) (aws.KubernetesVolumeID, int, map[string]string, string, error) {
	cloud, err := getCloudProvider(c.awsElasticBlockStore.plugin.host.GetCloudProvider())
	if err != nil {
		return "", 0, nil, "", err
	}

	// AWS volumes don't have Name field, store the name in Name tag
	var tags map[string]string
	if c.options.CloudTags == nil {
		tags = make(map[string]string)
	} else {
		tags = *c.options.CloudTags
	}
	tags["Name"] = volumeutil.GenerateVolumeName(c.options.ClusterName, c.options.PVName, 255) // AWS tags can have 255 characters

	capacity := c.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]

	zonesWithNodes, err := getCandidateZones(cloud, node)
	if err != nil {
		return "", 0, nil, "", fmt.Errorf("error finding candidate zone for pvc: %v", err)
	}

	volumeOptions, err := populateVolumeOptions(c.plugin.GetPluginName(), c.options.PVC.Name, capacity, tags, c.options.Parameters, node, allowedTopologies, zonesWithNodes)
	if err != nil {
		klog.V(2).Infof("Error populating EBS options: %v", err)
		return "", 0, nil, "", err
	}

	// TODO: implement PVC.Selector parsing
	if c.options.PVC.Spec.Selector != nil {
		return "", 0, nil, "", fmt.Errorf("claim.Spec.Selector is not supported for dynamic provisioning on AWS")
	}

	name, err := cloud.CreateDisk(volumeOptions)
	if err != nil {
		klog.V(2).Infof("Error creating EBS Disk volume: %v", err)
		return "", 0, nil, "", err
	}
	klog.V(2).Infof("Successfully created EBS Disk volume %s", name)

	labels, err := cloud.GetVolumeLabels(name)
	if err != nil {
		// We don't really want to leak the volume here...
		klog.Errorf("error building labels for new EBS volume %q: %v", name, err)
	}

	fstype := ""
	for k, v := range c.options.Parameters {
		if strings.ToLower(k) == volume.VolumeParameterFSType {
			fstype = v
		}
	}

	return name, volumeOptions.CapacityGB, labels, fstype, nil
}

// getCandidateZones finds possible zones that a volume can be created in
func getCandidateZones(cloud *aws.Cloud, selectedNode *v1.Node) (sets.String, error) {
	if selectedNode != nil {
		// For topology aware volume provisioning, node is already selected so we use the zone from
		// selected node directly instead of candidate zones.
		// We can assume the information is always available as node controller shall maintain it.
		return sets.NewString(), nil
	}

	// For non-topology-aware volumes (those that binds immediately), we fall back to original logic to query
	// cloud provider for possible zones
	return cloud.GetCandidateZonesForDynamicVolume()
}

// returns volumeOptions for EBS based on storageclass parameters and node configuration
func populateVolumeOptions(pluginName, pvcName string, capacityGB resource.Quantity, tags map[string]string, storageParams map[string]string, node *v1.Node, allowedTopologies []v1.TopologySelectorTerm, zonesWithNodes sets.String) (*aws.VolumeOptions, error) {
	requestGiB, err := volumehelpers.RoundUpToGiBInt(capacityGB)
	if err != nil {
		return nil, err
	}

	volumeOptions := &aws.VolumeOptions{
		CapacityGB: requestGiB,
		Tags:       tags,
	}

	// Apply Parameters (case-insensitive). We leave validation of
	// the values to the cloud provider.
	zonePresent := false
	zonesPresent := false
	var zone string
	var zones sets.String
	for k, v := range storageParams {
		switch strings.ToLower(k) {
		case "type":
			volumeOptions.VolumeType = v
		case "zone":
			zonePresent = true
			zone = v
		case "zones":
			zonesPresent = true
			zones, err = volumehelpers.ZonesToSet(v)
			if err != nil {
				return nil, fmt.Errorf("error parsing zones %s, must be strings separated by commas: %v", zones, err)
			}
		case "iopspergb":
			volumeOptions.IOPSPerGB, err = strconv.Atoi(v)
			if err != nil {
				return nil, fmt.Errorf("invalid iopsPerGB value %q: %v", v, err)
			}
		case "encrypted":
			volumeOptions.Encrypted, err = strconv.ParseBool(v)
			if err != nil {
				return nil, fmt.Errorf("invalid encrypted boolean value %q, must be true or false: %v", v, err)
			}
		case "kmskeyid":
			volumeOptions.KmsKeyID = v
		case volume.VolumeParameterFSType:
			// Do nothing but don't make this fail
		default:
			return nil, fmt.Errorf("invalid option %q for volume plugin %s", k, pluginName)
		}
	}

	volumeOptions.AvailabilityZone, err = volumehelpers.SelectZoneForVolume(zonePresent, zonesPresent, zone, zones, zonesWithNodes, node, allowedTopologies, pvcName)
	if err != nil {
		return nil, err
	}
	return volumeOptions, nil
}

// Returns the first path that exists, or empty string if none exist.
func verifyDevicePath(devicePaths []string) (string, error) {
	for _, path := range devicePaths {
		if pathExists, err := mount.PathExists(path); err != nil {
			return "", fmt.Errorf("Error checking if path exists: %v", err)
		} else if pathExists {
			return path, nil
		}
	}

	return "", nil
}

// Returns list of all paths for given EBS mount
// This is more interesting on GCE (where we are able to identify volumes under /dev/disk-by-id)
// Here it is mostly about applying the partition path
func getDiskByIDPaths(volumeID aws.KubernetesVolumeID, partition string, devicePath string) []string {
	devicePaths := []string{}
	if devicePath != "" {
		devicePaths = append(devicePaths, devicePath)
	}

	if partition != "" {
		for i, path := range devicePaths {
			devicePaths[i] = path + diskPartitionSuffix + partition
		}
	}

	// We need to find NVME volumes, which are mounted on a "random" nvme path ("/dev/nvme0n1"),
	// and we have to get the volume id from the nvme interface
	awsVolumeID, err := volumeID.MapToAWSVolumeID()
	if err != nil {
		klog.Warningf("error mapping volume %q to AWS volume: %v", volumeID, err)
	} else {
		// This is the magic name on which AWS presents NVME devices under /dev/disk/by-id/
		// For example, vol-0fab1d5e3f72a5e23 creates a symlink at /dev/disk/by-id/nvme-Amazon_Elastic_Block_Store_vol0fab1d5e3f72a5e23
		nvmeName := "nvme-Amazon_Elastic_Block_Store_" + strings.Replace(string(awsVolumeID), "-", "", -1)
		nvmePath, err := findNvmeVolume(nvmeName)
		if err != nil {
			klog.Warningf("error looking for nvme volume %q: %v", volumeID, err)
		} else if nvmePath != "" {
			devicePaths = append(devicePaths, nvmePath)
		}
	}

	return devicePaths
}

// Return cloud provider
func getCloudProvider(cloudProvider cloudprovider.Interface) (*aws.Cloud, error) {
	awsCloudProvider, ok := cloudProvider.(*aws.Cloud)
	if !ok || awsCloudProvider == nil {
		return nil, fmt.Errorf("Failed to get AWS Cloud Provider. GetCloudProvider returned %v instead", cloudProvider)
	}

	return awsCloudProvider, nil
}

// findNvmeVolume looks for the nvme volume with the specified name
// It follows the symlink (if it exists) and returns the absolute path to the device
func findNvmeVolume(findName string) (device string, err error) {
	p := filepath.Join("/dev/disk/by-id/", findName)
	stat, err := os.Lstat(p)
	if err != nil {
		if os.IsNotExist(err) {
			klog.V(6).Infof("nvme path not found %q", p)
			return "", nil
		}
		return "", fmt.Errorf("error getting stat of %q: %v", p, err)
	}

	if stat.Mode()&os.ModeSymlink != os.ModeSymlink {
		klog.Warningf("nvme file %q found, but was not a symlink", p)
		return "", nil
	}

	// Find the target, resolving to an absolute path
	// For example, /dev/disk/by-id/nvme-Amazon_Elastic_Block_Store_vol0fab1d5e3f72a5e23 -> ../../nvme2n1
	resolved, err := filepath.EvalSymlinks(p)
	if err != nil {
		return "", fmt.Errorf("error reading target of symlink %q: %v", p, err)
	}

	if !strings.HasPrefix(resolved, "/dev") {
		return "", fmt.Errorf("resolved symlink for %q was unexpected: %q", p, resolved)
	}

	return resolved, nil
}

func formatVolumeID(volumeID string) (string, error) {
	// This is a workaround to fix the issue in converting aws volume id from globalPDPath and globalMapPath
	// There are three formats for AWSEBSVolumeSource.VolumeID and they are stored on disk in paths like so:
	// VolumeID									mountPath								mapPath
	// aws:///vol-1234					aws/vol-1234						aws:/vol-1234
	// aws://us-east-1/vol-1234 aws/us-east-1/vol-1234  aws:/us-east-1/vol-1234
	// vol-1234									vol-1234								vol-1234
	// This code is for converting volume ids from paths back to AWS style VolumeIDs
	sourceName := volumeID
	if strings.HasPrefix(volumeID, "aws/") || strings.HasPrefix(volumeID, "aws:/") {
		names := strings.Split(volumeID, "/")
		length := len(names)
		if length < 2 || length > 3 {
			return "", fmt.Errorf("invalid volume name format %q", volumeID)
		}
		volName := names[length-1]
		if !strings.HasPrefix(volName, "vol-") {
			return "", fmt.Errorf("Invalid volume name format for AWS volume (%q)", volName)
		}
		if length == 2 {
			sourceName = awsURLNamePrefix + "" + "/" + volName // empty zone label
		}
		if length == 3 {
			sourceName = awsURLNamePrefix + names[1] + "/" + volName // names[1] is the zone label
		}
		klog.V(4).Infof("Convert aws volume name from %q to %q ", volumeID, sourceName)
	}
	return sourceName, nil
}

func newAWSVolumeSpec(volumeName, volumeID string, mode v1.PersistentVolumeMode) *volume.Spec {
	awsVolume := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: volumeName,
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
					VolumeID: volumeID,
				},
			},
			VolumeMode: &mode,
		},
	}
	return volume.NewSpecFromPersistentVolume(awsVolume, false)
}
