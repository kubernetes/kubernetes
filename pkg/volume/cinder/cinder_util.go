// +build !providerless

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
	"context"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	volumehelpers "k8s.io/cloud-provider/volume/helpers"
	"k8s.io/kubernetes/pkg/volume"
	volutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/utils/exec"
)

// DiskUtil has utility/helper methods
type DiskUtil struct{}

// AttachDisk attaches a disk specified by a volume.CinderPersistenDisk to the current kubelet.
// Mounts the disk to its global path.
func (util *DiskUtil) AttachDisk(b *cinderVolumeMounter, globalPDPath string) error {
	options := []string{}
	if b.readOnly {
		options = append(options, "ro")
	}
	cloud, err := b.plugin.getCloudProvider()
	if err != nil {
		return err
	}
	instanceid, err := cloud.InstanceID()
	if err != nil {
		return err
	}
	diskid, err := cloud.AttachDisk(instanceid, b.pdName)
	if err != nil {
		return err
	}

	var devicePath string
	numTries := 0
	for {
		devicePath = cloud.GetDevicePath(diskid)
		probeAttachedVolume()

		_, err := os.Stat(devicePath)
		if err == nil {
			break
		}
		if err != nil && !os.IsNotExist(err) {
			return err
		}
		numTries++
		if numTries == 10 {
			return errors.New("Could not attach disk: Timeout after 60s")
		}
		time.Sleep(time.Second * 6)
	}
	notmnt, err := b.mounter.IsLikelyNotMountPoint(globalPDPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(globalPDPath, 0750); err != nil {
				return err
			}
			notmnt = true
		} else {
			return err
		}
	}
	if notmnt {
		err = b.blockDeviceMounter.FormatAndMount(devicePath, globalPDPath, b.fsType, options)
		if err != nil {
			os.Remove(globalPDPath)
			return err
		}
		klog.V(2).Infof("Safe mount successful: %q\n", devicePath)
	}
	return nil
}

// DetachDisk unmounts the device and detaches the disk from the kubelet's host machine.
func (util *DiskUtil) DetachDisk(cd *cinderVolumeUnmounter) error {
	globalPDPath := makeGlobalPDName(cd.plugin.host, cd.pdName)
	if err := cd.mounter.Unmount(globalPDPath); err != nil {
		return err
	}
	if err := os.Remove(globalPDPath); err != nil {
		return err
	}
	klog.V(2).Infof("Successfully unmounted main device: %s\n", globalPDPath)

	cloud, err := cd.plugin.getCloudProvider()
	if err != nil {
		return err
	}
	instanceid, err := cloud.InstanceID()
	if err != nil {
		return err
	}
	if err = cloud.DetachDisk(instanceid, cd.pdName); err != nil {
		return err
	}
	klog.V(2).Infof("Successfully detached cinder volume %s", cd.pdName)
	return nil
}

// DeleteVolume uses the cloud entrypoint to delete specified volume
func (util *DiskUtil) DeleteVolume(cd *cinderVolumeDeleter) error {
	cloud, err := cd.plugin.getCloudProvider()
	if err != nil {
		return err
	}

	if err = cloud.DeleteVolume(cd.pdName); err != nil {
		// OpenStack cloud provider returns volume.tryAgainError when necessary,
		// no handling needed here.
		klog.V(2).Infof("Error deleting cinder volume %s: %v", cd.pdName, err)
		return err
	}
	klog.V(2).Infof("Successfully deleted cinder volume %s", cd.pdName)
	return nil
}

func getZonesFromNodes(kubeClient clientset.Interface) (sets.String, error) {
	// TODO: caching, currently it is overkill because it calls this function
	// only when it creates dynamic PV
	zones := make(sets.String)
	nodes, err := kubeClient.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		klog.V(2).Infof("Error listing nodes")
		return zones, err
	}
	for _, node := range nodes.Items {
		if zone, ok := node.Labels[v1.LabelZoneFailureDomain]; ok {
			zones.Insert(zone)
		}
	}
	klog.V(4).Infof("zones found: %v", zones)
	return zones, nil
}

// CreateVolume uses the cloud provider entrypoint for creating a volume
func (util *DiskUtil) CreateVolume(c *cinderVolumeProvisioner, node *v1.Node, allowedTopologies []v1.TopologySelectorTerm) (volumeID string, volumeSizeGB int, volumeLabels map[string]string, fstype string, err error) {
	cloud, err := c.plugin.getCloudProvider()
	if err != nil {
		return "", 0, nil, "", err
	}

	capacity := c.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	// Cinder works with gigabytes, convert to GiB with rounding up
	volSizeGiB, err := volumehelpers.RoundUpToGiBInt(capacity)
	if err != nil {
		return "", 0, nil, "", err
	}

	name := volutil.GenerateVolumeName(c.options.ClusterName, c.options.PVName, 255) // Cinder volume name can have up to 255 characters
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
		case volume.VolumeParameterFSType:
			fstype = v
		default:
			return "", 0, nil, "", fmt.Errorf("invalid option %q for volume plugin %s", k, c.plugin.GetPluginName())
		}
	}
	// TODO: implement PVC.Selector parsing
	if c.options.PVC.Spec.Selector != nil {
		return "", 0, nil, "", fmt.Errorf("claim.Spec.Selector is not supported for dynamic provisioning on Cinder")
	}

	if availability == "" {
		// No zone specified, choose one randomly in the same region
		zones, err := getZonesFromNodes(c.plugin.host.GetKubeClient())
		if err != nil {
			klog.V(2).Infof("error getting zone information: %v", err)
			return "", 0, nil, "", err
		}
		// if we did not get any zones, lets leave it blank and gophercloud will
		// use zone "nova" as default
		if len(zones) > 0 {
			availability, err = volumehelpers.SelectZoneForVolume(false, false, "", nil, zones, node, allowedTopologies, c.options.PVC.Name)
			if err != nil {
				klog.V(2).Infof("error selecting zone for volume: %v", err)
				return "", 0, nil, "", err
			}
		}
	}

	volumeID, volumeAZ, volumeRegion, IgnoreVolumeAZ, err := cloud.CreateVolume(name, volSizeGiB, vtype, availability, c.options.CloudTags)
	if err != nil {
		klog.V(2).Infof("Error creating cinder volume: %v", err)
		return "", 0, nil, "", err
	}
	klog.V(2).Infof("Successfully created cinder volume %s", volumeID)

	// these are needed that pod is spawning to same AZ
	volumeLabels = make(map[string]string)
	if IgnoreVolumeAZ == false {
		if volumeAZ != "" {
			volumeLabels[v1.LabelZoneFailureDomain] = volumeAZ
		}
		if volumeRegion != "" {
			volumeLabels[v1.LabelZoneRegion] = volumeRegion
		}
	}
	return volumeID, volSizeGiB, volumeLabels, fstype, nil
}

func probeAttachedVolume() error {
	// rescan scsi bus
	scsiHostRescan()

	executor := exec.New()

	// udevadm settle waits for udevd to process the device creation
	// events for all hardware devices, thus ensuring that any device
	// nodes have been created successfully before proceeding.
	argsSettle := []string{"settle"}
	cmdSettle := executor.Command("udevadm", argsSettle...)
	_, errSettle := cmdSettle.CombinedOutput()
	if errSettle != nil {
		klog.Errorf("error running udevadm settle %v\n", errSettle)
	}

	args := []string{"trigger"}
	cmd := executor.Command("udevadm", args...)
	_, err := cmd.CombinedOutput()
	if err != nil {
		klog.Errorf("error running udevadm trigger %v\n", err)
		return err
	}
	klog.V(4).Infof("Successfully probed all attachments")
	return nil
}

func scsiHostRescan() {
	scsiPath := "/sys/class/scsi_host/"
	if dirs, err := ioutil.ReadDir(scsiPath); err == nil {
		for _, f := range dirs {
			name := scsiPath + f.Name() + "/scan"
			data := []byte("- - -")
			ioutil.WriteFile(name, data, 0666)
		}
	}
}
