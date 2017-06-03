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
	"io/ioutil"
	"os"
	"strings"
	"time"

	"github.com/golang/glog"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/volume"
)

type CinderDiskUtil struct{}

// Attaches a disk specified by a volume.CinderPersistenDisk to the current kubelet.
// Mounts the disk to its global path.
func (util *CinderDiskUtil) AttachDisk(b *cinderVolumeMounter, globalPDPath string) error {
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
		glog.V(2).Infof("Safe mount successful: %q\n", devicePath)
	}
	return nil
}

// Unmounts the device and detaches the disk from the kubelet's host machine.
func (util *CinderDiskUtil) DetachDisk(cd *cinderVolumeUnmounter) error {
	globalPDPath := makeGlobalPDName(cd.plugin.host, cd.pdName)
	if err := cd.mounter.Unmount(globalPDPath); err != nil {
		return err
	}
	if err := os.Remove(globalPDPath); err != nil {
		return err
	}
	glog.V(2).Infof("Successfully unmounted main device: %s\n", globalPDPath)

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
	glog.V(2).Infof("Successfully detached cinder volume %s", cd.pdName)
	return nil
}

func (util *CinderDiskUtil) DeleteVolume(cd *cinderVolumeDeleter) error {
	cloud, err := cd.plugin.getCloudProvider()
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

func getZonesFromNodes(kubeClient clientset.Interface) (sets.String, error) {
	// TODO: caching, currently it is overkill because it calls this function
	// only when it creates dynamic PV
	zones := make(sets.String)
	nodes, err := kubeClient.Core().Nodes().List(metav1.ListOptions{})
	if err != nil {
		glog.V(2).Infof("Error listing nodes")
		return zones, err
	}
	for _, node := range nodes.Items {
		if zone, ok := node.Labels[kubeletapis.LabelZoneFailureDomain]; ok {
			zones.Insert(zone)
		}
	}
	glog.V(4).Infof("zones found: %v", zones)
	return zones, nil
}

func (util *CinderDiskUtil) CreateVolume(c *cinderVolumeProvisioner) (volumeID string, volumeSizeGB int, volumeLabels map[string]string, err error) {
	cloud, err := c.plugin.getCloudProvider()
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

	if availability == "" {
		// No zone specified, choose one randomly in the same region
		zones, err := getZonesFromNodes(c.plugin.host.GetKubeClient())
		if err != nil {
			glog.V(2).Infof("error getting zone information: %v", err)
			return "", 0, nil, err
		}
		// if we did not get any zones, lets leave it blank and gophercloud will
		// use zone "nova" as default
		if len(zones) > 0 {
			availability = volume.ChooseZoneForVolume(zones, c.options.PVC.Name)
		}
	}

	volumeID, volumeAZ, errr := cloud.CreateVolume(name, volSizeGB, vtype, availability, c.options.CloudTags)
	if errr != nil {
		glog.V(2).Infof("Error creating cinder volume: %v", errr)
		return "", 0, nil, errr
	}
	glog.V(2).Infof("Successfully created cinder volume %s", volumeID)

	// these are needed that pod is spawning to same AZ
	volumeLabels = make(map[string]string)
	volumeLabels[kubeletapis.LabelZoneFailureDomain] = volumeAZ

	return volumeID, volSizeGB, volumeLabels, nil
}

func probeAttachedVolume() error {
	// rescan scsi bus
	scsiHostRescan()

	executor := exec.New()
	args := []string{"trigger"}
	cmd := executor.Command("udevadm", args...)
	_, err := cmd.CombinedOutput()
	if err != nil {
		glog.Errorf("error running udevadm trigger %v\n", err)
		return err
	}
	glog.V(4).Infof("Successfully probed all attachments")
	return nil
}

func scsiHostRescan() {
	scsi_path := "/sys/class/scsi_host/"
	if dirs, err := ioutil.ReadDir(scsi_path); err == nil {
		for _, f := range dirs {
			name := scsi_path + f.Name() + "/scan"
			data := []byte("- - -")
			ioutil.WriteFile(name, data, 0666)
		}
	}
}
