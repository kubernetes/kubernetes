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

package digitalocean_volume

import (
	"errors"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/volume"
	"os"
	"strconv"
	"time"
)

type DoDiskUtil struct{}

func (util *DoDiskUtil) CreateVolume(d *doVolumeProvisioner) (volumeID string, volumeSizeGB int, err error) {
	cloud, err := getCloudProvider(d.doVolume.plugin.host.GetCloudProvider())
	if err != nil {
		return "", 0, err
	}

	capacity := d.options.PVC.Spec.Resources.Requests[api.ResourceName(api.ResourceStorage)]
	volSizeBytes := capacity.Value()
	// DigitalOcean works with gigabytes, convert to GiB with rounding up
	volSizeGB := int(volume.RoundUpSize(volSizeBytes, 1024*1024*1024))
	name := volume.GenerateVolumeName(d.options.ClusterName, d.options.PVName, 255) // DigitalOcean volume name can have up to 255 characters

	name, err = cloud.CreateVolume(cloud.GetRegion(), name, name, int64(volSizeGB))
	if err != nil {
		glog.V(2).Infof("Error creating DigitalOcean volume: %v", err)
		return "", 0, err
	}
	glog.V(2).Infof("Successfully created DigitalOcean volume %s", name)
	return name, volSizeGB, nil
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

// Attaches a disk specified by a volume.DigitalOceanPersistenDisk to the current kubelet.
// Mounts the disk to its global path.
func (util *DoDiskUtil) AttachVolume(d *doVolumeMounter, globalPDPath string) error {
	options := []string{}
	if d.readOnly {
		options = append(options, "ro")
	}
	cloud, err := getCloudProvider(d.doVolume.plugin.host.GetCloudProvider())
	if err != nil {
		return err
	}
	instanceID, err := cloud.LocalInstanceID()
	intInstanceID, _ := strconv.Atoi(instanceID)
	if err != nil {
		return err
	}
	glog.V(4).Infof("Attaching Volume: %v to %d", d.pdName, intInstanceID)
	volumeID, err := cloud.AttachVolume(intInstanceID, d.pdName)
	if err != nil {
		return err
	}

	var devicePath string
	numTries := 0
	for {
		devicePath = cloud.GetDevicePath(volumeID)
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
	notmnt, err := d.mounter.IsLikelyNotMountPoint(globalPDPath)
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
		err = d.blockDeviceMounter.FormatAndMount(devicePath, globalPDPath, d.fsType, options)
		if err != nil {
			os.Remove(globalPDPath)
			return err
		}
		glog.V(2).Infof("Safe mount successful: %q\n", devicePath)
	}
	return nil
}

// Unmounts the device and detaches the disk from the kubelet's host machine.
func (util *DoDiskUtil) DetachVolume(d *doVolumeUnmounter) error {
	globalPDPath := makeGlobalPDName(d.plugin.host, d.pdName)
	if err := d.mounter.Unmount(globalPDPath); err != nil {
		return err
	}
	if err := os.Remove(globalPDPath); err != nil {
		return err
	}
	glog.V(2).Infof("Successfully unmounted main device: %s\n", globalPDPath)

	cloud, err := getCloudProvider(d.doVolume.plugin.host.GetCloudProvider())
	if err != nil {
		return err
	}
	instanceID, err := cloud.LocalInstanceID()
	intInstanceID, _ := strconv.Atoi(instanceID)
	if err != nil {
		return err
	}
	if err = cloud.DetachVolume(intInstanceID, d.pdName); err != nil {
		return err
	}
	glog.V(2).Infof("Successfully detached DigitalOcean volume %s", d.pdName)
	return nil
}

func (util *DoDiskUtil) DeleteVolume(d *doVolumeDeleter) error {
	cloud, err := getCloudProvider(d.doVolume.plugin.host.GetCloudProvider())
	if err != nil {
		return err
	}

	if err = cloud.DeleteVolume(d.pdName); err != nil {
		glog.V(2).Infof("Error deleting DigitalOcean volume %s: %v", d.pdName, err)
		return err
	}
	glog.V(2).Infof("Successfully deleted DigitalOcean volume %s", d.pdName)
	return nil
}
