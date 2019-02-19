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

package photon_pd

import (
	"errors"
	"fmt"
	"io/ioutil"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	cloudprovider "k8s.io/cloud-provider"
	volumehelpers "k8s.io/cloud-provider/volume/helpers"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/photon"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

const (
	maxRetries         = 10
	checkSleepDuration = time.Second
	diskByIDPath       = "/dev/disk/by-id/"
	diskPhotonPrefix   = "wwn-0x"
)

var ErrProbeVolume = errors.New("Error scanning attached volumes")

// volNameToDeviceName is a mapping between spec.Name from detacher
// and the device name inside scsi path. Once pvscsi controller is
// supported, this won't be needed.
var volNameToDeviceName = make(map[string]string)

type PhotonDiskUtil struct{}

func removeFromScsiSubsystem(volName string) {
	// TODO: if using pvscsi controller, this won't be needed
	deviceName := volNameToDeviceName[volName]
	fileName := "/sys/block/" + deviceName + "/device/delete"
	data := []byte("1")
	ioutil.WriteFile(fileName, data, 0666)
}

func scsiHostScan() {
	// TODO: if using pvscsi controller, this won't be needed
	scsi_path := "/sys/class/scsi_host/"
	if dirs, err := ioutil.ReadDir(scsi_path); err == nil {
		for _, f := range dirs {
			name := scsi_path + f.Name() + "/scan"
			data := []byte("- - -")
			ioutil.WriteFile(name, data, 0666)
			klog.Errorf("scsiHostScan scan for %s", name)
		}
	}
}

func verifyDevicePath(path string) (string, error) {
	if pathExists, err := mount.PathExists(path); err != nil {
		return "", fmt.Errorf("Error checking if path exists: %v", err)
	} else if pathExists {
		return path, nil
	}

	klog.V(4).Infof("verifyDevicePath: path not exists yet")
	return "", nil
}

// CreateVolume creates a PhotonController persistent disk.
func (util *PhotonDiskUtil) CreateVolume(p *photonPersistentDiskProvisioner) (pdID string, capacityGB int, fstype string, err error) {
	cloud, err := getCloudProvider(p.plugin.host.GetCloudProvider())
	if err != nil {
		klog.Errorf("Photon Controller Util: CreateVolume failed to get cloud provider. Error [%v]", err)
		return "", 0, "", err
	}

	capacity := p.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	// PhotonController works with GiB, convert to GiB with rounding up
	volSizeGB, err := volumehelpers.RoundUpToGiBInt(capacity)
	if err != nil {
		return "", 0, "", err
	}
	name := volumeutil.GenerateVolumeName(p.options.ClusterName, p.options.PVName, 255)
	volumeOptions := &photon.VolumeOptions{
		CapacityGB: volSizeGB,
		Tags:       *p.options.CloudTags,
		Name:       name,
	}

	for parameter, value := range p.options.Parameters {
		switch strings.ToLower(parameter) {
		case "flavor":
			volumeOptions.Flavor = value
		case volume.VolumeParameterFSType:
			fstype = value
			klog.V(4).Infof("Photon Controller Util: Setting fstype to %s", fstype)
		default:
			klog.Errorf("Photon Controller Util: invalid option %s for volume plugin %s.", parameter, p.plugin.GetPluginName())
			return "", 0, "", fmt.Errorf("Photon Controller Util: invalid option %s for volume plugin %s.", parameter, p.plugin.GetPluginName())
		}
	}

	pdID, err = cloud.CreateDisk(volumeOptions)
	if err != nil {
		klog.Errorf("Photon Controller Util: failed to CreateDisk. Error [%v]", err)
		return "", 0, "", err
	}

	klog.V(4).Infof("Successfully created Photon Controller persistent disk %s", name)
	return pdID, volSizeGB, "", nil
}

// DeleteVolume deletes a vSphere volume.
func (util *PhotonDiskUtil) DeleteVolume(pd *photonPersistentDiskDeleter) error {
	cloud, err := getCloudProvider(pd.plugin.host.GetCloudProvider())
	if err != nil {
		klog.Errorf("Photon Controller Util: DeleteVolume failed to get cloud provider. Error [%v]", err)
		return err
	}

	if err = cloud.DeleteDisk(pd.pdID); err != nil {
		klog.Errorf("Photon Controller Util: failed to DeleteDisk for pdID %s. Error [%v]", pd.pdID, err)
		return err
	}

	klog.V(4).Infof("Successfully deleted PhotonController persistent disk %s", pd.pdID)
	return nil
}

func getCloudProvider(cloud cloudprovider.Interface) (*photon.PCCloud, error) {
	if cloud == nil {
		klog.Errorf("Photon Controller Util: Cloud provider not initialized properly")
		return nil, fmt.Errorf("Photon Controller Util: Cloud provider not initialized properly")
	}

	pcc := cloud.(*photon.PCCloud)
	if pcc == nil {
		klog.Errorf("Invalid cloud provider: expected Photon Controller")
		return nil, fmt.Errorf("Invalid cloud provider: expected Photon Controller")
	}
	return pcc, nil
}
