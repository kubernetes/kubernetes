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

package azure_dd

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"strconv"
	libstrings "strings"

	storage "github.com/Azure/azure-sdk-for-go/arm/storage"
	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/azure"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	defaultFSType             = "ext4"
	defaultStorageAccountType = storage.StandardLRS
)

type dataDisk struct {
	volume.MetricsProvider
	volumeName string
	diskName   string
	podUID     types.UID
}

var (
	supportedCachingModes = sets.NewString(
		string(api.AzureDataDiskCachingNone),
		string(api.AzureDataDiskCachingReadOnly),
		string(api.AzureDataDiskCachingReadWrite))

	supportedDiskKinds = sets.NewString(
		string(api.AzureSharedBlobDisk),
		string(api.AzureDedicatedBlobDisk),
		string(api.AzureManagedDisk))

	supportedStorageAccountTypes = sets.NewString("Premium_LRS", "Standard_LRS")
)

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, strings.EscapeQualifiedNameForDisk(azureDataDiskPluginName), volName)
}

// creates a unique path for disks (even if they share the same *.vhd name)
func makeGlobalPDPath(host volume.VolumeHost, diskUri string, isManaged bool) (string, error) {
	diskUri = libstrings.ToLower(diskUri) // always lower uri because users may enter it in caps.
	uniqueDiskNameTemplate := "%s%s"
	hashedDiskUri := azure.MakeCRC32(diskUri)
	prefix := "b"
	if isManaged {
		prefix = "m"
	}
	// "{m for managed b for blob}{hashed diskUri or DiskId depending on disk kind }"
	diskName := fmt.Sprintf(uniqueDiskNameTemplate, prefix, hashedDiskUri)
	pdPath := path.Join(host.GetPluginDir(azureDataDiskPluginName), mount.MountsInGlobalPDPath, diskName)

	return pdPath, nil
}

func makeDataDisk(volumeName string, podUID types.UID, diskName string, host volume.VolumeHost) *dataDisk {
	var metricProvider volume.MetricsProvider
	if podUID != "" {
		metricProvider = volume.NewMetricsStatFS(getPath(podUID, volumeName, host))
	}

	return &dataDisk{
		MetricsProvider: metricProvider,
		volumeName:      volumeName,
		diskName:        diskName,
		podUID:          podUID,
	}
}

func getVolumeSource(spec *volume.Spec) (*v1.AzureDiskVolumeSource, error) {
	if spec.Volume != nil && spec.Volume.AzureDisk != nil {
		return spec.Volume.AzureDisk, nil
	}

	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.AzureDisk != nil {
		return spec.PersistentVolume.Spec.AzureDisk, nil
	}

	return nil, fmt.Errorf("azureDisk - Spec does not reference an Azure disk volume type")
}

func normalizeFsType(fsType string) string {
	if fsType == "" {
		return defaultFSType
	}

	return fsType
}

func normalizeKind(kind string) (v1.AzureDataDiskKind, error) {
	if kind == "" {
		return v1.AzureDedicatedBlobDisk, nil
	}

	if !supportedDiskKinds.Has(kind) {
		return "", fmt.Errorf("azureDisk - %s is not supported disk kind. Supported values are %s", kind, supportedDiskKinds.List())
	}

	return v1.AzureDataDiskKind(kind), nil
}

func normalizeStorageAccountType(storageAccountType string) (storage.SkuName, error) {
	if storageAccountType == "" {
		return defaultStorageAccountType, nil
	}

	if !supportedStorageAccountTypes.Has(storageAccountType) {
		return "", fmt.Errorf("azureDisk - %s is not supported sku/storageaccounttype. Supported values are %s", storageAccountType, supportedStorageAccountTypes.List())
	}

	return storage.SkuName(storageAccountType), nil
}

func normalizeCachingMode(cachingMode v1.AzureDataDiskCachingMode) (v1.AzureDataDiskCachingMode, error) {
	if cachingMode == "" {
		return v1.AzureDataDiskCachingReadWrite, nil
	}

	if !supportedCachingModes.Has(string(cachingMode)) {
		return "", fmt.Errorf("azureDisk - %s is not supported cachingmode. Supported values are %s", cachingMode, supportedCachingModes.List())
	}

	return cachingMode, nil
}

type ioHandler interface {
	ReadDir(dirname string) ([]os.FileInfo, error)
	WriteFile(filename string, data []byte, perm os.FileMode) error
	Readlink(name string) (string, error)
	ReadFile(filename string) ([]byte, error)
}

//TODO: check if priming the iscsi interface is actually needed

type osIOHandler struct{}

func (handler *osIOHandler) ReadDir(dirname string) ([]os.FileInfo, error) {
	return ioutil.ReadDir(dirname)
}

func (handler *osIOHandler) WriteFile(filename string, data []byte, perm os.FileMode) error {
	return ioutil.WriteFile(filename, data, perm)
}

func (handler *osIOHandler) Readlink(name string) (string, error) {
	return os.Readlink(name)
}

func (handler *osIOHandler) ReadFile(filename string) ([]byte, error) {
	return ioutil.ReadFile(filename)
}

// exclude those used by azure as resource and OS root in /dev/disk/azure
func listAzureDiskPath(io ioHandler) []string {
	azureDiskPath := "/dev/disk/azure/"
	var azureDiskList []string
	if dirs, err := io.ReadDir(azureDiskPath); err == nil {
		for _, f := range dirs {
			name := f.Name()
			diskPath := azureDiskPath + name
			if link, linkErr := io.Readlink(diskPath); linkErr == nil {
				sd := link[(libstrings.LastIndex(link, "/") + 1):]
				azureDiskList = append(azureDiskList, sd)
			}
		}
	}
	glog.V(12).Infof("Azure sys disks paths: %v", azureDiskList)
	return azureDiskList
}

func scsiHostRescan(io ioHandler) {
	scsi_path := "/sys/class/scsi_host/"
	if dirs, err := io.ReadDir(scsi_path); err == nil {
		for _, f := range dirs {
			name := scsi_path + f.Name() + "/scan"
			data := []byte("- - -")
			if err = io.WriteFile(name, data, 0666); err != nil {
				glog.Warningf("failed to rescan scsi host %s", name)
			}
		}
	} else {
		glog.Warningf("failed to read %s, err %v", scsi_path, err)
	}
}

func findDiskByLun(lun int, io ioHandler) (string, error) {
	azureDisks := listAzureDiskPath(io)
	return findDiskByLunWithConstraint(lun, io, azureDisks)
}

// finds a device mounted to "current" node
func findDiskByLunWithConstraint(lun int, io ioHandler, azureDisks []string) (string, error) {
	var err error
	sys_path := "/sys/bus/scsi/devices"
	if dirs, err := io.ReadDir(sys_path); err == nil {
		for _, f := range dirs {
			name := f.Name()
			// look for path like /sys/bus/scsi/devices/3:0:0:1
			arr := libstrings.Split(name, ":")
			if len(arr) < 4 {
				continue
			}
			// extract LUN from the path.
			// LUN is the last index of the array, i.e. 1 in /sys/bus/scsi/devices/3:0:0:1
			l, err := strconv.Atoi(arr[3])
			if err != nil {
				// unknown path format, continue to read the next one
				glog.V(4).Infof("azure disk - failed to parse lun from %v (%v), err %v", arr[3], name, err)
				continue
			}
			if lun == l {
				// find the matching LUN
				// read vendor and model to ensure it is a VHD disk
				vendorPath := path.Join(sys_path, name, "vendor")
				vendorBytes, err := io.ReadFile(vendorPath)
				if err != nil {
					glog.Errorf("failed to read device vendor, err: %v", err)
					continue
				}
				vendor := libstrings.TrimSpace(string(vendorBytes))
				if libstrings.ToUpper(vendor) != "MSFT" {
					glog.V(4).Infof("vendor doesn't match VHD, got %s", vendor)
					continue
				}

				modelPath := path.Join(sys_path, name, "model")
				modelBytes, err := io.ReadFile(modelPath)
				if err != nil {
					glog.Errorf("failed to read device model, err: %v", err)
					continue
				}
				model := libstrings.TrimSpace(string(modelBytes))
				if libstrings.ToUpper(model) != "VIRTUAL DISK" {
					glog.V(4).Infof("model doesn't match VHD, got %s", model)
					continue
				}

				// find a disk, validate name
				dir := path.Join(sys_path, name, "block")
				if dev, err := io.ReadDir(dir); err == nil {
					found := false
					for _, diskName := range azureDisks {
						glog.V(12).Infof("azure disk - validating disk %q with sys disk %q", dev[0].Name(), diskName)
						if string(dev[0].Name()) == diskName {
							found = true
							break
						}
					}
					if !found {
						return "/dev/" + dev[0].Name(), nil
					}
				}
			}
		}
	}
	return "", err
}

func formatIfNotFormatted(disk string, fstype string, exec mount.Exec) {
	notFormatted, err := diskLooksUnformatted(disk, exec)
	if err == nil && notFormatted {
		args := []string{disk}
		// Disk is unformatted so format it.
		// Use 'ext4' as the default
		if len(fstype) == 0 {
			fstype = "ext4"
		}
		if fstype == "ext4" || fstype == "ext3" {
			args = []string{"-E", "lazy_itable_init=0,lazy_journal_init=0", "-F", disk}
		}
		glog.Infof("azureDisk - Disk %q appears to be unformatted, attempting to format as type: %q with options: %v", disk, fstype, args)

		_, err := exec.Run("mkfs."+fstype, args...)
		if err == nil {
			// the disk has been formatted successfully try to mount it again.
			glog.Infof("azureDisk - Disk successfully formatted (mkfs): %s - %s %s", fstype, disk, "tt")
		}
		glog.Warningf("azureDisk - format of disk %q failed: type:(%q) target:(%q) options:(%q)error:(%v)", disk, fstype, "tt", "o", err)
	} else {
		if err != nil {
			glog.Warningf("azureDisk - Failed to check if the disk %s formatted with error %s, will attach anyway", disk, err)
		} else {
			glog.Infof("azureDisk - Disk %s already formatted, will not format", disk)
		}
	}
}

func diskLooksUnformatted(disk string, exec mount.Exec) (bool, error) {
	args := []string{"-nd", "-o", "FSTYPE", disk}
	glog.V(4).Infof("Attempting to determine if disk %q is formatted using lsblk with args: (%v)", disk, args)
	dataOut, err := exec.Run("lsblk", args...)
	if err != nil {
		glog.Errorf("Could not determine if disk %q is formatted (%v)", disk, err)
		return false, err
	}
	output := libstrings.TrimSpace(string(dataOut))
	return output == "", nil
}

func getDiskController(host volume.VolumeHost) (DiskController, error) {
	cloudProvider := host.GetCloudProvider()
	az, ok := cloudProvider.(*azure.Cloud)

	if !ok || az == nil {
		return nil, fmt.Errorf("AzureDisk -  failed to get Azure Cloud Provider. GetCloudProvider returned %v instead", cloudProvider)
	}
	return az, nil
}

func getCloud(host volume.VolumeHost) (*azure.Cloud, error) {
	cloudProvider := host.GetCloudProvider()
	az, ok := cloudProvider.(*azure.Cloud)

	if !ok || az == nil {
		return nil, fmt.Errorf("AzureDisk -  failed to get Azure Cloud Provider. GetCloudProvider returned %v instead", cloudProvider)
	}
	return az, nil
}

func strFirstLetterToUpper(str string) string {
	if len(str) < 2 {
		return str
	}
	return libstrings.ToUpper(string(str[0])) + str[1:]
}
