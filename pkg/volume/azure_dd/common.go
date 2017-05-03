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

package azure_dd

import (
	"fmt"
	"hash/crc32"
	"io/ioutil"
	"net/url"
	"os"
	"path"
	"regexp"
	"strconv"
	STRINGS "strings"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	default_fstype               = "ext4"
	default_storage_account_type = "standard_lrs"
)

type dataDisk struct {
	volume.MetricsProvider
	volumeName string
	diskName   string
	podUID     types.UID
}

var supportedCachingModes = sets.NewString(
	string(api.AzureDataDiskCachingNone),
	string(api.AzureDataDiskCachingReadOnly),
	string(api.AzureDataDiskCachingReadWrite),
)

var supportedDiskKinds = sets.NewString(
	string(api.AzureSharedBlobDisk),
	string(api.AzureDedicatedBlobDisk),
	string(api.AzureManagedDisk),
)

var supportedStorageAccountTypes = sets.NewString("premium_lrs", "standard_lrs")
var polyTable *crc32.Table = crc32.MakeTable(crc32.Koopman)

func makeCRC32(str string) string {
	crc := crc32.New(polyTable)
	crc.Write([]byte(str))
	hash := crc.Sum32()
	return strconv.FormatUint(uint64(hash), 10)
}

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, strings.EscapeQualifiedNameForDisk(azureDataDiskPluginName), volName)
}

// creates a unique path for disks (even if they share the same *.vhd name)
func makeGlobalPDPath(host volume.VolumeHost, diskUri string, isManaged bool) (string, error) {
	diskUri = STRINGS.ToLower(diskUri) // always lower uri because users may enter it in caps.
	uniqueDiskNameTemplate := "%s%s"
	hashedDiskUri := makeCRC32(diskUri)
	prefix := "b"
	if isManaged {
		prefix = "m"
	}
	// "{m for managed b for blob}{hashed diskUri or DiskId depending on disk kind }"
	diskName := fmt.Sprintf(uniqueDiskNameTemplate, prefix, hashedDiskUri)
	pdPath := path.Join(host.GetPluginDir(azureDataDiskPluginName), mount.MountsInGlobalPDPath, diskName)

	return pdPath, nil
}

func diskKindHashfromPDName(diskName string) (bool, string) {
	diskKind := diskName[0:1]
	diskHash := diskName[1:]

	return (diskKind == "m"), diskHash
}

func diskNameandSANameFromUri(diskUri string) (string, string, error) {
	uri, err := url.Parse(diskUri)
	if err != nil {
		return "", "", err
	}

	hostName := uri.Host
	storageAccountName := STRINGS.Split(hostName, ".")[0]

	segments := STRINGS.Split(uri.Path, "/")
	diskNameVhd := segments[len(segments)-1]

	return storageAccountName, diskNameVhd, nil
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
		return default_fstype
	}

	return fsType
}

func normalizeKind(kind v1.AzureDataDiskKind) (v1.AzureDataDiskKind, error) {
	if kind == "" {
		return v1.AzureSharedBlobDisk, nil
	}

	if !supportedDiskKinds.Has(string(kind)) {
		return "", fmt.Errorf("azureDisk - %s is not supported disk kind. Supported values are %s", kind, supportedDiskKinds.List())
	}

	return v1.AzureDataDiskKind(kind), nil
}

func normalizeStorageAccountType(storageAccountType string) (string, error) {
	if storageAccountType == "" {
		return default_storage_account_type, nil
	}

	if !supportedStorageAccountTypes.Has(storageAccountType) {
		return "", fmt.Errorf("azureDisk - %s is not supported sku/storageaccounttype. Supported values are %s", storageAccountType, supportedStorageAccountTypes.List())
	}

	return storageAccountType, nil
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

// exclude those used by azure as resource and OS root in /dev/disk/azure
func listAzureDiskPath(io ioHandler) []string {
	azureDiskPath := "/dev/disk/azure/"
	var azureDiskList []string
	if dirs, err := io.ReadDir(azureDiskPath); err == nil {
		for _, f := range dirs {
			name := f.Name()
			diskPath := azureDiskPath + name
			if link, linkErr := io.Readlink(diskPath); linkErr == nil {
				sd := link[(STRINGS.LastIndex(link, "/") + 1):]
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

func findDiskByLun(lun int, io ioHandler, exe exec.Interface) (string, error) {
	azureDisks := listAzureDiskPath(io)
	return findDiskByLunWithConstraint(lun, io, exe, azureDisks)
}

// finds a device mounted to "current" node
func findDiskByLunWithConstraint(lun int, io ioHandler, exe exec.Interface, azureDisks []string) (string, error) {
	var err error
	sys_path := "/sys/bus/scsi/devices"
	if dirs, err := io.ReadDir(sys_path); err == nil {
		for _, f := range dirs {
			name := f.Name()
			// look for path like /sys/bus/scsi/devices/3:0:0:1
			arr := STRINGS.Split(name, ":")
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
				vendor := path.Join(sys_path, name, "vendor")
				model := path.Join(sys_path, name, "model")
				out, err := exe.Command("cat", vendor, model).CombinedOutput()
				if err != nil {
					glog.V(4).Infof("azure disk - failed to cat device vendor and model, err: %v", err)
					continue
				}
				matched, err := regexp.MatchString("^MSFT[ ]{0,}\nVIRTUAL DISK[ ]{0,}\n$", STRINGS.ToUpper(string(out)))
				if err != nil || !matched {
					glog.V(4).Infof("azure disk - doesn't match VHD, output %v, error %v", string(out), err)
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

func formatIfNotFormatted(disk string, fstype string) {
	notFormatted, err := diskLooksUnformatted(disk)
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
		runner := exec.New()
		cmd := runner.Command("mkfs."+fstype, args...)
		_, err := cmd.CombinedOutput()
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

func diskLooksUnformatted(disk string) (bool, error) {
	args := []string{"-nd", "-o", "FSTYPE", disk}
	runner := exec.New()
	cmd := runner.Command("lsblk", args...)
	glog.V(4).Infof("Attempting to determine if disk %q is formatted using lsblk with args: (%v)", disk, args)
	dataOut, err := cmd.CombinedOutput()
	if err != nil {
		glog.Errorf("Could not determine if disk %q is formatted (%v)", disk, err)
		return false, err
	}
	output := STRINGS.TrimSpace(string(dataOut))
	return output == "", nil
}
