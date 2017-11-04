// +build linux

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
	"path"
	"strconv"
	libstrings "strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/mount"
)

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

func scsiHostRescan(io ioHandler, exec mount.Exec) {
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

func findDiskByLun(lun int, io ioHandler, exec mount.Exec) (string, error) {
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
			if len(azureDisks) == 0 {
				glog.V(4).Infof("/dev/disk/azure is not populated, now try to parse %v directly", name)
				target, err := strconv.Atoi(arr[0])
				if err != nil {
					glog.Errorf("failed to parse target from %v (%v), err %v", arr[0], name, err)
					continue
				}
				// as observed, targets 0-3 are used by OS disks. Skip them
				if target <= 3 {
					continue
				}
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
			glog.Infof("azureDisk - Disk successfully formatted with 'mkfs.%s %v'", fstype, args)
		} else {
			glog.Warningf("azureDisk - Error formatting volume with 'mkfs.%s %v': %v", fstype, args, err)
		}
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
