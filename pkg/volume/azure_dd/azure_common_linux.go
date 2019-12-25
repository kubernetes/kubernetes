// +build !providerless
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
	"fmt"
	"path/filepath"
	"strconv"
	libstrings "strings"

	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/util/mount"
)

// exclude those used by azure as resource and OS root in /dev/disk/azure, /dev/disk/azure/scsi0
// "/dev/disk/azure/scsi0" dir is populated in Standard_DC4s/DC2s on Ubuntu 18.04
func listAzureDiskPath(io ioHandler) []string {
	var azureDiskList []string
	azureResourcePaths := []string{"/dev/disk/azure/", "/dev/disk/azure/scsi0/"}
	for _, azureDiskPath := range azureResourcePaths {
		if dirs, err := io.ReadDir(azureDiskPath); err == nil {
			for _, f := range dirs {
				name := f.Name()
				diskPath := filepath.Join(azureDiskPath, name)
				if link, linkErr := io.Readlink(diskPath); linkErr == nil {
					sd := link[(libstrings.LastIndex(link, "/") + 1):]
					azureDiskList = append(azureDiskList, sd)
				}
			}
		}
	}
	klog.V(12).Infof("Azure sys disks paths: %v", azureDiskList)
	return azureDiskList
}

// getDiskLinkByDevName get disk link by device name from devLinkPath, e.g. /dev/disk/azure/, /dev/disk/by-id/
func getDiskLinkByDevName(io ioHandler, devLinkPath, devName string) (string, error) {
	dirs, err := io.ReadDir(devLinkPath)
	klog.V(12).Infof("azureDisk - begin to find %s from %s", devName, devLinkPath)
	if err == nil {
		for _, f := range dirs {
			diskPath := devLinkPath + f.Name()
			klog.V(12).Infof("azureDisk - begin to Readlink: %s", diskPath)
			link, linkErr := io.Readlink(diskPath)
			if linkErr != nil {
				klog.Warningf("azureDisk - read link (%s) error: %v", diskPath, linkErr)
				continue
			}
			if libstrings.HasSuffix(link, devName) {
				return diskPath, nil
			}
		}
		return "", fmt.Errorf("device name(%s) is not found under %s", devName, devLinkPath)
	}
	return "", fmt.Errorf("read %s error: %v", devLinkPath, err)
}

func scsiHostRescan(io ioHandler, exec mount.Exec) {
	scsi_path := "/sys/class/scsi_host/"
	if dirs, err := io.ReadDir(scsi_path); err == nil {
		for _, f := range dirs {
			name := scsi_path + f.Name() + "/scan"
			data := []byte("- - -")
			if err = io.WriteFile(name, data, 0666); err != nil {
				klog.Warningf("failed to rescan scsi host %s", name)
			}
		}
	} else {
		klog.Warningf("failed to read %s, err %v", scsi_path, err)
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
				klog.V(4).Infof("/dev/disk/azure is not populated, now try to parse %v directly", name)
				target, err := strconv.Atoi(arr[0])
				if err != nil {
					klog.Errorf("failed to parse target from %v (%v), err %v", arr[0], name, err)
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
				klog.V(4).Infof("azure disk - failed to parse lun from %v (%v), err %v", arr[3], name, err)
				continue
			}
			if lun == l {
				// find the matching LUN
				// read vendor and model to ensure it is a VHD disk
				vendorPath := filepath.Join(sys_path, name, "vendor")
				vendorBytes, err := io.ReadFile(vendorPath)
				if err != nil {
					klog.Errorf("failed to read device vendor, err: %v", err)
					continue
				}
				vendor := libstrings.TrimSpace(string(vendorBytes))
				if libstrings.ToUpper(vendor) != "MSFT" {
					klog.V(4).Infof("vendor doesn't match VHD, got %s", vendor)
					continue
				}

				modelPath := filepath.Join(sys_path, name, "model")
				modelBytes, err := io.ReadFile(modelPath)
				if err != nil {
					klog.Errorf("failed to read device model, err: %v", err)
					continue
				}
				model := libstrings.TrimSpace(string(modelBytes))
				if libstrings.ToUpper(model) != "VIRTUAL DISK" {
					klog.V(4).Infof("model doesn't match VIRTUAL DISK, got %s", model)
					continue
				}

				// find a disk, validate name
				dir := filepath.Join(sys_path, name, "block")
				if dev, err := io.ReadDir(dir); err == nil {
					found := false
					devName := dev[0].Name()
					for _, diskName := range azureDisks {
						klog.V(12).Infof("azureDisk - validating disk %q with sys disk %q", devName, diskName)
						if devName == diskName {
							found = true
							break
						}
					}
					if !found {
						devLinkPaths := []string{"/dev/disk/azure/scsi1/", "/dev/disk/by-id/"}
						for _, devLinkPath := range devLinkPaths {
							diskPath, err := getDiskLinkByDevName(io, devLinkPath, devName)
							if err == nil {
								klog.V(4).Infof("azureDisk - found %s by %s under %s", diskPath, devName, devLinkPath)
								return diskPath, nil
							}
							klog.Warningf("azureDisk - getDiskLinkByDevName by %s under %s failed, error: %v", devName, devLinkPath, err)
						}
						return "/dev/" + devName, nil
					}
				}
			}
		}
	}
	return "", err
}
