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
	"io/ioutil"
	"os"
	"path"
	"regexp"
	"strconv"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/exec"
)

type ioHandler interface {
	ReadDir(dirname string) ([]os.FileInfo, error)
	WriteFile(filename string, data []byte, perm os.FileMode) error
	Readlink(name string) (string, error)
}

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
				sd := link[(strings.LastIndex(link, "/") + 1):]
				azureDiskList = append(azureDiskList, sd)
			}
		}
	}
	glog.V(12).Infof("Azure sys disks paths: %v", azureDiskList)
	return azureDiskList
}

// given a LUN find the VHD device path like /dev/sdd
// exclude those disks used by Azure resources and OS root
func findDiskByLun(lun int, io ioHandler, exe exec.Interface) (string, error) {
	azureDisks := listAzureDiskPath(io)
	return findDiskByLunWithConstraint(lun, io, exe, azureDisks)
}

// look for device /dev/sdX and validate it is a VHD
// return empty string if no disk is found
func findDiskByLunWithConstraint(lun int, io ioHandler, exe exec.Interface, azureDisks []string) (string, error) {
	var err error
	sys_path := "/sys/bus/scsi/devices"
	if dirs, err := io.ReadDir(sys_path); err == nil {
		for _, f := range dirs {
			name := f.Name()
			// look for path like /sys/bus/scsi/devices/3:0:0:1
			arr := strings.Split(name, ":")
			if len(arr) < 4 {
				continue
			}
			// extract LUN from the path.
			// LUN is the last index of the array, i.e. 1 in /sys/bus/scsi/devices/3:0:0:1
			l, err := strconv.Atoi(arr[3])
			if err != nil {
				// unknown path format, continue to read the next one
				glog.Errorf("failed to parse lun from %v (%v), err %v", arr[3], name, err)
				continue
			}
			if lun == l {
				// find the matching LUN
				// read vendor and model to ensure it is a VHD disk
				vendor := path.Join(sys_path, name, "vendor")
				model := path.Join(sys_path, name, "model")
				out, err := exe.Command("cat", vendor, model).CombinedOutput()
				if err != nil {
					glog.Errorf("failed to cat device vendor and model, err: %v", err)
					continue
				}
				matched, err := regexp.MatchString("^MSFT[ ]{0,}\nVIRTUAL DISK[ ]{0,}\n$", strings.ToUpper(string(out)))
				if err != nil || !matched {
					glog.V(4).Infof("doesn't match VHD, output %v, error %v", string(out), err)
					continue
				}
				// find a disk, validate name
				dir := path.Join(sys_path, name, "block")
				if dev, err := io.ReadDir(dir); err == nil {
					found := false
					for _, diskName := range azureDisks {
						glog.V(12).Infof("validating disk %q with sys disk %q", dev[0].Name(), diskName)
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

// rescan scsi bus
func scsiHostRescan(io ioHandler) {
	scsi_path := "/sys/class/scsi_host/"
	if dirs, err := io.ReadDir(scsi_path); err == nil {
		for _, f := range dirs {
			name := scsi_path + f.Name() + "/scan"
			data := []byte("- - -")
			if err = io.WriteFile(name, data, 0666); err != nil {
				glog.Errorf("failed to rescan scsi host %s", name)
			}
		}
	} else {
		glog.Errorf("failed to read %s, err %v", scsi_path, err)
	}
}
