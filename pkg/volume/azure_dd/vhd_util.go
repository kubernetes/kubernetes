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

	"k8s.io/kubernetes/pkg/util/exec"
)

type ioHandler interface {
	ReadDir(dirname string) ([]os.FileInfo, error)
	WriteFile(filename string, data []byte, perm os.FileMode) error
}

type osIOHandler struct{}

func (handler *osIOHandler) ReadDir(dirname string) ([]os.FileInfo, error) {
	return ioutil.ReadDir(dirname)
}
func (handler *osIOHandler) WriteFile(filename string, data []byte, perm os.FileMode) error {
	return ioutil.WriteFile(filename, data, perm)
}

// given a LUN find the VHD device path like /dev/sdb
// VHD disks under sysfs are like /sys/bus/scsi/devices/3:0:1:0
// return empty string if no disk is found
func findDiskByLun(lun int, io ioHandler) string {
	sys_path := "/sys/bus/scsi/devices"
	if dirs, err := io.ReadDir(sys_path); err == nil {
		for _, f := range dirs {
			name := f.Name()
			// look for path like /sys/bus/scsi/devices/3:0:1:0
			arr := strings.Split(name, ":")
			if len(arr) < 4 {
				continue
			}
			target, err := strconv.Atoi(arr[0])
			// as observed, targets 0-3 are used by OS disks. Skip them
			if err == nil && target > 3 {
				l, err := strconv.Atoi(arr[3])
				if err == nil && lun == l {
					// find the matching LUN
					// read vendor and model to ensure it is a VHD disk
					vendor := path.Join(sys_path, name, "vendor")
					model := path.Join(sys_path, name, "model")
					exe := exec.New()
					out, err := exe.Command("cat", vendor, model).CombinedOutput()
					if err != nil {
						continue
					}
					matched, err := regexp.MatchString("^MSFT[ ]{0,}\nVIRTUAL DISK[ ]{0,}\n$", strings.ToUpper(string(out)))
					if err != nil || !matched {
						continue
					}
					// find it!
					if dev, err := io.ReadDir(path.Join(sys_path, name, "block")); err == nil {
						return "/dev/" + dev[0].Name()
					}
				}
			}
		}
	}
	return ""
}

// rescan scsi bus
func scsiHostRescan(io ioHandler) {
	scsi_path := "/sys/class/scsi_host/"
	if dirs, err := io.ReadDir(scsi_path); err == nil {
		for _, f := range dirs {
			name := scsi_path + f.Name() + "/scan"
			data := []byte("- - -")
			io.WriteFile(name, data, 0666)
		}
	}
}
