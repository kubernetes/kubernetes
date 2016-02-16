/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package util

import (
	"errors"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)

//IoUtil is a util for common IO operations
//it also backports certain operations from golang 1.5
type IoUtil interface {
	ReadDir(dirname string) ([]os.FileInfo, error)
	Lstat(name string) (os.FileInfo, error)
	EvalSymlinks(path string) (string, error)
	WriteFile(filename string, data []byte, perm os.FileMode) error
	FindMultipathDeviceForDevice(disk string) string
	FindDevicesForMultipathDevice(disk string) []string
}

type osIOHandler struct{}

//NewIOHandler Create a new IoHandler implementation
func NewIOHandler() IoUtil {
	return &osIOHandler{}
}

func (handler *osIOHandler) ReadDir(dirname string) ([]os.FileInfo, error) {
	return ioutil.ReadDir(dirname)
}
func (handler *osIOHandler) Lstat(name string) (os.FileInfo, error) {
	return os.Lstat(name)
}
func (handler *osIOHandler) EvalSymlinks(path string) (string, error) {
	return filepath.EvalSymlinks(path)
}
func (handler *osIOHandler) WriteFile(filename string, data []byte, perm os.FileMode) error {
	return ioutil.WriteFile(filename, data, perm)
}

//FindMultipathDeviceForDevice given a device name like /dev/sdx, find the devicemapper parent
func (handler *osIOHandler) FindMultipathDeviceForDevice(device string) string {
	disk, err := handler.FindDeviceForPath(device)
	if err != nil {
		return ""
	}
	sysPath := "/sys/block/"
	if dirs, err := handler.ReadDir(sysPath); err == nil {
		for _, f := range dirs {
			name := f.Name()
			if strings.HasPrefix(name, "dm-") {
				if _, err1 := handler.Lstat(sysPath + name + "/slaves/" + disk); err1 == nil {
					return "/dev/" + name
				}
			}
		}
	}
	return ""
}

//FindDevicesForMultipathDevice given a disk name of /dev/dm-XX return a
//slice including all the mapped devices
//if the device isn't a mapped device return an empty slice
func (handler *osIOHandler) FindDevicesForMultipathDevice(device string) []string {
	disk, err := handler.FindDeviceForPath(device)
	if err != nil {
		return make([]string, 0, 0)
	}
	sysPath := "/sys/block/" + disk + "/slaves/"
	if dirs, err := handler.ReadDir(sysPath); err == nil {
		devs := make([]string, len(dirs), len(dirs))
		for i, f := range dirs {
			devs[i] = f.Name()
		}
		return devs
	}
	return make([]string, 0, 0)
}

//FindDeviceForVirtual Find the underlaying disk for a linked path such as /dev/disk/by-path/XXXX or /dev/mapper/XXXX
// will return sdX or hdX etc, if /dev/sdX is passed in then sdX will be returned
func (handler *osIOHandler) FindDeviceForPath(path string) (string, error) {
	parts := strings.Split(path, "/")
	//if path /dev/hdX splits into "", "dev", "hdX" then we will
	//return just the last part
	if len(parts) == 3 && strings.HasPrefix(parts[1], "dev") {
		return parts[2], nil
	}
	devicePath, err := handler.EvalSymlinks(path)
	if err != nil {
		return "", err
	}
	parts = strings.Split(devicePath, "/")
	if len(parts) == 3 && strings.HasPrefix(parts[1], "dev") {
		return parts[2], nil
	}
	return "", errors.New("Illegal path for device " + devicePath)
}
