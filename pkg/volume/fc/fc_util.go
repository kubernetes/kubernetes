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

package fc

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/volume"
)

type ioHandler interface {
	ReadDir(dirname string) ([]os.FileInfo, error)
	Lstat(name string) (os.FileInfo, error)
	EvalSymlinks(path string) (string, error)
	WriteFile(filename string, data []byte, perm os.FileMode) error
}

type osIOHandler struct{}

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

// given a disk path like /dev/sdx, find the devicemapper parent
// TODO #23192 Convert this code to use the generic code in ../util
// which is used by the iSCSI implementation
func findMultipathDeviceMapper(disk string, io ioHandler) string {
	sys_path := "/sys/block/"
	if dirs, err := io.ReadDir(sys_path); err == nil {
		for _, f := range dirs {
			name := f.Name()
			if strings.HasPrefix(name, "dm-") {
				if _, err1 := io.Lstat(sys_path + name + "/slaves/" + disk); err1 == nil {
					return "/dev/" + name
				}
			}
		}
	}
	return ""
}

// given a wwn and lun, find the device and associated devicemapper parent
func findDisk(wwn, lun string, io ioHandler) (string, string) {
	fc_path := "-fc-0x" + wwn + "-lun-" + lun
	dev_path := "/dev/disk/by-path/"
	if dirs, err := io.ReadDir(dev_path); err == nil {
		for _, f := range dirs {
			name := f.Name()
			if strings.Contains(name, fc_path) {
				if disk, err1 := io.EvalSymlinks(dev_path + name); err1 == nil {
					arr := strings.Split(disk, "/")
					l := len(arr) - 1
					dev := arr[l]
					dm := findMultipathDeviceMapper(dev, io)
					return disk, dm
				}
			}
		}
	}
	return "", ""
}

// given a wwid, find the device and associated devicemapper parent
func findDiskWWIDs(wwid string, io ioHandler) (string, string) {
	// Example wwid format:
	//   3600508b400105e210000900000490000
	//   <VENDOR NAME> <IDENTIFIER NUMBER>
	// Example of symlink under by-id:
	//   /dev/by-id/scsi-3600508b400105e210000900000490000
	//   /dev/by-id/scsi-<VENDOR NAME>_<IDENTIFIER NUMBER>
	// The wwid could contain white space and it will be replaced
	// underscore when wwid is exposed under /dev/by-id.

	fc_path := "scsi-" + wwid
	dev_id := "/dev/disk/by-id/"
	if dirs, err := io.ReadDir(dev_id); err == nil {
		for _, f := range dirs {
			name := f.Name()
			if name == fc_path {
				disk, err := io.EvalSymlinks(dev_id + name)
				if err != nil {
					glog.V(2).Infof("fc: failed to find a corresponding disk from symlink[%s], error %v", dev_id+name, err)
					return "", ""
				}
				arr := strings.Split(disk, "/")
				l := len(arr) - 1
				dev := arr[l]
				dm := findMultipathDeviceMapper(dev, io)
				return disk, dm
			}
		}
	}
	glog.V(2).Infof("fc: failed to find a disk [%s]", dev_id+fc_path)
	return "", ""
}

// Removes a scsi device based upon /dev/sdX name
func removeFromScsiSubsystem(deviceName string, io ioHandler) {
	fileName := "/sys/block/" + deviceName + "/device/delete"
	glog.V(4).Infof("fc: remove device from scsi-subsystem: path: %s", fileName)
	data := []byte("1")
	io.WriteFile(fileName, data, 0666)
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

// make a directory like /var/lib/kubelet/plugins/kubernetes.io/pod/fc/target-lun-0
func makePDNameInternal(host volume.VolumeHost, wwns []string, lun string, wwids []string) string {
	if len(wwns) != 0 {
		return path.Join(host.GetPluginDir(fcPluginName), wwns[0]+"-lun-"+lun)
	} else {
		return path.Join(host.GetPluginDir(fcPluginName), wwids[0])
	}
}

type FCUtil struct{}

func (util *FCUtil) MakeGlobalPDName(fc fcDisk) string {
	return makePDNameInternal(fc.plugin.host, fc.wwns, fc.lun, fc.wwids)
}

func searchDisk(b fcDiskMounter) (string, string) {
	var diskIds []string
	var disk string
	var dm string
	io := b.io
	wwids := b.wwids
	wwns := b.wwns
	lun := b.lun

	if len(wwns) != 0 {
		diskIds = wwns
	} else {
		diskIds = wwids
	}

	rescaned := false
	// two-phase search:
	// first phase, search existing device path, if a multipath dm is found, exit loop
	// otherwise, in second phase, rescan scsi bus and search again, return with any findings
	for true {
		for _, diskId := range diskIds {
			if len(wwns) != 0 {
				disk, dm = findDisk(diskId, lun, io)
			} else {
				disk, dm = findDiskWWIDs(diskId, io)
			}
			// if multipath device is found, break
			if dm != "" {
				break
			}
		}
		// if a dm is found, exit loop
		if rescaned || dm != "" {
			break
		}
		// rescan and search again
		// rescan scsi bus
		scsiHostRescan(io)
		rescaned = true
	}
	return disk, dm
}

func (util *FCUtil) AttachDisk(b fcDiskMounter) (string, error) {
	devicePath := ""
	var disk, dm string

	disk, dm = searchDisk(b)
	// if no disk matches input wwn and lun, exit
	if disk == "" && dm == "" {
		return "", fmt.Errorf("no fc disk found")
	}

	// if multipath devicemapper device is found, use it; otherwise use raw disk
	if dm != "" {
		devicePath = dm
	} else {
		devicePath = disk
	}
	// mount it
	globalPDPath := util.MakeGlobalPDName(*b.fcDisk)
	if err := os.MkdirAll(globalPDPath, 0750); err != nil {
		return devicePath, fmt.Errorf("fc: failed to mkdir %s, error", globalPDPath)
	}

	noMnt, err := b.mounter.IsLikelyNotMountPoint(globalPDPath)
	if err != nil {
		return devicePath, fmt.Errorf("Heuristic determination of mount point failed:%v", err)
	}
	if !noMnt {
		glog.Infof("fc: %s already mounted", globalPDPath)
		return devicePath, nil
	}

	err = b.mounter.FormatAndMount(devicePath, globalPDPath, b.fsType, nil)
	if err != nil {
		return devicePath, fmt.Errorf("fc: failed to mount fc volume %s [%s] to %s, error %v", devicePath, b.fsType, globalPDPath, err)
	}

	return devicePath, err
}

func (util *FCUtil) DetachDisk(c fcDiskUnmounter, devName string) error {
	// Remove scsi device from the node.
	if !strings.HasPrefix(devName, "/dev/") {
		return fmt.Errorf("fc detach disk: invalid device name: %s", devName)
	}
	arr := strings.Split(devName, "/")
	dev := arr[len(arr)-1]
	removeFromScsiSubsystem(dev, c.io)
	return nil
}
