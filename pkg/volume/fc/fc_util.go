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
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"k8s.io/klog/v2"
	"k8s.io/mount-utils"
	utilexec "k8s.io/utils/exec"

	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

type ioHandler interface {
	ReadDir(dirname string) ([]os.FileInfo, error)
	Lstat(name string) (os.FileInfo, error)
	EvalSymlinks(path string) (string, error)
	WriteFile(filename string, data []byte, perm os.FileMode) error
}

type osIOHandler struct{}

const (
	byPath = "/dev/disk/by-path/"
	byID   = "/dev/disk/by-id/"
)

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

// given a wwn and lun, find the device and associated devicemapper parent
func findDisk(wwn, lun string, io ioHandler, deviceUtil volumeutil.DeviceUtil) (string, string) {
	fcPathExp := "^(pci-.*-fc|fc)-0x" + wwn + "-lun-" + lun + "$"
	r := regexp.MustCompile(fcPathExp)
	devPath := byPath
	if dirs, err := io.ReadDir(devPath); err == nil {
		for _, f := range dirs {
			name := f.Name()
			if r.MatchString(name) {
				if disk, err1 := io.EvalSymlinks(devPath + name); err1 == nil {
					dm := deviceUtil.FindMultipathDeviceForDevice(disk)
					klog.Infof("fc: find disk: %v, dm: %v, fc path: %v", disk, dm, name)
					return disk, dm
				}
			}
		}
	}
	return "", ""
}

// given a wwid, find the device and associated devicemapper parent
func findDiskWWIDs(wwid string, io ioHandler, deviceUtil volumeutil.DeviceUtil) (string, string) {
	// Example wwid format:
	//   3600508b400105e210000900000490000
	//   <VENDOR NAME> <IDENTIFIER NUMBER>
	// Example of symlink under by-id:
	//   /dev/by-id/scsi-3600508b400105e210000900000490000
	//   /dev/by-id/scsi-<VENDOR NAME>_<IDENTIFIER NUMBER>
	// The wwid could contain white space and it will be replaced
	// underscore when wwid is exposed under /dev/by-id.

	fcPath := "scsi-" + wwid
	devID := byID
	if dirs, err := io.ReadDir(devID); err == nil {
		for _, f := range dirs {
			name := f.Name()
			if name == fcPath {
				disk, err := io.EvalSymlinks(devID + name)
				if err != nil {
					klog.V(2).Infof("fc: failed to find a corresponding disk from symlink[%s], error %v", devID+name, err)
					return "", ""
				}
				dm := deviceUtil.FindMultipathDeviceForDevice(disk)
				klog.Infof("fc: find disk: %v, dm: %v", disk, dm)
				return disk, dm
			}
		}
	}
	klog.V(2).Infof("fc: failed to find a disk [%s]", devID+fcPath)
	return "", ""
}

// Flushes any outstanding I/O to the device
func flushDevice(deviceName string, exec utilexec.Interface) {
	out, err := exec.Command("blockdev", "--flushbufs", deviceName).CombinedOutput()
	if err != nil {
		// Ignore the error and continue deleting the device. There is will be no retry on error.
		klog.Warningf("Failed to flush device %s: %s\n%s", deviceName, err, string(out))
	}
	klog.V(4).Infof("Flushed device %s", deviceName)
}

// Removes a scsi device based upon /dev/sdX name
func removeFromScsiSubsystem(deviceName string, io ioHandler) {
	fileName := "/sys/block/" + deviceName + "/device/delete"
	klog.V(4).Infof("fc: remove device from scsi-subsystem: path: %s", fileName)
	data := []byte("1")
	io.WriteFile(fileName, data, 0666)
}

// rescan scsi bus
func scsiHostRescan(io ioHandler) {
	scsiPath := "/sys/class/scsi_host/"
	if dirs, err := io.ReadDir(scsiPath); err == nil {
		for _, f := range dirs {
			name := scsiPath + f.Name() + "/scan"
			data := []byte("- - -")
			io.WriteFile(name, data, 0666)
		}
	}
}

// make a directory like /var/lib/kubelet/plugins/kubernetes.io/fc/target1-target2-lun-0
func makePDNameInternal(host volume.VolumeHost, wwns []string, lun string, wwids []string) string {
	if len(wwns) != 0 {
		w := strings.Join(wwns, "-")
		return filepath.Join(host.GetPluginDir(fcPluginName), w+"-lun-"+lun)
	}
	return filepath.Join(host.GetPluginDir(fcPluginName), strings.Join(wwids, "-"))
}

// make a directory like /var/lib/kubelet/plugins/kubernetes.io/fc/volumeDevices/target-lun-0
func makeVDPDNameInternal(host volume.VolumeHost, wwns []string, lun string, wwids []string) string {
	if len(wwns) != 0 {
		w := strings.Join(wwns, "-")
		return filepath.Join(host.GetVolumeDevicePluginDir(fcPluginName), w+"-lun-"+lun)
	}
	return filepath.Join(host.GetVolumeDevicePluginDir(fcPluginName), strings.Join(wwids, "-"))
}

func parsePDName(path string) (wwns []string, lun int32, wwids []string, err error) {
	// parse directory name created by makePDNameInternal or makeVDPDNameInternal
	dirname := filepath.Base(path)
	components := strings.Split(dirname, "-")
	l := len(components)
	if l == 1 {
		// No '-', it must be single WWID
		return nil, 0, components, nil
	}
	if components[l-2] == "lun" {
		// it has -lun-, it's list of WWNs + lun number as the last component
		if l == 2 {
			return nil, 0, nil, fmt.Errorf("no wwn in: %s", dirname)
		}
		lun, err := strconv.Atoi(components[l-1])
		if err != nil {
			return nil, 0, nil, err
		}

		return components[:l-2], int32(lun), nil, nil
	}
	// no -lun-, it's just list of WWIDs
	return nil, 0, components, nil
}

type fcUtil struct{}

func (util *fcUtil) MakeGlobalPDName(fc fcDisk) string {
	return makePDNameInternal(fc.plugin.host, fc.wwns, fc.lun, fc.wwids)
}

// Global volume device plugin dir
func (util *fcUtil) MakeGlobalVDPDName(fc fcDisk) string {
	return makeVDPDNameInternal(fc.plugin.host, fc.wwns, fc.lun, fc.wwids)
}

func searchDisk(b fcDiskMounter) (string, error) {
	var diskIDs []string
	var disk string
	var dm string
	io := b.io
	wwids := b.wwids
	wwns := b.wwns
	lun := b.lun

	if len(wwns) != 0 {
		diskIDs = wwns
	} else {
		diskIDs = wwids
	}

	rescanned := false
	// two-phase search:
	// first phase, search existing device path, if a multipath dm is found, exit loop
	// otherwise, in second phase, rescan scsi bus and search again, return with any findings
	for true {
		for _, diskID := range diskIDs {
			if len(wwns) != 0 {
				disk, dm = findDisk(diskID, lun, io, b.deviceUtil)
			} else {
				disk, dm = findDiskWWIDs(diskID, io, b.deviceUtil)
			}
			// if multipath device is found, break
			if dm != "" {
				break
			}
		}
		// if a dm is found, exit loop
		if rescanned || dm != "" {
			break
		}
		// rescan and search again
		// rescan scsi bus
		scsiHostRescan(io)
		rescanned = true
	}
	// if no disk matches input wwn and lun, exit
	if disk == "" && dm == "" {
		return "", fmt.Errorf("no fc disk found")
	}

	// if multipath devicemapper device is found, use it; otherwise use raw disk
	if dm != "" {
		return dm, nil
	}
	return disk, nil
}

func (util *fcUtil) AttachDisk(b fcDiskMounter) (string, error) {
	devicePath, err := searchDisk(b)
	if err != nil {
		return "", err
	}

	exists, err := mount.PathExists(devicePath)
	if exists && err == nil {
		return devicePath, nil
	}
	if exists == false {
		return "", fmt.Errorf("device %s does not exist", devicePath)
	} else {
		return "", err
	}
}

// DetachDisk removes scsi device file such as /dev/sdX from the node.
func (util *fcUtil) DetachDisk(c fcDiskUnmounter, devicePath string) error {
	var devices []string
	// devicePath might be like /dev/mapper/mpathX. Find destination.
	dstPath, err := c.io.EvalSymlinks(devicePath)
	if err != nil {
		return err
	}
	// Find slave
	if strings.HasPrefix(dstPath, "/dev/dm-") {
		devices = c.deviceUtil.FindSlaveDevicesOnMultipath(dstPath)
		if err := util.deleteMultipathDevice(c.exec, dstPath); err != nil {
			return err
		}
	} else {
		// Add single devicepath to devices
		devices = append(devices, dstPath)
	}
	klog.V(4).Infof("fc: DetachDisk devicePath: %v, dstPath: %v, devices: %v", devicePath, dstPath, devices)
	var lastErr error
	for _, device := range devices {
		err := util.detachFCDisk(c.io, c.exec, device)
		if err != nil {
			klog.Errorf("fc: detachFCDisk failed. device: %v err: %v", device, err)
			lastErr = fmt.Errorf("fc: detachFCDisk failed. device: %v err: %v", device, err)
		}
	}
	if lastErr != nil {
		klog.Errorf("fc: last error occurred during detach disk:\n%v", lastErr)
		return lastErr
	}
	return nil
}

// detachFCDisk removes scsi device file such as /dev/sdX from the node.
func (util *fcUtil) detachFCDisk(io ioHandler, exec utilexec.Interface, devicePath string) error {
	// Remove scsi device from the node.
	if !strings.HasPrefix(devicePath, "/dev/") {
		return fmt.Errorf("fc detach disk: invalid device name: %s", devicePath)
	}
	flushDevice(devicePath, exec)
	arr := strings.Split(devicePath, "/")
	dev := arr[len(arr)-1]
	removeFromScsiSubsystem(dev, io)
	return nil
}

// DetachBlockFCDisk detaches a volume from kubelet node, removes scsi device file
// such as /dev/sdX from the node, and then removes loopback for the scsi device.
func (util *fcUtil) DetachBlockFCDisk(c fcDiskUnmapper, mapPath, devicePath string) error {
	// Check if devicePath is valid
	if len(devicePath) != 0 {
		if pathExists, pathErr := checkPathExists(devicePath); !pathExists || pathErr != nil {
			return pathErr
		}
	} else {
		// TODO: FC plugin can't obtain the devicePath from kubelet because devicePath
		// in volume object isn't updated when volume is attached to kubelet node.
		klog.Infof("fc: devicePath is empty. Try to retrieve FC configuration from global map path: %v", mapPath)
	}

	// Check if global map path is valid
	// global map path examples:
	//   wwn+lun: plugins/kubernetes.io/fc/volumeDevices/50060e801049cfd1-lun-0/
	//   wwid: plugins/kubernetes.io/fc/volumeDevices/3600508b400105e210000900000490000/
	if pathExists, pathErr := checkPathExists(mapPath); !pathExists || pathErr != nil {
		return pathErr
	}

	// Retrieve volume plugin dependent path like '50060e801049cfd1-lun-0' from global map path
	arr := strings.Split(mapPath, "/")
	if len(arr) < 1 {
		return fmt.Errorf("failed to retrieve volume plugin information from global map path: %v", mapPath)
	}
	volumeInfo := arr[len(arr)-1]

	// Search symbolic link which matches volumeInfo under /dev/disk/by-path or /dev/disk/by-id
	// then find destination device path from the link
	searchPath := byID
	if strings.Contains(volumeInfo, "-lun-") {
		searchPath = byPath
	}
	fis, err := ioutil.ReadDir(searchPath)
	if err != nil {
		return err
	}
	for _, fi := range fis {
		if strings.Contains(fi.Name(), volumeInfo) {
			devicePath = filepath.Join(searchPath, fi.Name())
			klog.V(5).Infof("fc: updated devicePath: %s", devicePath)
			break
		}
	}
	if len(devicePath) == 0 {
		return fmt.Errorf("fc: failed to find corresponding device from searchPath: %v", searchPath)
	}
	dstPath, err := c.io.EvalSymlinks(devicePath)
	if err != nil {
		return err
	}
	klog.V(4).Infof("fc: find destination device path from symlink: %v", dstPath)

	var devices []string
	dm := c.deviceUtil.FindMultipathDeviceForDevice(dstPath)
	if len(dm) != 0 {
		dstPath = dm
	}

	// Detach volume from kubelet node
	if len(dm) != 0 {
		// Find all devices which are managed by multipath
		devices = c.deviceUtil.FindSlaveDevicesOnMultipath(dm)
		if err := util.deleteMultipathDevice(c.exec, dm); err != nil {
			return err
		}
	} else {
		// Add single device path to devices
		devices = append(devices, dstPath)
	}
	var lastErr error
	for _, device := range devices {
		err = util.detachFCDisk(c.io, c.exec, device)
		if err != nil {
			klog.Errorf("fc: detachFCDisk failed. device: %v err: %v", device, err)
			lastErr = fmt.Errorf("fc: detachFCDisk failed. device: %v err: %v", device, err)
		}
	}
	if lastErr != nil {
		klog.Errorf("fc: last error occurred during detach disk:\n%v", lastErr)
		return lastErr
	}
	return nil
}

func (util *fcUtil) deleteMultipathDevice(exec utilexec.Interface, dmDevice string) error {
	out, err := exec.Command("multipath", "-f", dmDevice).CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to flush multipath device %s: %s\n%s", dmDevice, err, string(out))
	}
	klog.V(4).Infof("Flushed multipath device: %s", dmDevice)
	return nil
}

func checkPathExists(path string) (bool, error) {
	if pathExists, pathErr := mount.PathExists(path); pathErr != nil {
		return pathExists, fmt.Errorf("error checking if path exists: %w", pathErr)
	} else if !pathExists {
		klog.Warningf("Warning: Unmap skipped because path does not exist: %v", path)
		return pathExists, nil
	}
	return true, nil
}
