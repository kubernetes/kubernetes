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

// given a disk path like /dev/sdx, find the devicemapper parent
func findMultipathDeviceMapper(disk string) string {
	sys_path := "/sys/block/"
	if dirs, err := ioutil.ReadDir(sys_path); err == nil {
		for _, f := range dirs {
			name := f.Name()
			if strings.HasPrefix(name, "dm-") {
				if _, err1 := os.Lstat(sys_path + name + "/slaves/" + disk); err1 == nil {
					return "/dev/" + name
				}
			}
		}
	}
	return ""
}

// given a wwn and lun, find the device and associated devicemapper parent
func findDisk(wwn, lun string) (string, string) {
	fc_path := "-fc-0x" + wwn + "-lun-" + lun
	dev_path := "/dev/disk/by-path/"
	if dirs, err := ioutil.ReadDir(dev_path); err == nil {
		for _, f := range dirs {
			name := f.Name()
			if strings.Contains(name, fc_path) {
				if disk, err1 := filepath.EvalSymlinks(dev_path + name); err1 == nil {
					arr := strings.Split(disk, "/")
					l := len(arr) - 1
					dev := arr[l]
					dm := findMultipathDeviceMapper(dev)
					return disk, dm
				}
			}
		}
	}
	return "", ""
}

func createMultipathConf(path string) {
	if _, err := os.Lstat(path); err != nil {
		data := []byte(`defaults {
	find_multipaths yes
	user_friendly_names yes
}


blacklist {
}
`)
		ioutil.WriteFile(path, data, 0664)
	}
}

// rescan scsi bus
func scsiHostRescan() {
	scsi_path := "/sys/class/scsi_host/"
	if dirs, err := ioutil.ReadDir(scsi_path); err == nil {
		for _, f := range dirs {
			name := scsi_path + f.Name() + "/scan"
			data := []byte("- - -")
			ioutil.WriteFile(name, data, 0666)
		}
	}
}

// make a directory like /var/lib/kubelet/plugins/kubernetes.io/pod/fc/target-lun-0
func makePDNameInternal(host volume.VolumeHost, wwns []string, lun string) string {
	return path.Join(host.GetPluginDir(fcPluginName), wwns[0]+"-lun-"+lun)
}

type FCUtil struct{}

func (util *FCUtil) MakeGlobalPDName(fc fcDisk) string {
	return makePDNameInternal(fc.plugin.host, fc.wwns, fc.lun)
}

func (util *FCUtil) AttachDisk(b fcDiskBuilder) error {
	disk := ""
	dm := ""
	devicePath := ""
	wwns := b.wwns
	lun := b.lun
	rescaned := false
	// two-phase search:
	// first phase, search existing device path, if a multipath dm is found, exit loop
	// otherwise, in second phase, rescan scsi bus and search again, return with any findings
	for true {
		for _, wwn := range wwns {
			disk, dm = findDisk(wwn, lun)
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
		// create multipath conf if it is not there
		createMultipathConf("/etc/multipath.conf")
		// rescan scsi bus
		scsiHostRescan()
		rescaned = true
	}
	// if no disk matches input wwn and lun, exit
	if disk == "" && dm == "" {
		return fmt.Errorf("no fc disk found")
	}

	// if multipath devicemapper device is found, use it; otherwise use raw disk
	if dm != "" {
		devicePath = dm
	} else {
		devicePath = disk
	}
	// mount it
	globalPDPath := b.manager.MakeGlobalPDName(*b.fcDisk)
	noMnt, err := b.mounter.IsLikelyNotMountPoint(globalPDPath)
	if !noMnt {
		glog.Infof("fc: %s already mounted", globalPDPath)
		return nil
	}

	if err := os.MkdirAll(globalPDPath, 0750); err != nil {
		return fmt.Errorf("fc: failed to mkdir %s, error", globalPDPath)
	}

	err = b.mounter.Mount(devicePath, globalPDPath, b.fsType, nil)
	if err != nil {
		return fmt.Errorf("fc: failed to mount fc volume %s [%s] to %s, error %v", devicePath, b.fsType, globalPDPath, err)
	}

	return err
}

func (util *FCUtil) DetachDisk(c fcDiskCleaner, mntPath string) error {
	if err := c.mounter.Unmount(mntPath); err != nil {
		return fmt.Errorf("fc detach disk: failed to unmount: %s\nError: %v", mntPath, err)
	}
	return nil
}
