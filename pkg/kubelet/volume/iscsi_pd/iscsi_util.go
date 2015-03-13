/*
Copyright 2015 Red Hat Inc. All rights reserved.

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
package iscsi_pd

import (
	"errors"
	"os"
	"path"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume/gce_pd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/golang/glog"
)

func makeGlobalPDName(host volume.Host, portal string, iqn string, lun string) string {
	return path.Join(host.GetPluginDir(ISCSIDiskPluginName), "iscsi", portal, iqn, "lun", lun)
}

func probeDevicePath(devicePath string, maxRetries int) bool {
	numTries := 0
	for {
		_, err := os.Stat(devicePath)
		if err == nil {
			return true
		}
		if err != nil && !os.IsNotExist(err) {
			return false
		}
		numTries++
		if numTries == maxRetries {
			break
		}
		time.Sleep(time.Second)
	}
	return false
}

type ISCSIDiskUtil struct{}

func (util *ISCSIDiskUtil) AttachDisk(iscsi *iscsiDisk) error {
	devicePath := strings.Join([]string{"/dev/disk/by-path/ip", iscsi.portal, "iscsi", iscsi.iqn, "lun", iscsi.lun}, "-")
	exist := probeDevicePath(devicePath, 1)
	if exist == false {
		// discover iscsi target
		_, err := iscsi.execCommand("iscsiadm", []string{"-m", "discovery", "-t", "sendtargets", "-p",
			iscsi.portal})
		if err != nil {
			glog.Errorf("iscsiPersistentDisk: failed to sendtargets to portal %s error:%v", iscsi.portal, err)
			return err
		}
		// login to iscsi target
		_, err = iscsi.execCommand("iscsiadm", []string{"-m", "node", "-p",
			iscsi.portal, "-T", iscsi.iqn, "--login"})
		if err != nil {
			glog.Errorf("iscsiPersistentDisk: failed to attach disk:Error: %v", err)
			return err
		}
		exist = probeDevicePath(devicePath, 10)
		if !exist {
			return errors.New("Could not attach disk: Timeout after 10s")
		}
	}
	// mount it
	globalPDPath := makeGlobalPDName(iscsi.plugin.host, iscsi.portal, iscsi.iqn, iscsi.lun)
	mountpoint, err := gce_pd.IsMountPoint(globalPDPath)
	if mountpoint {
		glog.Infof("iscsiPersistentDisk: %s already mounted", globalPDPath)
		return nil
	}

	if err := os.MkdirAll(globalPDPath, 0750); err != nil {
		glog.Errorf("iSCSIPersistentDisk: failed to mkdir %s, error", globalPDPath)
		return err
	}

	err = iscsi.mounter.Mount(devicePath, globalPDPath, iscsi.fsType, uintptr(0), "")
	if err != nil {
		glog.Errorf("iSCSIPersistentDisk: failed to mount iscsi volume %s [%s] to %s, error %v",
			devicePath, iscsi.fsType, globalPDPath, err)
	}

	return err
}

func (util *ISCSIDiskUtil) DetachDisk(iscsi *iscsiDisk, devicePath string) error {
	globalPDPath, refCount, err := GetMountFromDevicePath(iscsi.mounter, devicePath)
	if refCount > 1 {
		glog.Errorf("iSCSIPersistentDisk detach disk: mount %s is used %d times", globalPDPath, refCount)
		return errors.New("path busy")
	}
	//glog.Infof("iSCSIPersistentDisk detach disk:  umount:%s", globalPDPath)
	if err := iscsi.mounter.Unmount(globalPDPath, 0); err != nil {
		glog.Errorf("iSCSIPersistentDisk detach disk: failed to umount: %s\nError: %v", globalPDPath, err)
		return err
	}

	// if the iscsi portal is no longer used, logout the target
	prefix := strings.Join([]string{"/dev/disk/by-path/ip", iscsi.portal, "iscsi"}, "-")
	refCount, err = GetDeviceRefCount(iscsi.mounter, prefix)
	//glog.Infof("iSCSIPersistentDisk: log out target: dev %s ref %d error %v", prefix, refCount, err)
	if err == nil && refCount == 0 {
		glog.Infof("iSCSIPersistentDisk: log out target %s", iscsi.portal)
		_, err = iscsi.execCommand("iscsiadm", []string{"-m", "node", "-p",
			iscsi.portal, "-T", iscsi.iqn, "--logout"})
		if err != nil {
			glog.Errorf("iSCSIPersistentDisk: failed to detach disk Error: %v", err)
		}
	}
	return nil
}

// given a device path, find its reference count from /proc/mounts
// mostly borrowed from gce-pd
// export it so other block volume plugin can use it.
func GetDeviceRefCount(mounter mount.Interface, deviceName string) (int, error) {
	mps, err := mounter.List()
	if err != nil {
		return -1, err
	}

	// Find the number of references to the device.
	refCount := 0
	for i := range mps {
		if strings.HasPrefix(mps[i].Device, deviceName) {
			refCount++
		}
	}
	return refCount, nil
}

// given a device path, find the mount on that device from /proc/mounts
// mostly borrowed from gce-pd
// export it so other block volume plugin can use it.
func GetMountFromDevicePath(mounter mount.Interface, deviceName string) (string, int, error) {
	// mostly borrowed from gce-pd
	// export it so other block volume plugin can use it.
	mps, err := mounter.List()
	if err != nil {
		return "", -1, err
	}
	mnt := ""
	// Find the number of references to the device.
	refCount := 0
	for i := range mps {
		if mps[i].Device == deviceName {
			mnt = mps[i].Path
			refCount++
		}
	}
	return mnt, refCount, nil
}
