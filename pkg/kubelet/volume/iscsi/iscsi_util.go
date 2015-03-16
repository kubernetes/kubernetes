/*
Copyright 2015 Google Inc. All rights reserved.

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

package iscsi

import (
	"errors"
	"os"
	"path"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/golang/glog"
)

func probeDevicePath(devicePath string, maxRetries int) bool {
	for i := 0; i < maxRetries; i++ {
		_, err := os.Stat(devicePath)
		if err == nil {
			return true
		}
		if err != nil && !os.IsNotExist(err) {
			return false
		}
		time.Sleep(time.Second)
	}
	return false
}

// make a directory like /var/lib/kubelet/plugins/kubernetes.io/pod/iscsi/targetIP/iqn/lun/0
func makePDNameInternal(host volume.Host, portal string, iqn string, lun string) string {
	return path.Join(host.GetPluginDir(ISCSIDiskPluginName), "iscsi", portal+"-iqn-"+iqn+"-lun-"+lun)
}

type ISCSIDiskUtil struct{}

func (util *ISCSIDiskUtil) MakeGlobalPDName(iscsi iscsiDisk) string {
	return makePDNameInternal(iscsi.plugin.host, iscsi.portal, iscsi.iqn, iscsi.lun)
}

func (util *ISCSIDiskUtil) AttachDisk(iscsi iscsiDisk) error {
	devicePath := strings.Join([]string{"/dev/disk/by-path/ip", iscsi.portal, "iscsi", iscsi.iqn, "lun", iscsi.lun}, "-")
	exist := probeDevicePath(devicePath, 1)
	if exist == false {
		// discover iscsi target
		_, err := iscsi.execCommand("iscsiadm", []string{"-m", "discovery", "-t", "sendtargets", "-p", iscsi.portal})
		if err != nil {
			glog.Errorf("iscsiPersistentDisk: failed to sendtargets to portal %s error:%v", iscsi.portal, err)
			return err
		}
		// login to iscsi target
		_, err = iscsi.execCommand("iscsiadm", []string{"-m", "node", "-p", iscsi.portal, "-T", iscsi.iqn, "--login"})
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
	globalPDPath := iscsi.manager.MakeGlobalPDName(iscsi)
	mountpoint, err := mount.IsMountPoint(globalPDPath)
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
		glog.Errorf("iSCSIPersistentDisk: failed to mount iscsi volume %s [%s] to %s, error %v", devicePath, iscsi.fsType, globalPDPath, err)
	}

	return err
}

func (util *ISCSIDiskUtil) DetachDisk(iscsi iscsiDisk, mntPath string) error {
	device, cnt, err := mount.GetDeviceFromMnt(iscsi.mounter, mntPath)
	if err != nil {
		glog.Errorf("iSCSIPersistentDisk detach disk: failed to get device from mnt: %s\nError: %v", mntPath, err)
		return err
	}
	if err = iscsi.mounter.Unmount(mntPath, 0); err != nil {
		glog.Errorf("iSCSIPersistentDisk detach disk: failed to umount: %s\nError: %v", mntPath, err)
		return err
	}
	os.RemoveAll(mntPath)
	cnt--
	// if device is no longer used, see if need to logout the target
	if cnt == 0 {
		// strip -lun- from device path
		ind := strings.LastIndex(device, "-lun-")
		prefix := device[:(ind - 1)]
		refCount, err := mount.GetDeviceRefCount(iscsi.mounter, prefix)

		if err == nil && refCount == 0 {
			// this portal/iqn are no longer referenced, log out
			// extract portal and iqn from device path
			ind1 := strings.LastIndex(device, "-iscsi-")
			portal := device[(len("/dev/disk/by-path/ip-")):ind1]
			iqn := device[ind1+len("-iscsi-") : ind]

			glog.Infof("iSCSIPersistentDisk: log out target %s iqn %s", portal, iqn)
			_, err = iscsi.execCommand("iscsiadm", []string{"-m", "node", "-p", portal, "-T", iqn, "--logout"})
			if err != nil {
				glog.Errorf("iSCSIPersistentDisk: failed to detach disk Error: %v", err)
			}
		}
	}
	return nil
}
