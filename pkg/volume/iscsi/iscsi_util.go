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

package iscsi

import (
	"errors"
	"os"
	"path"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/golang/glog"
)

// stat a path, if not exists, retry maxRetries times
func waitForPathToExist(devicePath string, maxRetries int) bool {
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

// getDevicePrefixRefCount: given a prefix of device path, find its reference count from /proc/mounts
// returns the reference count to the device and error code
// for services like iscsi construct multiple device paths with the same prefix pattern.
// this function aggregates all references to a service based on the prefix pattern
// More specifically, this prefix semantics is to aggregate disk paths that belong to the same iSCSI target/iqn pair.
// an iSCSI target could expose multiple LUNs through the same IQN, and Linux iSCSI initiator creates disk paths that start the same prefix but end with different LUN number
// When we decide whether it is time to logout a target, we have to see if none of the LUNs are used any more.
// That's where the prefix based ref count kicks in. If we only count the disks using exact match, we could log other disks out.
func getDevicePrefixRefCount(mounter mount.Interface, deviceNamePrefix string) (int, error) {
	mps, err := mounter.List()
	if err != nil {
		return -1, err
	}

	// Find the number of references to the device.
	refCount := 0
	for i := range mps {
		if strings.HasPrefix(mps[i].Device, deviceNamePrefix) {
			refCount++
		}
	}
	return refCount, nil
}

// make a directory like /var/lib/kubelet/plugins/kubernetes.io/pod/iscsi/portal-iqn-some_iqn-lun-0
func makePDNameInternal(host volume.VolumeHost, portal string, iqn string, lun string) string {
	return path.Join(host.GetPluginDir(iscsiPluginName), "iscsi", portal+"-iqn-"+iqn+"-lun-"+lun)
}

type ISCSIUtil struct{}

func (util *ISCSIUtil) MakeGlobalPDName(iscsi iscsiDisk) string {
	return makePDNameInternal(iscsi.plugin.host, iscsi.portal, iscsi.iqn, iscsi.lun)
}

func (util *ISCSIUtil) AttachDisk(b iscsiDiskBuilder) error {
	devicePath := strings.Join([]string{"/dev/disk/by-path/ip", b.portal, "iscsi", b.iqn, "lun", b.lun}, "-")
	exist := waitForPathToExist(devicePath, 1)
	if exist == false {
		// discover iscsi target
		out, err := b.plugin.execCommand("iscsiadm", []string{"-m", "discovery", "-t", "sendtargets", "-p", b.portal})
		if err != nil {
			glog.Errorf("iscsi: failed to sendtargets to portal %s error: %s", b.portal, string(out))
			return err
		}
		// login to iscsi target
		out, err = b.plugin.execCommand("iscsiadm", []string{"-m", "node", "-p", b.portal, "-T", b.iqn, "--login"})
		if err != nil {
			glog.Errorf("iscsi: failed to attach disk:Error: %s (%v)", string(out), err)
			return err
		}
		exist = waitForPathToExist(devicePath, 10)
		if !exist {
			return errors.New("Could not attach disk: Timeout after 10s")
		}
	}
	// mount it
	globalPDPath := b.manager.MakeGlobalPDName(*b.iscsiDisk)
	mountpoint, err := b.mounter.IsMountPoint(globalPDPath)
	if mountpoint {
		glog.Infof("iscsi: %s already mounted", globalPDPath)
		return nil
	}

	if err := os.MkdirAll(globalPDPath, 0750); err != nil {
		glog.Errorf("iscsi: failed to mkdir %s, error", globalPDPath)
		return err
	}

	err = b.mounter.Mount(devicePath, globalPDPath, b.fsType, nil)
	if err != nil {
		glog.Errorf("iscsi: failed to mount iscsi volume %s [%s] to %s, error %v", devicePath, b.fsType, globalPDPath, err)
	}

	return err
}

func (util *ISCSIUtil) DetachDisk(c iscsiDiskCleaner, mntPath string) error {
	device, cnt, err := mount.GetDeviceNameFromMount(c.mounter, mntPath)
	if err != nil {
		glog.Errorf("iscsi detach disk: failed to get device from mnt: %s\nError: %v", mntPath, err)
		return err
	}
	if err = c.mounter.Unmount(mntPath); err != nil {
		glog.Errorf("iscsi detach disk: failed to unmount: %s\nError: %v", mntPath, err)
		return err
	}
	cnt--
	// if device is no longer used, see if need to logout the target
	if cnt == 0 {
		// strip -lun- from device path
		ind := strings.LastIndex(device, "-lun-")
		prefix := device[:(ind - 1)]
		refCount, err := getDevicePrefixRefCount(c.mounter, prefix)

		if err == nil && refCount == 0 {
			// this portal/iqn are no longer referenced, log out
			// extract portal and iqn from device path
			ind1 := strings.LastIndex(device, "-iscsi-")
			portal := device[(len("/dev/disk/by-path/ip-")):ind1]
			iqn := device[ind1+len("-iscsi-") : ind]

			glog.Infof("iscsi: log out target %s iqn %s", portal, iqn)
			out, err := c.plugin.execCommand("iscsiadm", []string{"-m", "node", "-p", portal, "-T", iqn, "--logout"})
			if err != nil {
				glog.Errorf("iscsi: failed to detach disk Error: %s", string(out))
			}
		}
	}
	return nil
}
