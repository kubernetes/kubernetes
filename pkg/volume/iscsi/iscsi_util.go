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

package iscsi

import (
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

// stat a path, if not exists, retry maxRetries times
// when iscsi transports other than default are used,  use glob instead as pci id of device is unknown
type StatFunc func(string) (os.FileInfo, error)
type GlobFunc func(string) ([]string, error)

func waitForPathToExist(devicePath string, maxRetries int, deviceTransport string) bool {
	// This makes unit testing a lot easier
	return waitForPathToExistInternal(devicePath, maxRetries, deviceTransport, os.Stat, filepath.Glob)
}

func waitForPathToExistInternal(devicePath string, maxRetries int, deviceTransport string, osStat StatFunc, filepathGlob GlobFunc) bool {
	for i := 0; i < maxRetries; i++ {
		var err error
		if deviceTransport == "tcp" {
			_, err = osStat(devicePath)
		} else {
			fpath, _ := filepathGlob(devicePath)
			if fpath == nil {
				err = os.ErrNotExist
			}
		}
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
		if strings.HasPrefix(mps[i].Path, deviceNamePrefix) {
			refCount++
		}
	}
	return refCount, nil
}

// make a directory like /var/lib/kubelet/plugins/kubernetes.io/iscsi/portal-some_iqn-lun-lun_id
func makePDNameInternal(host volume.VolumeHost, portal string, iqn string, lun string) string {
	return path.Join(host.GetPluginDir(iscsiPluginName), portal+"-"+iqn+"-lun-"+lun)
}

type ISCSIUtil struct{}

func (util *ISCSIUtil) MakeGlobalPDName(iscsi iscsiDisk) string {
	return makePDNameInternal(iscsi.plugin.host, iscsi.portal, iscsi.iqn, iscsi.lun)
}

func (util *ISCSIUtil) AttachDisk(b iscsiDiskMounter) error {
	var devicePath string
	var iscsiTransport string

	out, err := b.plugin.execCommand("iscsiadm", []string{"-m", "iface", "-I", b.iface, "-o", "show"})
	if err != nil {
		glog.Errorf("iscsi: could not read iface %s error: %s", b.iface, string(out))
		return err
	}

	iscsiTransport = extractTransportname(string(out))

	if iscsiTransport == "" {
		glog.Errorf("iscsi: could not find transport name in iface %s", b.iface)
		return errors.New(fmt.Sprintf("Could not parse iface file for %s", b.iface))
	} else if iscsiTransport == "tcp" {
		devicePath = strings.Join([]string{"/dev/disk/by-path/ip", b.portal, "iscsi", b.iqn, "lun", b.lun}, "-")
	} else {
		devicePath = strings.Join([]string{"/dev/disk/by-path/pci", "*", "ip", b.portal, "iscsi", b.iqn, "lun", b.lun}, "-")
	}
	exist := waitForPathToExist(devicePath, 1, iscsiTransport)
	if exist == false {
		// discover iscsi target
		out, err := b.plugin.execCommand("iscsiadm", []string{"-m", "discovery", "-t", "sendtargets", "-p", b.portal, "-I", b.iface})
		if err != nil {
			glog.Errorf("iscsi: failed to sendtargets to portal %s error: %s", b.portal, string(out))
			return err
		}
		// login to iscsi target
		out, err = b.plugin.execCommand("iscsiadm", []string{"-m", "node", "-p", b.portal, "-T", b.iqn, "-I", b.iface, "--login"})
		if err != nil {
			glog.Errorf("iscsi: failed to attach disk:Error: %s (%v)", string(out), err)
			return err
		}
		exist = waitForPathToExist(devicePath, 10, iscsiTransport)
		if !exist {
			return errors.New("Could not attach disk: Timeout after 10s")
		}
	}
	// mount it
	globalPDPath := b.manager.MakeGlobalPDName(*b.iscsiDisk)
	notMnt, err := b.mounter.IsLikelyNotMountPoint(globalPDPath)
	if !notMnt {
		glog.Infof("iscsi: %s already mounted", globalPDPath)
		return nil
	}

	if err := os.MkdirAll(globalPDPath, 0750); err != nil {
		glog.Errorf("iscsi: failed to mkdir %s, error", globalPDPath)
		return err
	}

	// check if the dev is using mpio and if so mount it via the dm-XX device
	if mappedDevicePath := b.deviceUtil.FindMultipathDeviceForDevice(devicePath); mappedDevicePath != "" {
		devicePath = mappedDevicePath
	}
	err = b.mounter.FormatAndMount(devicePath, globalPDPath, b.fsType, nil)
	if err != nil {
		glog.Errorf("iscsi: failed to mount iscsi volume %s [%s] to %s, error %v", devicePath, b.fsType, globalPDPath, err)
	}

	return err
}

func (util *ISCSIUtil) DetachDisk(c iscsiDiskUnmounter, mntPath string) error {
	_, cnt, err := mount.GetDeviceNameFromMount(c.mounter, mntPath)
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
		device, prefix, err := extractDeviceAndPrefix(mntPath)
		if err != nil {
			return err
		}
		refCount, err := getDevicePrefixRefCount(c.mounter, prefix)

		if err == nil && refCount == 0 {
			// this portal/iqn are no longer referenced, log out
			// extract portal and iqn from device path
			portal, iqn, err := extractPortalAndIqn(device)
			if err != nil {
				return err
			}
			glog.Infof("iscsi: log out target %s iqn %s", portal, iqn)
			out, err := c.plugin.execCommand("iscsiadm", []string{"-m", "node", "-p", portal, "-T", iqn, "--logout"})
			if err != nil {
				glog.Errorf("iscsi: failed to detach disk Error: %s", string(out))
			}
		}
	}
	return nil
}

func extractTransportname(ifaceOutput string) (iscsiTransport string) {
	re := regexp.MustCompile(`iface.transport_name = (.*)\n`)

	rex_output := re.FindStringSubmatch(ifaceOutput)
	if rex_output != nil {
		iscsiTransport = rex_output[1]
	} else {
		return ""
	}

	// While iface.transport_name is a required parameter, handle it being unspecified anyways
	if iscsiTransport == "<empty>" {
		iscsiTransport = "tcp"
	}
	return iscsiTransport
}

func extractDeviceAndPrefix(mntPath string) (string, string, error) {
	ind := strings.LastIndex(mntPath, "/")
	if ind < 0 {
		return "", "", fmt.Errorf("iscsi detach disk: malformatted mnt path: %s", mntPath)
	}
	device := mntPath[(ind + 1):]
	// strip -lun- from device path
	ind = strings.LastIndex(device, "-lun-")
	if ind < 0 {
		return "", "", fmt.Errorf("iscsi detach disk: malformatted mnt path: %s", mntPath)
	}
	prefix := device[:ind]
	return device, prefix, nil
}

func extractPortalAndIqn(device string) (string, string, error) {
	ind1 := strings.Index(device, "-")
	if ind1 < 0 {
		return "", "", fmt.Errorf("iscsi detach disk: no portal in %s", device)
	}
	portal := device[0:ind1]
	ind2 := strings.Index(device, "iqn.")
	if ind2 < 0 {
		ind2 = strings.Index(device, "eui.")
	}
	if ind2 < 0 {
		return "", "", fmt.Errorf("iscsi detach disk: no iqn in %s", device)
	}
	ind := strings.LastIndex(device, "-lun-")
	iqn := device[ind2:ind]
	return portal, iqn, nil
}
