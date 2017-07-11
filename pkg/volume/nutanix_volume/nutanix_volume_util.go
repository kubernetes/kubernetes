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

package nutanix_volume

import (
	"fmt"
	"os"
	"path"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

// StatFunc stat a path, if path does not exist, retry maxRetries times.
type StatFunc func(string) (os.FileInfo, error)

func waitForPathToExist(devicePath string, maxRetries int) bool {
	// This makes unit testing a lot easier
	return waitForPathToExistInternal(devicePath, maxRetries, os.Stat)
}

func waitForPathToExistInternal(devicePath string, maxRetries int, osStat StatFunc) bool {
	for i := 0; i < maxRetries; i++ {
		var err error
		_, err = osStat(devicePath)
		if err == nil {
			return true
		}
		if err != nil && !os.IsNotExist(err) {
			return false
		}
		if i == maxRetries-1 {
			break
		}
		time.Sleep(time.Second)
	}
	return false
}

// getDevicePrefixRefCount: given a prefix of device path, find its reference count from /proc/mounts
// returns the reference count to the device and error code.
// For services like iscsi construct multiple device paths with the same prefix pattern.
// This function aggregates all references to a service based on the prefix pattern.
// More specifically, this prefix semantics is to aggregate disk paths that belong to the same iSCSI target/iqn pair.
// An iSCSI target could expose multiple LUNs through the same IQN, and Linux iSCSI initiator creates disk paths that
// start the same prefix but end with different LUN number
// When we decide whether it is time to logout a target, we have to see if none of the LUNs are used any more.
// That's where the prefix based ref count kicks in. If we only count the disks using exact match, we could log other
// disks out.
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

// Return list of matching targets.
func getMatchingTargets(b nutanixVolumeMounter) (string, error) {
	var targets []string

	portal := b.nutanixVolume.dataServiceEndPoint
	out, err := b.plugin.execCommand("iscsiadm", []string{"-m", "discovery", "-t", "st", "-p", portal})
	if err != nil {
		return "", fmt.Errorf("Could not discover target, output:%s, error: %v", string(out), err)
	}

	output := strings.TrimSuffix(string(out), "\n")
	for _, line := range strings.Split(output, "\n") {
		if strings.Contains(line, b.nutanixVolume.iscsiTarget) {
			targets = append(targets, strings.Split(line, " ")[1])
		}
	}

	// Nutanix volume supports only one disk per volume group.
	if len(targets) == 0 {
		return "", fmt.Errorf("Could not find matching target for %s", b.nutanixVolume.iscsiTarget)
	}
	if len(targets) > 1 {
		return "", fmt.Errorf("num_targets %d, only one target is supported", len(targets))
	}
	return targets[0], nil
}

// make a directory like /var/lib/kubelet/plugins/kubernetes.io/nutanix_volume/portal-some_iqn-lun-lun_id.
func makePDNameInternal(host volume.VolumeHost, podUID, portal, iqn string) string {
	return path.Join(host.GetPluginDir(nutanixVolumePluginName), podUID, portal+"-"+iqn+"-lun-0")
}

type NutanixVolumeUtil struct{}

var _ diskManager = &NutanixVolumeUtil{}

func (util *NutanixVolumeUtil) MakeGlobalPDName(volume nutanixVolume, target string) string {
	portal := volume.dataServiceEndPoint
	return makePDNameInternal(volume.plugin.host, string(volume.podUID), portal, target)
}

func (util *NutanixVolumeUtil) AttachDisk(b nutanixVolumeMounter) error {
	var devicePath string

	target, err := getMatchingTargets(b)
	if err != nil {
		return err
	}

	portal := b.nutanixVolume.dataServiceEndPoint
	out, err := b.plugin.execCommand("iscsiadm", []string{"-m", "node", "-T", target, "-l", "-p", portal})
	if err != nil {
		return fmt.Errorf("Could not login to target, out: %s, error: %v", string(out), err)
	}
	devicePath = strings.Join([]string{"/dev/disk/by-path/ip", portal, "iscsi", target, "lun-0"}, "-")
	// Wait up to 10 sec for path to exist.
	exist := waitForPathToExist(devicePath, 10)
	if !exist {
		return fmt.Errorf("failed to get any path for iscsi disk, Timeout after 10s")
	}

	// mount it.
	globalPDPath := b.manager.MakeGlobalPDName(*b.nutanixVolume, target)
	notMnt, err := b.mounter.IsLikelyNotMountPoint(globalPDPath)
	if !notMnt {
		glog.Infof("nutanix_volume: %s already mounted", globalPDPath)
		return nil
	}

	if err := os.MkdirAll(globalPDPath, 0750); err != nil {
		return fmt.Errorf("Failed to mkdir %s, error %v", globalPDPath, err)
	}

	err = b.mounter.FormatAndMount(devicePath, globalPDPath, b.fsType, nil)
	if err != nil {
		return fmt.Errorf("Failed to mount iscsi volume %s [%s] to %s, error %v", devicePath, b.fsType, globalPDPath, err)
	}

	return err
}

func (util *NutanixVolumeUtil) DetachDisk(c nutanixVolumeUnmounter, mntPath string) error {
	_, cnt, err := mount.GetDeviceNameFromMount(c.mounter, mntPath)
	if err != nil {
		return fmt.Errorf("Failed to get device from mnt: %s\nError: %v", mntPath, err)
	}
	if err = c.mounter.Unmount(mntPath); err != nil {
		return fmt.Errorf("Failed to unmount: %s\nError: %v", mntPath, err)
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
			// This portal/iqn/iface is no longer referenced, log out.
			// Extract the portal and iqn from device path.
			portal, iqn, err := extractPortalAndIqn(device)
			if err != nil {
				return err
			}

			glog.V(4).Infof("nutanix_volume: log out target %s iqn %s", portal, iqn)
			out, err := c.plugin.execCommand("iscsiadm", []string{"-m", "node", "-p", portal, "-T", iqn, "--logout"})
			if err != nil {
				glog.Errorf("nutanix_volume: failed to detach disk Error: %s", string(out))
			}
		}
	}
	return nil
}

func extractDeviceAndPrefix(mntPath string) (string, string, error) {
	ind := strings.LastIndex(mntPath, "/")
	if ind < 0 {
		return "", "", fmt.Errorf("Malformatted mnt path: %s", mntPath)
	}
	device := mntPath[(ind + 1):]
	// strip -lun- from mount path
	ind = strings.LastIndex(mntPath, "-lun-")
	if ind < 0 {
		return "", "", fmt.Errorf("Malformatted mnt path: %s", mntPath)
	}
	prefix := mntPath[:ind]
	return device, prefix, nil
}

func extractPortalAndIqn(device string) (string, string, error) {
	// Example device: 10.5.65.156:3260-c50d776301b75e568ab232af3123087f492b02fccecacc8e3aa0a9ddb78d8912:nutanix-k8-volume-plugin-lun-0
	ind1 := strings.Index(device, "-")
	if ind1 < 0 {
		return "", "", fmt.Errorf("No portal in %s", device)
	}
	portal := device[0:ind1]
	ind := strings.LastIndex(device, "-lun-")
	iqn := device[ind1+1 : ind]
	return portal, iqn, nil
}
