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
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"k8s.io/klog/v2"
	utilexec "k8s.io/utils/exec"
	"k8s.io/utils/mount"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/types"
)

const (
	// Minimum number of paths that the volume plugin considers enough when a multipath volume is requested.
	minMultipathCount = 2

	// Minimal number of attempts to attach all paths of a multipath volumes. If at least minMultipathCount paths
	// are available after this nr. of attempts, the volume plugin continues with mounting the volume.
	minAttachAttempts = 2

	// Total number of attempts to attach at least minMultipathCount paths. If there are less than minMultipathCount,
	// the volume plugin tries to attach the remaining paths at least this number of times in total. After
	// maxAttachAttempts attempts, it mounts even a single path.
	maxAttachAttempts = 5

	// How many seconds to wait for a multipath device if at least two paths are available.
	multipathDeviceTimeout = 10

	// How many seconds to wait for a device/path to appear before giving up.
	deviceDiscoveryTimeout = 30

	// 'iscsiadm' error code stating that a session is logged in
	// See https://github.com/open-iscsi/open-iscsi/blob/7d121d12ad6ba7783308c25ffd338a9fa0cc402b/include/iscsi_err.h#L37-L38
	iscsiadmErrorSessExists = 15

	// iscsiadm exit code for "session could not be found"
	exit_ISCSI_ERR_SESS_NOT_FOUND = 2
	// iscsiadm exit code for "no records/targets/sessions/portals found to execute operation on."
	exit_ISCSI_ERR_NO_OBJS_FOUND = 21
)

var (
	chapSt = []string{
		"discovery.sendtargets.auth.username",
		"discovery.sendtargets.auth.password",
		"discovery.sendtargets.auth.username_in",
		"discovery.sendtargets.auth.password_in"}
	chapSess = []string{
		"node.session.auth.username",
		"node.session.auth.password",
		"node.session.auth.username_in",
		"node.session.auth.password_in"}
	ifaceTransportNameRe = regexp.MustCompile(`iface.transport_name = (.*)\n`)
	ifaceRe              = regexp.MustCompile(`.+/iface-([^/]+)/.+`)
)

func updateISCSIDiscoverydb(b iscsiDiskMounter, tp string) error {
	if !b.chapDiscovery {
		return nil
	}
	out, err := execWithLog(b, "iscsiadm", "-m", "discoverydb", "-t", "sendtargets", "-p", tp, "-I", b.Iface, "-o", "update", "-n", "discovery.sendtargets.auth.authmethod", "-v", "CHAP")
	if err != nil {
		return fmt.Errorf("iscsi: failed to update discoverydb with CHAP, output: %v", out)
	}

	for _, k := range chapSt {
		v := b.secret[k]
		if len(v) > 0 {
			// explicitly not using execWithLog so secrets are not logged
			out, err := b.exec.Command("iscsiadm", "-m", "discoverydb", "-t", "sendtargets", "-p", tp, "-I", b.Iface, "-o", "update", "-n", k, "-v", v).CombinedOutput()
			if err != nil {
				return fmt.Errorf("iscsi: failed to update discoverydb key %q error: %v", k, string(out))
			}
		}
	}
	return nil
}

func updateISCSINode(b iscsiDiskMounter, tp string) error {
	// setting node.session.scan to manual to handle https://github.com/kubernetes/kubernetes/issues/90982
	out, err := execWithLog(b, "iscsiadm", "-m", "node", "-p", tp, "-T", b.Iqn, "-I", b.Iface, "-o", "update", "-n", "node.session.scan", "-v", "manual")
	if err != nil {
		// don't fail if iscsiadm fails or the version does not support node.session.scan - log a warning to highlight the potential exposure
		klog.Warningf("iscsi: failed to update node with node.session.scan=manual, possible exposure to issue 90982: %v", out)
	}

	if !b.chapSession {
		return nil
	}

	out, err = execWithLog(b, "iscsiadm", "-m", "node", "-p", tp, "-T", b.Iqn, "-I", b.Iface, "-o", "update", "-n", "node.session.auth.authmethod", "-v", "CHAP")
	if err != nil {
		return fmt.Errorf("iscsi: failed to update node with CHAP, output: %v", out)
	}

	for _, k := range chapSess {
		v := b.secret[k]
		if len(v) > 0 {
			// explicitly not using execWithLog so secrets are not logged
			out, err := b.exec.Command("iscsiadm", "-m", "node", "-p", tp, "-T", b.Iqn, "-I", b.Iface, "-o", "update", "-n", k, "-v", v).CombinedOutput()
			if err != nil {
				return fmt.Errorf("iscsi: failed to update node session key %q error: %v", k, string(out))
			}
		}
	}
	return nil
}

// stat a path, if not exists, retry maxRetries times
// when iscsi transports other than default are used,  use glob instead as pci id of device is unknown
type StatFunc func(string) (os.FileInfo, error)
type GlobFunc func(string) ([]string, error)

func waitForPathToExist(devicePath *string, maxRetries int, deviceTransport string) bool {
	// This makes unit testing a lot easier
	return waitForPathToExistInternal(devicePath, maxRetries, deviceTransport, os.Stat, filepath.Glob)
}

func waitForPathToExistInternal(devicePath *string, maxRetries int, deviceTransport string, osStat StatFunc, filepathGlob GlobFunc) bool {
	if devicePath == nil {
		return false
	}

	for i := 0; i < maxRetries; i++ {
		var err error
		if deviceTransport == "tcp" {
			_, err = osStat(*devicePath)
		} else {
			fpath, _ := filepathGlob(*devicePath)
			if fpath == nil {
				err = os.ErrNotExist
			} else {
				// There might be a case that fpath contains multiple device paths if
				// multiple PCI devices connect to same iscsi target. We handle this
				// case at subsequent logic. Pick up only first path here.
				*devicePath = fpath[0]
			}
		}
		if err == nil {
			return true
		}
		if !os.IsNotExist(err) {
			return false
		}
		if i == maxRetries-1 {
			break
		}
		time.Sleep(time.Second)
	}
	return false
}

// make a directory like /var/lib/kubelet/plugins/kubernetes.io/iscsi/iface_name/portal-some_iqn-lun-lun_id
func makePDNameInternal(host volume.VolumeHost, portal string, iqn string, lun string, iface string) string {
	return filepath.Join(host.GetPluginDir(iscsiPluginName), "iface-"+iface, portal+"-"+iqn+"-lun-"+lun)
}

// make a directory like /var/lib/kubelet/plugins/kubernetes.io/iscsi/volumeDevices/iface_name/portal-some_iqn-lun-lun_id
func makeVDPDNameInternal(host volume.VolumeHost, portal string, iqn string, lun string, iface string) string {
	return filepath.Join(host.GetVolumeDevicePluginDir(iscsiPluginName), "iface-"+iface, portal+"-"+iqn+"-lun-"+lun)
}

type ISCSIUtil struct{}

// MakeGlobalPDName returns path of global plugin dir
func (util *ISCSIUtil) MakeGlobalPDName(iscsi iscsiDisk) string {
	return makePDNameInternal(iscsi.plugin.host, iscsi.Portals[0], iscsi.Iqn, iscsi.Lun, iscsi.Iface)
}

// MakeGlobalVDPDName returns path of global volume device plugin dir
func (util *ISCSIUtil) MakeGlobalVDPDName(iscsi iscsiDisk) string {
	return makeVDPDNameInternal(iscsi.plugin.host, iscsi.Portals[0], iscsi.Iqn, iscsi.Lun, iscsi.Iface)
}

// persistISCSIFile saves iSCSI volume configuration for DetachDisk
// into given directory.
func (util *ISCSIUtil) persistISCSIFile(conf iscsiDisk, mnt string) error {
	file := filepath.Join(mnt, "iscsi.json")
	fp, err := os.Create(file)
	if err != nil {
		return fmt.Errorf("iscsi: create %s err %s", file, err)
	}
	defer fp.Close()
	encoder := json.NewEncoder(fp)
	if err = encoder.Encode(conf); err != nil {
		return fmt.Errorf("iscsi: encode err: %v", err)
	}
	return nil
}

func (util *ISCSIUtil) loadISCSI(conf *iscsiDisk, mnt string) error {
	file := filepath.Join(mnt, "iscsi.json")
	fp, err := os.Open(file)
	if err != nil {
		return fmt.Errorf("iscsi: open %s err %s", file, err)
	}
	defer fp.Close()
	decoder := json.NewDecoder(fp)
	if err = decoder.Decode(conf); err != nil {
		return fmt.Errorf("iscsi: decode err: %v", err)
	}
	return nil
}

// scanOneLun scans a single LUN on one SCSI bus
// Use this to avoid scanning the whole SCSI bus for all of the LUNs, which
// would result in the kernel on this node discovering LUNs that it shouldn't
// know about. Extraneous LUNs cause problems because they may get deleted
// without us getting notified, since we were never supposed to know about
// them. When LUNs are deleted without proper cleanup in the kernel, I/O errors
// and timeouts result, which can noticeably degrade performance of future
// operations.
func scanOneLun(hostNumber int, lunNumber int) error {
	filename := fmt.Sprintf("/sys/class/scsi_host/host%d/scan", hostNumber)
	fd, err := os.OpenFile(filename, os.O_WRONLY, 0)
	if err != nil {
		return err
	}
	defer fd.Close()

	// Channel/Target are always 0 for iSCSI
	scanCmd := fmt.Sprintf("0 0 %d", lunNumber)
	if written, err := fd.WriteString(scanCmd); err != nil {
		return err
	} else if 0 == written {
		return fmt.Errorf("No data written to file: %s", filename)
	}

	klog.V(3).Infof("Scanned SCSI host %d LUN %d", hostNumber, lunNumber)
	return nil
}

func waitForMultiPathToExist(devicePaths []string, maxRetries int, deviceUtil volumeutil.DeviceUtil) string {
	if 0 == len(devicePaths) {
		return ""
	}

	for i := 0; i < maxRetries; i++ {
		for _, path := range devicePaths {
			// There shouldn't be any empty device paths. However adding this check
			// for safer side to avoid the possibility of an empty entry.
			if path == "" {
				continue
			}
			// check if the dev is using mpio and if so mount it via the dm-XX device
			if mappedDevicePath := deviceUtil.FindMultipathDeviceForDevice(path); mappedDevicePath != "" {
				return mappedDevicePath
			}
		}
		if i == maxRetries-1 {
			break
		}
		time.Sleep(time.Second)
	}
	return ""
}

// AttachDisk returns devicePath of volume if attach succeeded otherwise returns error
func (util *ISCSIUtil) AttachDisk(b iscsiDiskMounter) (string, error) {
	var devicePath string
	devicePaths := map[string]string{}
	var iscsiTransport string
	var lastErr error

	out, err := execWithLog(b, "iscsiadm", "-m", "iface", "-I", b.InitIface, "-o", "show")
	if err != nil {
		klog.Errorf("iscsi: could not read iface %s error: %s", b.InitIface, out)
		return "", err
	}

	iscsiTransport = extractTransportname(out)

	bkpPortal := b.Portals

	// If the initiator name was set, the iface isn't created yet,
	// so create it and copy parameters from the pre-configured one
	if b.InitiatorName != "" {
		if err = cloneIface(b); err != nil {
			klog.Errorf("iscsi: failed to clone iface: %s error: %v", b.InitIface, err)
			return "", err
		}
	}

	// Lock the target while we login to avoid races between 2 volumes that share the same
	// target both logging in or one logging out while another logs in.
	b.plugin.targetLocks.LockKey(b.Iqn)
	defer b.plugin.targetLocks.UnlockKey(b.Iqn)

	// Build a map of SCSI hosts for each target portal. We will need this to
	// issue the bus rescans.
	portalHostMap, err := b.deviceUtil.GetISCSIPortalHostMapForTarget(b.Iqn)
	if err != nil {
		return "", err
	}
	klog.V(4).Infof("AttachDisk portal->host map for %s is %v", b.Iqn, portalHostMap)

	for i := 1; i <= maxAttachAttempts; i++ {
		for _, tp := range bkpPortal {
			if _, found := devicePaths[tp]; found {
				klog.V(4).Infof("Device for portal %q already known", tp)
				continue
			}

			hostNumber, loggedIn := portalHostMap[tp]
			if !loggedIn {
				klog.V(4).Infof("Could not get SCSI host number for portal %s, will attempt login", tp)

				// build discoverydb and discover iscsi target
				execWithLog(b, "iscsiadm", "-m", "discoverydb", "-t", "sendtargets", "-p", tp, "-I", b.Iface, "-o", "new")

				// update discoverydb with CHAP secret
				err = updateISCSIDiscoverydb(b, tp)
				if err != nil {
					lastErr = fmt.Errorf("iscsi: failed to update discoverydb to portal %s error: %v", tp, err)
					continue
				}

				out, err = execWithLog(b, "iscsiadm", "-m", "discoverydb", "-t", "sendtargets", "-p", tp, "-I", b.Iface, "--discover")
				if err != nil {
					// delete discoverydb record
					execWithLog(b, "iscsiadm", "-m", "discoverydb", "-t", "sendtargets", "-p", tp, "-I", b.Iface, "-o", "delete")
					lastErr = fmt.Errorf("iscsi: failed to sendtargets to portal %s output: %s, err %v", tp, out, err)
					continue
				}

				err = updateISCSINode(b, tp)
				if err != nil {
					// failure to update node db is rare. But deleting record will likely impact those who already start using it.
					lastErr = fmt.Errorf("iscsi: failed to update iscsi node to portal %s error: %v", tp, err)
					continue
				}

				// login to iscsi target
				out, err = execWithLog(b, "iscsiadm", "-m", "node", "-p", tp, "-T", b.Iqn, "-I", b.Iface, "--login")
				if err != nil {
					// delete the node record from database
					execWithLog(b, "iscsiadm", "-m", "node", "-p", tp, "-I", b.Iface, "-T", b.Iqn, "-o", "delete")
					lastErr = fmt.Errorf("iscsi: failed to attach disk: Error: %s (%v)", out, err)
					continue
				}

				// in case of node failure/restart, explicitly set to manual login so it doesn't hang on boot
				_, err = execWithLog(b, "iscsiadm", "-m", "node", "-p", tp, "-T", b.Iqn, "-o", "update", "-n", "node.startup", "-v", "manual")
				if err != nil {
					// don't fail if we can't set startup mode, but log warning so there is a clue
					klog.Warningf("Warning: Failed to set iSCSI login mode to manual. Error: %v", err)
				}

				// Rebuild the host map after logging in
				portalHostMap, err := b.deviceUtil.GetISCSIPortalHostMapForTarget(b.Iqn)
				if err != nil {
					return "", err
				}
				klog.V(6).Infof("AttachDisk portal->host map for %s is %v", b.Iqn, portalHostMap)

				hostNumber, loggedIn = portalHostMap[tp]
				if !loggedIn {
					klog.Warningf("Could not get SCSI host number for portal %s after logging in", tp)
					continue
				}
			}

			klog.V(5).Infof("AttachDisk: scanning SCSI host %d LUN %s", hostNumber, b.Lun)
			lunNumber, err := strconv.Atoi(b.Lun)
			if err != nil {
				return "", fmt.Errorf("AttachDisk: lun is not a number: %s\nError: %v", b.Lun, err)
			}

			// Scan the iSCSI bus for the LUN
			err = scanOneLun(hostNumber, lunNumber)
			if err != nil {
				return "", err
			}

			if iscsiTransport == "" {
				klog.Errorf("iscsi: could not find transport name in iface %s", b.Iface)
				return "", fmt.Errorf("Could not parse iface file for %s", b.Iface)
			}
			if iscsiTransport == "tcp" {
				devicePath = strings.Join([]string{"/dev/disk/by-path/ip", tp, "iscsi", b.Iqn, "lun", b.Lun}, "-")
			} else {
				devicePath = strings.Join([]string{"/dev/disk/by-path/pci", "*", "ip", tp, "iscsi", b.Iqn, "lun", b.Lun}, "-")
			}

			if exist := waitForPathToExist(&devicePath, deviceDiscoveryTimeout, iscsiTransport); !exist {
				msg := fmt.Sprintf("Timed out waiting for device at path %s after %ds", devicePath, deviceDiscoveryTimeout)
				klog.Error(msg)
				// update last error
				lastErr = errors.New(msg)
				continue
			} else {
				devicePaths[tp] = devicePath
			}
		}
		klog.V(4).Infof("iscsi: tried all devices for %q %d times, %d paths found", b.Iqn, i, len(devicePaths))
		if len(devicePaths) == 0 {
			// No path attached, report error and stop trying. kubelet will try again in a short while
			// delete cloned iface
			execWithLog(b, "iscsiadm", "-m", "iface", "-I", b.Iface, "-o", "delete")
			klog.Errorf("iscsi: failed to get any path for iscsi disk, last err seen:\n%v", lastErr)
			return "", fmt.Errorf("failed to get any path for iscsi disk, last err seen:\n%v", lastErr)
		}
		if len(devicePaths) == len(bkpPortal) {
			// We have all paths
			klog.V(4).Infof("iscsi: all devices for %q found", b.Iqn)
			break
		}
		if len(devicePaths) >= minMultipathCount && i >= minAttachAttempts {
			// We have at least two paths for multipath and we tried the other paths long enough
			klog.V(4).Infof("%d devices found for %q", len(devicePaths), b.Iqn)
			break
		}
	}

	if lastErr != nil {
		klog.Errorf("iscsi: last error occurred during iscsi init:\n%v", lastErr)
	}

	devicePathList := []string{}
	for _, path := range devicePaths {
		devicePathList = append(devicePathList, path)
	}
	// Try to find a multipath device for the volume
	if len(bkpPortal) > 1 {
		// Multipath volume was requested. Wait up to multipathDeviceTimeout seconds for the multipath device to appear.
		devicePath = waitForMultiPathToExist(devicePathList, multipathDeviceTimeout, b.deviceUtil)
	} else {
		// For PVs with 1 portal, just try one time to find the multipath device. This
		// avoids a long pause when the multipath device will never get created, and
		// matches legacy behavior.
		devicePath = waitForMultiPathToExist(devicePathList, 1, b.deviceUtil)
	}

	// When no multipath device is found, just use the first (and presumably only) device
	if devicePath == "" {
		devicePath = devicePathList[0]
	}

	klog.V(5).Infof("iscsi: AttachDisk devicePath: %s", devicePath)

	if err = util.persistISCSI(b); err != nil {
		// Return uncertain error so kubelet calls Unmount / Unmap when the pod
		// is deleted.
		return "", types.NewUncertainProgressError(err.Error())
	}
	return devicePath, nil
}

// persistISCSI saves iSCSI volume configuration for DetachDisk into global
// mount / map directory.
func (util *ISCSIUtil) persistISCSI(b iscsiDiskMounter) error {
	klog.V(5).Infof("iscsi: AttachDisk volumeMode: %s", b.volumeMode)
	var globalPDPath string
	if b.volumeMode == v1.PersistentVolumeBlock {
		globalPDPath = b.manager.MakeGlobalVDPDName(*b.iscsiDisk)
	} else {
		globalPDPath = b.manager.MakeGlobalPDName(*b.iscsiDisk)
	}

	if err := os.MkdirAll(globalPDPath, 0750); err != nil {
		klog.Errorf("iscsi: failed to mkdir %s, error", globalPDPath)
		return err
	}

	if b.volumeMode == v1.PersistentVolumeFilesystem {
		notMnt, err := b.mounter.IsLikelyNotMountPoint(globalPDPath)
		if err != nil {
			return err
		}
		if !notMnt {
			// The volume is already mounted, therefore the previous WaitForAttach must have
			// persisted the volume metadata. In addition, the metadata is actually *inside*
			// globalPDPath and we can't write it here, because it was shadowed by the volume
			// mount.
			klog.V(4).Infof("Skipping persistISCSI, the volume is already mounted at %s", globalPDPath)
			return nil
		}
	}

	// Persist iscsi disk config to json file for DetachDisk path
	return util.persistISCSIFile(*(b.iscsiDisk), globalPDPath)
}

// Delete 1 block device of the form "sd*"
func deleteDevice(deviceName string) error {
	filename := fmt.Sprintf("/sys/block/%s/device/delete", deviceName)
	fd, err := os.OpenFile(filename, os.O_WRONLY, 0)
	if err != nil {
		// The file was not present, so just return without error
		return nil
	}
	defer fd.Close()

	if written, err := fd.WriteString("1"); err != nil {
		return err
	} else if 0 == written {
		return fmt.Errorf("No data written to file: %s", filename)
	}
	klog.V(4).Infof("Deleted block device: %s", deviceName)
	return nil
}

// deleteDevices tries to remove all the block devices and multipath map devices
// associated with a given iscsi device
func deleteDevices(c iscsiDiskUnmounter) error {
	lunNumber, err := strconv.Atoi(c.iscsiDisk.Lun)
	if err != nil {
		klog.Errorf("iscsi delete devices: lun is not a number: %s\nError: %v", c.iscsiDisk.Lun, err)
		return err
	}
	// Enumerate the devices so we can delete them
	deviceNames, err := c.deviceUtil.FindDevicesForISCSILun(c.iscsiDisk.Iqn, lunNumber)
	if err != nil {
		klog.Errorf("iscsi delete devices: could not get devices associated with LUN %d on target %s\nError: %v",
			lunNumber, c.iscsiDisk.Iqn, err)
		return err
	}
	// Find the multipath device path(s)
	mpathDevices := make(map[string]bool)
	for _, deviceName := range deviceNames {
		path := "/dev/" + deviceName
		// check if the dev is using mpio and if so mount it via the dm-XX device
		if mappedDevicePath := c.deviceUtil.FindMultipathDeviceForDevice(path); mappedDevicePath != "" {
			mpathDevices[mappedDevicePath] = true
		}
	}
	// Flush any multipath device maps
	for mpathDevice := range mpathDevices {
		_, err = c.exec.Command("multipath", "-f", mpathDevice).CombinedOutput()
		if err != nil {
			klog.Warningf("Warning: Failed to flush multipath device map: %s\nError: %v", mpathDevice, err)
			// Fall through -- keep deleting the block devices
		}
		klog.V(4).Infof("Flushed multipath device: %s", mpathDevice)
	}
	for _, deviceName := range deviceNames {
		err = deleteDevice(deviceName)
		if err != nil {
			klog.Warningf("Warning: Failed to delete block device: %s\nError: %v", deviceName, err)
			// Fall through -- keep deleting other block devices
		}
	}
	return nil
}

// DetachDisk unmounts and detaches a volume from node
func (util *ISCSIUtil) DetachDisk(c iscsiDiskUnmounter, mntPath string) error {
	if pathExists, pathErr := mount.PathExists(mntPath); pathErr != nil {
		return fmt.Errorf("Error checking if path exists: %v", pathErr)
	} else if !pathExists {
		klog.Warningf("Warning: Unmount skipped because path does not exist: %v", mntPath)
		return nil
	}

	notMnt, err := c.mounter.IsLikelyNotMountPoint(mntPath)
	if err != nil {
		return err
	}
	if !notMnt {
		if err := c.mounter.Unmount(mntPath); err != nil {
			klog.Errorf("iscsi detach disk: failed to unmount: %s\nError: %v", mntPath, err)
			return err
		}
	}

	// if device is no longer used, see if need to logout the target
	device, _, err := extractDeviceAndPrefix(mntPath)
	if err != nil {
		return err
	}

	var bkpPortal []string
	var volName, iqn, iface, initiatorName string
	found := true

	// load iscsi disk config from json file
	if err := util.loadISCSI(c.iscsiDisk, mntPath); err == nil {
		bkpPortal, iqn, iface, volName = c.iscsiDisk.Portals, c.iscsiDisk.Iqn, c.iscsiDisk.Iface, c.iscsiDisk.VolName
		initiatorName = c.iscsiDisk.InitiatorName
	} else {
		// If the iscsi disk config is not found, fall back to the original behavior.
		// This portal/iqn/iface is no longer referenced, log out.
		// Extract the portal and iqn from device path.
		bkpPortal = make([]string, 1)
		bkpPortal[0], iqn, err = extractPortalAndIqn(device)
		if err != nil {
			return err
		}
		// Extract the iface from the mountPath and use it to log out. If the iface
		// is not found, maintain the previous behavior to facilitate kubelet upgrade.
		// Logout may fail as no session may exist for the portal/IQN on the specified interface.
		iface, found = extractIface(mntPath)
	}

	// Delete all the scsi devices and any multipath devices after unmounting
	if err = deleteDevices(c); err != nil {
		klog.Warningf("iscsi detach disk: failed to delete devices\nError: %v", err)
		// Fall through -- even if deleting fails, a logout may fix problems
	}

	// Lock the target while we determine if we can safely log out or not
	c.plugin.targetLocks.LockKey(iqn)
	defer c.plugin.targetLocks.UnlockKey(iqn)

	portals := removeDuplicate(bkpPortal)
	if len(portals) == 0 {
		return fmt.Errorf("iscsi detach disk: failed to detach iscsi disk. Couldn't get connected portals from configurations")
	}

	// If device is no longer used, see if need to logout the target
	if isSessionBusy(c.iscsiDisk.plugin.host, portals[0], iqn) {
		return nil
	}

	err = util.detachISCSIDisk(c.exec, portals, iqn, iface, volName, initiatorName, found)
	if err != nil {
		return fmt.Errorf("failed to finish detachISCSIDisk, err: %v", err)
	}
	return nil
}

// DetachBlockISCSIDisk removes loopback device for a volume and detaches a volume from node
func (util *ISCSIUtil) DetachBlockISCSIDisk(c iscsiDiskUnmapper, mapPath string) error {
	if pathExists, pathErr := mount.PathExists(mapPath); pathErr != nil {
		return fmt.Errorf("Error checking if path exists: %v", pathErr)
	} else if !pathExists {
		klog.Warningf("Warning: Unmap skipped because path does not exist: %v", mapPath)
		return nil
	}
	// If we arrive here, device is no longer used, see if need to logout the target
	// device: 192.168.0.10:3260-iqn.2017-05.com.example:test-lun-0
	device, _, err := extractDeviceAndPrefix(mapPath)
	if err != nil {
		return err
	}
	var bkpPortal []string
	var volName, iqn, lun, iface, initiatorName string
	found := true
	// load iscsi disk config from json file
	if err := util.loadISCSI(c.iscsiDisk, mapPath); err == nil {
		bkpPortal, iqn, lun, iface, volName = c.iscsiDisk.Portals, c.iscsiDisk.Iqn, c.iscsiDisk.Lun, c.iscsiDisk.Iface, c.iscsiDisk.VolName
		initiatorName = c.iscsiDisk.InitiatorName
	} else {
		// If the iscsi disk config is not found, fall back to the original behavior.
		// This portal/iqn/iface is no longer referenced, log out.
		// Extract the portal and iqn from device path.
		bkpPortal = make([]string, 1)
		bkpPortal[0], iqn, err = extractPortalAndIqn(device)
		if err != nil {
			return err
		}
		arr := strings.Split(device, "-lun-")
		if len(arr) < 2 {
			return fmt.Errorf("failed to retrieve lun from mapPath: %v", mapPath)
		}
		lun = arr[1]
		// Extract the iface from the mountPath and use it to log out. If the iface
		// is not found, maintain the previous behavior to facilitate kubelet upgrade.
		// Logout may fail as no session may exist for the portal/IQN on the specified interface.
		iface, found = extractIface(mapPath)
	}
	portals := removeDuplicate(bkpPortal)
	if len(portals) == 0 {
		return fmt.Errorf("iscsi detach disk: failed to detach iscsi disk. Couldn't get connected portals from configurations")
	}

	devicePath := getDevByPath(portals[0], iqn, lun)
	klog.V(5).Infof("iscsi: devicePath: %s", devicePath)
	if _, err = os.Stat(devicePath); err != nil {
		return fmt.Errorf("failed to validate devicePath: %s", devicePath)
	}

	// Lock the target while we determine if we can safely log out or not
	c.plugin.targetLocks.LockKey(iqn)
	defer c.plugin.targetLocks.UnlockKey(iqn)

	// If device is no longer used, see if need to logout the target
	if isSessionBusy(c.iscsiDisk.plugin.host, portals[0], iqn) {
		return nil
	}

	// Detach a volume from kubelet node
	err = util.detachISCSIDisk(c.exec, portals, iqn, iface, volName, initiatorName, found)
	if err != nil {
		return fmt.Errorf("failed to finish detachISCSIDisk, err: %v", err)
	}
	return nil
}

func (util *ISCSIUtil) detachISCSIDisk(exec utilexec.Interface, portals []string, iqn, iface, volName, initiatorName string, found bool) error {
	for _, portal := range portals {
		logoutArgs := []string{"-m", "node", "-p", portal, "-T", iqn, "--logout"}
		deleteArgs := []string{"-m", "node", "-p", portal, "-T", iqn, "-o", "delete"}
		if found {
			logoutArgs = append(logoutArgs, []string{"-I", iface}...)
			deleteArgs = append(deleteArgs, []string{"-I", iface}...)
		}
		klog.Infof("iscsi: log out target %s iqn %s iface %s", portal, iqn, iface)
		out, err := exec.Command("iscsiadm", logoutArgs...).CombinedOutput()
		err = ignoreExitCodes(err, exit_ISCSI_ERR_NO_OBJS_FOUND, exit_ISCSI_ERR_SESS_NOT_FOUND)
		if err != nil {
			klog.Errorf("iscsi: failed to detach disk Error: %s", string(out))
			return err
		}
		// Delete the node record
		klog.Infof("iscsi: delete node record target %s iqn %s", portal, iqn)
		out, err = exec.Command("iscsiadm", deleteArgs...).CombinedOutput()
		err = ignoreExitCodes(err, exit_ISCSI_ERR_NO_OBJS_FOUND, exit_ISCSI_ERR_SESS_NOT_FOUND)
		if err != nil {
			klog.Errorf("iscsi: failed to delete node record Error: %s", string(out))
			return err
		}
	}
	// Delete the iface after all sessions have logged out
	// If the iface is not created via iscsi plugin, skip to delete
	if initiatorName != "" && found && iface == (portals[0]+":"+volName) {
		deleteArgs := []string{"-m", "iface", "-I", iface, "-o", "delete"}
		out, err := exec.Command("iscsiadm", deleteArgs...).CombinedOutput()
		err = ignoreExitCodes(err, exit_ISCSI_ERR_NO_OBJS_FOUND, exit_ISCSI_ERR_SESS_NOT_FOUND)
		if err != nil {
			klog.Errorf("iscsi: failed to delete iface Error: %s", string(out))
			return err
		}
	}

	return nil
}

func getDevByPath(portal, iqn, lun string) string {
	return "/dev/disk/by-path/ip-" + portal + "-iscsi-" + iqn + "-lun-" + lun
}

func extractTransportname(ifaceOutput string) (iscsiTransport string) {
	rexOutput := ifaceTransportNameRe.FindStringSubmatch(ifaceOutput)
	if rexOutput == nil {
		return ""
	}
	iscsiTransport = rexOutput[1]

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
	// strip -lun- from mount path
	ind = strings.LastIndex(mntPath, "-lun-")
	if ind < 0 {
		return "", "", fmt.Errorf("iscsi detach disk: malformatted mnt path: %s", mntPath)
	}
	prefix := mntPath[:ind]
	return device, prefix, nil
}

func extractIface(mntPath string) (string, bool) {
	reOutput := ifaceRe.FindStringSubmatch(mntPath)
	if reOutput != nil && len(reOutput) > 1 {
		return reOutput[1], true
	}

	return "", false
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

// Remove duplicates or string
func removeDuplicate(s []string) []string {
	m := map[string]bool{}
	for _, v := range s {
		if v != "" && !m[v] {
			s[len(m)] = v
			m[v] = true
		}
	}
	s = s[:len(m)]
	return s
}

func parseIscsiadmShow(output string) (map[string]string, error) {
	params := make(map[string]string)
	slice := strings.Split(output, "\n")
	for _, line := range slice {
		if !strings.HasPrefix(line, "iface.") || strings.Contains(line, "<empty>") {
			continue
		}
		iface := strings.Fields(line)
		if len(iface) != 3 || iface[1] != "=" {
			return nil, fmt.Errorf("Error: invalid iface setting: %v", iface)
		}
		// iscsi_ifacename is immutable once the iface is created
		if iface[0] == "iface.iscsi_ifacename" {
			continue
		}
		params[iface[0]] = iface[2]
	}
	return params, nil
}

func cloneIface(b iscsiDiskMounter) error {
	var lastErr error
	if b.InitIface == b.Iface {
		return fmt.Errorf("iscsi: cannot clone iface with same name: %s", b.InitIface)
	}
	// get pre-configured iface records
	out, err := execWithLog(b, "iscsiadm", "-m", "iface", "-I", b.InitIface, "-o", "show")
	if err != nil {
		lastErr = fmt.Errorf("iscsi: failed to show iface records: %s (%v)", out, err)
		return lastErr
	}
	// parse obtained records
	params, err := parseIscsiadmShow(out)
	if err != nil {
		lastErr = fmt.Errorf("iscsi: failed to parse iface records: %s (%v)", out, err)
		return lastErr
	}
	// update initiatorname
	params["iface.initiatorname"] = b.InitiatorName
	// create new iface
	out, err = execWithLog(b, "iscsiadm", "-m", "iface", "-I", b.Iface, "-o", "new")
	if err != nil {
		exit, ok := err.(utilexec.ExitError)
		if ok && exit.ExitStatus() == iscsiadmErrorSessExists {
			klog.Infof("iscsi: there is a session already logged in with iface %s", b.Iface)
		} else {
			lastErr = fmt.Errorf("iscsi: failed to create new iface: %s (%v)", out, err)
			return lastErr
		}
	}
	// Get and sort keys to maintain a stable iteration order
	var keys []string
	for k := range params {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	// update new iface records
	for _, key := range keys {
		_, err = execWithLog(b, "iscsiadm", "-m", "iface", "-I", b.Iface, "-o", "update", "-n", key, "-v", params[key])
		if err != nil {
			execWithLog(b, "iscsiadm", "-m", "iface", "-I", b.Iface, "-o", "delete")
			lastErr = fmt.Errorf("iscsi: failed to update iface records: %s (%v). iface(%s) will be used", out, err, b.InitIface)
			break
		}
	}
	return lastErr
}

// isSessionBusy determines if the iSCSI session is busy by counting both FS and block volumes in use.
func isSessionBusy(host volume.VolumeHost, portal, iqn string) bool {
	fsDir := host.GetPluginDir(iscsiPluginName)
	countFS, err := getVolCount(fsDir, portal, iqn)
	if err != nil {
		klog.Errorf("iscsi: could not determine FS volumes in use: %v", err)
		return true
	}

	blockDir := host.GetVolumeDevicePluginDir(iscsiPluginName)
	countBlock, err := getVolCount(blockDir, portal, iqn)
	if err != nil {
		klog.Errorf("iscsi: could not determine block volumes in use: %v", err)
		return true
	}

	return countFS+countBlock > 1
}

// getVolCount returns the number of volumes in use by the kubelet.
// It does so by counting the number of directories prefixed by the given portal and IQN.
func getVolCount(dir, portal, iqn string) (int, error) {
	// For FileSystem volumes, the topmost dirs are named after the ifaces, e.g., iface-default or iface-127.0.0.1:3260:pv0.
	// For Block volumes, the default topmost dir is volumeDevices.
	contents, err := ioutil.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, err
	}

	// Inside each iface dir, we look for volume dirs prefixed by the given
	// portal + iqn, e.g., 127.0.0.1:3260-iqn.2003-01.io.k8s:e2e.volume-1-lun-2
	var counter int
	for _, c := range contents {
		if !c.IsDir() || c.Name() == config.DefaultKubeletVolumeDevicesDirName {
			continue
		}

		mounts, err := ioutil.ReadDir(filepath.Join(dir, c.Name()))
		if err != nil {
			return 0, err
		}

		for _, m := range mounts {
			volumeMount := m.Name()
			prefix := portal + "-" + iqn
			if strings.HasPrefix(volumeMount, prefix) {
				counter++
			}
		}
	}

	return counter, nil
}

func ignoreExitCodes(err error, ignoredExitCodes ...int) error {
	exitError, ok := err.(utilexec.ExitError)
	if !ok {
		return err
	}
	for _, code := range ignoredExitCodes {
		if exitError.ExitStatus() == code {
			klog.V(4).Infof("ignored iscsiadm exit code %d", code)
			return nil
		}
	}
	return err
}

func execWithLog(b iscsiDiskMounter, cmd string, args ...string) (string, error) {
	start := time.Now()
	out, err := b.exec.Command(cmd, args...).CombinedOutput()
	if klog.V(5).Enabled() {
		d := time.Since(start)
		klog.V(5).Infof("Executed %s %v in %v, err: %v", cmd, args, d, err)
		klog.V(5).Infof("Output: %s", string(out))
	}
	return string(out), err
}
