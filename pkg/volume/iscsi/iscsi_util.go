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
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/volumepathhandler"
)

var (
	chap_st = []string{
		"discovery.sendtargets.auth.username",
		"discovery.sendtargets.auth.password",
		"discovery.sendtargets.auth.username_in",
		"discovery.sendtargets.auth.password_in"}
	chap_sess = []string{
		"node.session.auth.username",
		"node.session.auth.password",
		"node.session.auth.username_in",
		"node.session.auth.password_in"}
	ifaceTransportNameRe = regexp.MustCompile(`iface.transport_name = (.*)\n`)
	ifaceRe              = regexp.MustCompile(`.+/iface-([^/]+)/.+`)
)

func updateISCSIDiscoverydb(b iscsiDiskMounter, tp string) error {
	if !b.chap_discovery {
		return nil
	}
	out, err := b.exec.Run("iscsiadm", "-m", "discoverydb", "-t", "sendtargets", "-p", tp, "-I", b.Iface, "-o", "update", "-n", "discovery.sendtargets.auth.authmethod", "-v", "CHAP")
	if err != nil {
		return fmt.Errorf("iscsi: failed to update discoverydb with CHAP, output: %v", string(out))
	}

	for _, k := range chap_st {
		v := b.secret[k]
		if len(v) > 0 {
			out, err := b.exec.Run("iscsiadm", "-m", "discoverydb", "-t", "sendtargets", "-p", tp, "-I", b.Iface, "-o", "update", "-n", k, "-v", v)
			if err != nil {
				return fmt.Errorf("iscsi: failed to update discoverydb key %q with value %q error: %v", k, v, string(out))
			}
		}
	}
	return nil
}

func updateISCSINode(b iscsiDiskMounter, tp string) error {
	if !b.chap_session {
		return nil
	}

	out, err := b.exec.Run("iscsiadm", "-m", "node", "-p", tp, "-T", b.Iqn, "-I", b.Iface, "-o", "update", "-n", "node.session.auth.authmethod", "-v", "CHAP")
	if err != nil {
		return fmt.Errorf("iscsi: failed to update node with CHAP, output: %v", string(out))
	}

	for _, k := range chap_sess {
		v := b.secret[k]
		if len(v) > 0 {
			out, err := b.exec.Run("iscsiadm", "-m", "node", "-p", tp, "-T", b.Iqn, "-I", b.Iface, "-o", "update", "-n", k, "-v", v)
			if err != nil {
				return fmt.Errorf("iscsi: failed to update node session key %q with value %q error: %v", k, v, string(out))
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

// make a directory like /var/lib/kubelet/plugins/kubernetes.io/iscsi/iface_name/portal-some_iqn-lun-lun_id
func makePDNameInternal(host volume.VolumeHost, portal string, iqn string, lun string, iface string) string {
	return path.Join(host.GetPluginDir(iscsiPluginName), "iface-"+iface, portal+"-"+iqn+"-lun-"+lun)
}

// make a directory like /var/lib/kubelet/plugins/kubernetes.io/iscsi/volumeDevices/iface_name/portal-some_iqn-lun-lun_id
func makeVDPDNameInternal(host volume.VolumeHost, portal string, iqn string, lun string, iface string) string {
	return path.Join(host.GetVolumeDevicePluginDir(iscsiPluginName), "iface-"+iface, portal+"-"+iqn+"-lun-"+lun)
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

func (util *ISCSIUtil) persistISCSI(conf iscsiDisk, mnt string) error {
	file := path.Join(mnt, "iscsi.json")
	fp, err := os.Create(file)
	if err != nil {
		return fmt.Errorf("iscsi: create %s err %s", file, err)
	}
	defer fp.Close()
	encoder := json.NewEncoder(fp)
	if err = encoder.Encode(conf); err != nil {
		return fmt.Errorf("iscsi: encode err: %v.", err)
	}
	return nil
}

func (util *ISCSIUtil) loadISCSI(conf *iscsiDisk, mnt string) error {
	file := path.Join(mnt, "iscsi.json")
	fp, err := os.Open(file)
	if err != nil {
		return fmt.Errorf("iscsi: open %s err %s", file, err)
	}
	defer fp.Close()
	decoder := json.NewDecoder(fp)
	if err = decoder.Decode(conf); err != nil {
		return fmt.Errorf("iscsi: decode err: %v.", err)
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

	glog.V(3).Infof("Scanned SCSI host %d LUN %d", hostNumber, lunNumber)
	return nil
}

func waitForMultiPathToExist(devicePaths []string, maxRetries int, deviceUtil volumeutil.DeviceUtil) string {
	if 0 == len(devicePaths) {
		return ""
	}

	for i := 0; i < maxRetries; i++ {
		for _, path := range devicePaths {
			// There shouldnt be any empty device paths. However adding this check
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
	var devicePaths []string
	var iscsiTransport string
	var lastErr error

	out, err := b.exec.Run("iscsiadm", "-m", "iface", "-I", b.Iface, "-o", "show")
	if err != nil {
		glog.Errorf("iscsi: could not read iface %s error: %s", b.Iface, string(out))
		return "", err
	}

	iscsiTransport = extractTransportname(string(out))

	bkpPortal := b.Portals

	// create new iface and copy parameters from pre-configured iface to the created iface
	if b.InitiatorName != "" {
		// new iface name is <target portal>:<volume name>
		newIface := bkpPortal[0] + ":" + b.VolName
		err = cloneIface(b, newIface)
		if err != nil {
			glog.Errorf("iscsi: failed to clone iface: %s error: %v", b.Iface, err)
			return "", err
		}
		// update iface name
		b.Iface = newIface
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
	glog.V(4).Infof("AttachDisk portal->host map for %s is %v", b.Iqn, portalHostMap)

	for _, tp := range bkpPortal {
		hostNumber, loggedIn := portalHostMap[tp]
		if !loggedIn {
			glog.V(4).Infof("Could not get SCSI host number for portal %s, will attempt login", tp)

			// build discoverydb and discover iscsi target
			b.exec.Run("iscsiadm", "-m", "discoverydb", "-t", "sendtargets", "-p", tp, "-I", b.Iface, "-o", "new")
			// update discoverydb with CHAP secret
			err = updateISCSIDiscoverydb(b, tp)
			if err != nil {
				lastErr = fmt.Errorf("iscsi: failed to update discoverydb to portal %s error: %v", tp, err)
				continue
			}
			out, err = b.exec.Run("iscsiadm", "-m", "discoverydb", "-t", "sendtargets", "-p", tp, "-I", b.Iface, "--discover")
			if err != nil {
				// delete discoverydb record
				b.exec.Run("iscsiadm", "-m", "discoverydb", "-t", "sendtargets", "-p", tp, "-I", b.Iface, "-o", "delete")
				lastErr = fmt.Errorf("iscsi: failed to sendtargets to portal %s output: %s, err %v", tp, string(out), err)
				continue
			}
			err = updateISCSINode(b, tp)
			if err != nil {
				// failure to update node db is rare. But deleting record will likely impact those who already start using it.
				lastErr = fmt.Errorf("iscsi: failed to update iscsi node to portal %s error: %v", tp, err)
				continue
			}
			// login to iscsi target
			out, err = b.exec.Run("iscsiadm", "-m", "node", "-p", tp, "-T", b.Iqn, "-I", b.Iface, "--login")
			if err != nil {
				// delete the node record from database
				b.exec.Run("iscsiadm", "-m", "node", "-p", tp, "-I", b.Iface, "-T", b.Iqn, "-o", "delete")
				lastErr = fmt.Errorf("iscsi: failed to attach disk: Error: %s (%v)", string(out), err)
				continue
			}
			// in case of node failure/restart, explicitly set to manual login so it doesn't hang on boot
			out, err = b.exec.Run("iscsiadm", "-m", "node", "-p", tp, "-T", b.Iqn, "-o", "update", "-n", "node.startup", "-v", "manual")
			if err != nil {
				// don't fail if we can't set startup mode, but log warning so there is a clue
				glog.Warningf("Warning: Failed to set iSCSI login mode to manual. Error: %v", err)
			}

			// Rebuild the host map after logging in
			portalHostMap, err := b.deviceUtil.GetISCSIPortalHostMapForTarget(b.Iqn)
			if err != nil {
				return "", err
			}
			glog.V(6).Infof("AttachDisk portal->host map for %s is %v", b.Iqn, portalHostMap)

			hostNumber, loggedIn = portalHostMap[tp]
			if !loggedIn {
				glog.Warningf("Could not get SCSI host number for portal %s after logging in", tp)
				continue
			}
		}

		glog.V(5).Infof("AttachDisk: scanning SCSI host %d LUN %s", hostNumber, b.Lun)
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
			glog.Errorf("iscsi: could not find transport name in iface %s", b.Iface)
			return "", fmt.Errorf("Could not parse iface file for %s", b.Iface)
		}
		if iscsiTransport == "tcp" {
			devicePath = strings.Join([]string{"/dev/disk/by-path/ip", tp, "iscsi", b.Iqn, "lun", b.Lun}, "-")
		} else {
			devicePath = strings.Join([]string{"/dev/disk/by-path/pci", "*", "ip", tp, "iscsi", b.Iqn, "lun", b.Lun}, "-")
		}

		if exist := waitForPathToExist(&devicePath, 10, iscsiTransport); !exist {
			glog.Errorf("Could not attach disk: Timeout after 10s")
			// update last error
			lastErr = fmt.Errorf("Could not attach disk: Timeout after 10s")
			continue
		} else {
			devicePaths = append(devicePaths, devicePath)
		}
	}

	if len(devicePaths) == 0 {
		// delete cloned iface
		b.exec.Run("iscsiadm", "-m", "iface", "-I", b.Iface, "-o", "delete")
		glog.Errorf("iscsi: failed to get any path for iscsi disk, last err seen:\n%v", lastErr)
		return "", fmt.Errorf("failed to get any path for iscsi disk, last err seen:\n%v", lastErr)
	}
	if lastErr != nil {
		glog.Errorf("iscsi: last error occurred during iscsi init:\n%v", lastErr)
	}

	// Try to find a multipath device for the volume
	if 1 < len(bkpPortal) {
		// If the PV has 2 or more portals, wait up to 10 seconds for the multipath
		// device to appear
		devicePath = waitForMultiPathToExist(devicePaths, 10, b.deviceUtil)
	} else {
		// For PVs with 1 portal, just try one time to find the multipath device. This
		// avoids a long pause when the multipath device will never get created, and
		// matches legacy behavior.
		devicePath = waitForMultiPathToExist(devicePaths, 1, b.deviceUtil)
	}

	// When no multipath device is found, just use the first (and presumably only) device
	if devicePath == "" {
		devicePath = devicePaths[0]
	}

	glog.V(5).Infof("iscsi: AttachDisk devicePath: %s", devicePath)
	// run global mount path related operations based on volumeMode
	return globalPDPathOperation(b)(b, devicePath, util)
}

// globalPDPathOperation returns global mount path related operations based on volumeMode.
// If the volumeMode is 'Filesystem' or not defined, plugin needs to create a dir, persist
// iscsi configurations, and then format/mount the volume.
// If the volumeMode is 'Block', plugin creates a dir and persists iscsi configurations.
// Since volume type is block, plugin doesn't need to format/mount the volume.
func globalPDPathOperation(b iscsiDiskMounter) func(iscsiDiskMounter, string, *ISCSIUtil) (string, error) {
	// TODO: remove feature gate check after no longer needed
	if utilfeature.DefaultFeatureGate.Enabled(features.BlockVolume) {
		glog.V(5).Infof("iscsi: AttachDisk volumeMode: %s", b.volumeMode)
		if b.volumeMode == v1.PersistentVolumeBlock {
			// If the volumeMode is 'Block', plugin don't need to format the volume.
			return func(b iscsiDiskMounter, devicePath string, util *ISCSIUtil) (string, error) {
				globalPDPath := b.manager.MakeGlobalVDPDName(*b.iscsiDisk)
				// Create dir like /var/lib/kubelet/plugins/kubernetes.io/iscsi/volumeDevices/{ifaceName}/{portal-some_iqn-lun-lun_id}
				if err := os.MkdirAll(globalPDPath, 0750); err != nil {
					glog.Errorf("iscsi: failed to mkdir %s, error", globalPDPath)
					return "", err
				}
				// Persist iscsi disk config to json file for DetachDisk path
				util.persistISCSI(*(b.iscsiDisk), globalPDPath)

				return devicePath, nil
			}
		}
	}
	// If the volumeMode is 'Filesystem', plugin needs to format the volume
	// and mount it to globalPDPath.
	return func(b iscsiDiskMounter, devicePath string, util *ISCSIUtil) (string, error) {
		globalPDPath := b.manager.MakeGlobalPDName(*b.iscsiDisk)
		notMnt, err := b.mounter.IsLikelyNotMountPoint(globalPDPath)
		if err != nil && !os.IsNotExist(err) {
			return "", fmt.Errorf("Heuristic determination of mount point failed:%v", err)
		}
		// Return confirmed devicePath to caller
		if !notMnt {
			glog.Infof("iscsi: %s already mounted", globalPDPath)
			return devicePath, nil
		}
		// Create dir like /var/lib/kubelet/plugins/kubernetes.io/iscsi/{ifaceName}/{portal-some_iqn-lun-lun_id}
		if err := os.MkdirAll(globalPDPath, 0750); err != nil {
			glog.Errorf("iscsi: failed to mkdir %s, error", globalPDPath)
			return "", err
		}
		// Persist iscsi disk config to json file for DetachDisk path
		util.persistISCSI(*(b.iscsiDisk), globalPDPath)

		err = b.mounter.FormatAndMount(devicePath, globalPDPath, b.fsType, nil)
		if err != nil {
			glog.Errorf("iscsi: failed to mount iscsi volume %s [%s] to %s, error %v", devicePath, b.fsType, globalPDPath, err)
		}

		return devicePath, nil
	}
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
	glog.V(4).Infof("Deleted block device: %s", deviceName)
	return nil
}

// deleteDevices tries to remove all the block devices and multipath map devices
// associated with a given iscsi device
func deleteDevices(c iscsiDiskUnmounter) error {
	lunNumber, err := strconv.Atoi(c.iscsiDisk.Lun)
	if err != nil {
		glog.Errorf("iscsi delete devices: lun is not a number: %s\nError: %v", c.iscsiDisk.Lun, err)
		return err
	}
	// Enumerate the devices so we can delete them
	deviceNames, err := c.deviceUtil.FindDevicesForISCSILun(c.iscsiDisk.Iqn, lunNumber)
	if err != nil {
		glog.Errorf("iscsi delete devices: could not get devices associated with LUN %d on target %s\nError: %v",
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
		_, err = c.exec.Run("multipath", "-f", mpathDevice)
		if err != nil {
			glog.Warningf("Warning: Failed to flush multipath device map: %s\nError: %v", mpathDevice, err)
			// Fall through -- keep deleting the block devices
		}
		glog.V(4).Infof("Flushed multipath device: %s", mpathDevice)
	}
	for _, deviceName := range deviceNames {
		err = deleteDevice(deviceName)
		if err != nil {
			glog.Warningf("Warning: Failed to delete block device: %s\nError: %v", deviceName, err)
			// Fall through -- keep deleting other block devices
		}
	}
	return nil
}

// DetachDisk unmounts and detaches a volume from node
func (util *ISCSIUtil) DetachDisk(c iscsiDiskUnmounter, mntPath string) error {
	if pathExists, pathErr := volumeutil.PathExists(mntPath); pathErr != nil {
		return fmt.Errorf("Error checking if path exists: %v", pathErr)
	} else if !pathExists {
		glog.Warningf("Warning: Unmount skipped because path does not exist: %v", mntPath)
		return nil
	}

	notMnt, err := c.mounter.IsLikelyNotMountPoint(mntPath)
	if err != nil {
		return err
	}
	if !notMnt {
		if err := c.mounter.Unmount(mntPath); err != nil {
			glog.Errorf("iscsi detach disk: failed to unmount: %s\nError: %v", mntPath, err)
			return err
		}
	}

	// if device is no longer used, see if need to logout the target
	device, prefix, err := extractDeviceAndPrefix(mntPath)
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
		glog.Warningf("iscsi detach disk: failed to delete devices\nError: %v", err)
		// Fall through -- even if deleting fails, a logout may fix problems
	}

	// Lock the target while we determine if we can safely log out or not
	c.plugin.targetLocks.LockKey(iqn)
	defer c.plugin.targetLocks.UnlockKey(iqn)

	// if device is no longer used, see if need to logout the target
	refCount, err := getDevicePrefixRefCount(c.mounter, prefix)
	if err != nil || refCount != 0 {
		return nil
	}

	portals := removeDuplicate(bkpPortal)
	if len(portals) == 0 {
		return fmt.Errorf("iscsi detach disk: failed to detach iscsi disk. Couldn't get connected portals from configurations")
	}

	err = util.detachISCSIDisk(c.exec, portals, iqn, iface, volName, initiatorName, found)
	if err != nil {
		return fmt.Errorf("failed to finish detachISCSIDisk, err: %v", err)
	}
	return nil
}

// DetachBlockISCSIDisk removes loopback device for a volume and detaches a volume from node
func (util *ISCSIUtil) DetachBlockISCSIDisk(c iscsiDiskUnmapper, mapPath string) error {
	if pathExists, pathErr := volumeutil.PathExists(mapPath); pathErr != nil {
		return fmt.Errorf("Error checking if path exists: %v", pathErr)
	} else if !pathExists {
		glog.Warningf("Warning: Unmap skipped because path does not exist: %v", mapPath)
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
	glog.V(5).Infof("iscsi: devicePath: %s", devicePath)
	if _, err = os.Stat(devicePath); err != nil {
		return fmt.Errorf("failed to validate devicePath: %s", devicePath)
	}
	// check if the dev is using mpio and if so mount it via the dm-XX device
	if mappedDevicePath := c.deviceUtil.FindMultipathDeviceForDevice(devicePath); mappedDevicePath != "" {
		devicePath = mappedDevicePath
	}
	// Get loopback device which takes fd lock for devicePath before detaching a volume from node.
	// TODO: This is a workaround for issue #54108
	// Currently local attach plugins such as FC, iSCSI, RBD can't obtain devicePath during
	// GenerateUnmapDeviceFunc() in operation_generator. As a result, these plugins fail to get
	// and remove loopback device then it will be remained on kubelet node. To avoid the problem,
	// local attach plugins needs to remove loopback device during TearDownDevice().
	blkUtil := volumepathhandler.NewBlockVolumePathHandler()
	loop, err := volumepathhandler.BlockVolumePathHandler.GetLoopDevice(blkUtil, devicePath)
	if err != nil {
		if err.Error() != volumepathhandler.ErrDeviceNotFound {
			return fmt.Errorf("failed to get loopback for device: %v, err: %v", devicePath, err)
		}
		glog.Warningf("iscsi: loopback for device: %s not found", device)
	}
	// Detach a volume from kubelet node
	err = util.detachISCSIDisk(c.exec, portals, iqn, iface, volName, initiatorName, found)
	if err != nil {
		return fmt.Errorf("failed to finish detachISCSIDisk, err: %v", err)
	}
	if len(loop) != 0 {
		// The volume was successfully detached from node. We can safely remove the loopback.
		err = volumepathhandler.BlockVolumePathHandler.RemoveLoopDevice(blkUtil, loop)
		if err != nil {
			return fmt.Errorf("failed to remove loopback :%v, err: %v", loop, err)
		}
	}
	return nil
}

func (util *ISCSIUtil) detachISCSIDisk(exec mount.Exec, portals []string, iqn, iface, volName, initiatorName string, found bool) error {
	for _, portal := range portals {
		logoutArgs := []string{"-m", "node", "-p", portal, "-T", iqn, "--logout"}
		deleteArgs := []string{"-m", "node", "-p", portal, "-T", iqn, "-o", "delete"}
		if found {
			logoutArgs = append(logoutArgs, []string{"-I", iface}...)
			deleteArgs = append(deleteArgs, []string{"-I", iface}...)
		}
		glog.Infof("iscsi: log out target %s iqn %s iface %s", portal, iqn, iface)
		out, err := exec.Run("iscsiadm", logoutArgs...)
		if err != nil {
			glog.Errorf("iscsi: failed to detach disk Error: %s", string(out))
		}
		// Delete the node record
		glog.Infof("iscsi: delete node record target %s iqn %s", portal, iqn)
		out, err = exec.Run("iscsiadm", deleteArgs...)
		if err != nil {
			glog.Errorf("iscsi: failed to delete node record Error: %s", string(out))
		}
	}
	// Delete the iface after all sessions have logged out
	// If the iface is not created via iscsi plugin, skip to delete
	if initiatorName != "" && found && iface == (portals[0]+":"+volName) {
		deleteArgs := []string{"-m", "iface", "-I", iface, "-o", "delete"}
		out, err := exec.Run("iscsiadm", deleteArgs...)
		if err != nil {
			glog.Errorf("iscsi: failed to delete iface Error: %s", string(out))
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
	if reOutput != nil {
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

func cloneIface(b iscsiDiskMounter, newIface string) error {
	var lastErr error
	// get pre-configured iface records
	out, err := b.exec.Run("iscsiadm", "-m", "iface", "-I", b.Iface, "-o", "show")
	if err != nil {
		lastErr = fmt.Errorf("iscsi: failed to show iface records: %s (%v)", string(out), err)
		return lastErr
	}
	// parse obtained records
	params, err := parseIscsiadmShow(string(out))
	if err != nil {
		lastErr = fmt.Errorf("iscsi: failed to parse iface records: %s (%v)", string(out), err)
		return lastErr
	}
	// update initiatorname
	params["iface.initiatorname"] = b.InitiatorName
	// create new iface
	out, err = b.exec.Run("iscsiadm", "-m", "iface", "-I", newIface, "-o", "new")
	if err != nil {
		lastErr = fmt.Errorf("iscsi: failed to create new iface: %s (%v)", string(out), err)
		return lastErr
	}
	// update new iface records
	for key, val := range params {
		_, err = b.exec.Run("iscsiadm", "-m", "iface", "-I", newIface, "-o", "update", "-n", key, "-v", val)
		if err != nil {
			b.exec.Run("iscsiadm", "-m", "iface", "-I", newIface, "-o", "delete")
			lastErr = fmt.Errorf("iscsi: failed to update iface records: %s (%v). iface(%s) will be used", string(out), err, b.Iface)
			break
		}
	}
	return lastErr
}
