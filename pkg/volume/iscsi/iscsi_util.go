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
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
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

type ISCSIUtil struct{}

func (util *ISCSIUtil) MakeGlobalPDName(iscsi iscsiDisk) string {
	return makePDNameInternal(iscsi.plugin.host, iscsi.Portals[0], iscsi.Iqn, iscsi.lun, iscsi.Iface)
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
	// NOTE: The iscsi config json is not deleted after logging out from target portals.
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

	for _, tp := range bkpPortal {
		// Rescan sessions to discover newly mapped LUNs. Do not specify the interface when rescanning
		// to avoid establishing additional sessions to the same target.
		out, err := b.exec.Run("iscsiadm", "-m", "node", "-p", tp, "-T", b.Iqn, "-R")
		if err != nil {
			glog.Errorf("iscsi: failed to rescan session with error: %s (%v)", string(out), err)
		}

		if iscsiTransport == "" {
			glog.Errorf("iscsi: could not find transport name in iface %s", b.Iface)
			return "", fmt.Errorf("Could not parse iface file for %s", b.Iface)
		}
		if iscsiTransport == "tcp" {
			devicePath = strings.Join([]string{"/dev/disk/by-path/ip", tp, "iscsi", b.Iqn, "lun", b.lun}, "-")
		} else {
			devicePath = strings.Join([]string{"/dev/disk/by-path/pci", "*", "ip", tp, "iscsi", b.Iqn, "lun", b.lun}, "-")
		}

		if exist := waitForPathToExist(&devicePath, 1, iscsiTransport); exist {
			glog.V(4).Infof("iscsi: devicepath (%s) exists", devicePath)
			devicePaths = append(devicePaths, devicePath)
			continue
		}
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

	//Make sure we use a valid devicepath to find mpio device.
	devicePath = devicePaths[0]

	// mount it
	globalPDPath := b.manager.MakeGlobalPDName(*b.iscsiDisk)
	notMnt, err := b.mounter.IsLikelyNotMountPoint(globalPDPath)
	if err != nil && !os.IsNotExist(err) {
		return "", fmt.Errorf("Heuristic determination of mount point failed:%v", err)
	}
	if !notMnt {
		glog.Infof("iscsi: %s already mounted", globalPDPath)
		return "", nil
	}

	if err := os.MkdirAll(globalPDPath, 0750); err != nil {
		glog.Errorf("iscsi: failed to mkdir %s, error", globalPDPath)
		return "", err
	}

	// Persist iscsi disk config to json file for DetachDisk path
	util.persistISCSI(*(b.iscsiDisk), globalPDPath)

	for _, path := range devicePaths {
		// There shouldnt be any empty device paths. However adding this check
		// for safer side to avoid the possibility of an empty entry.
		if path == "" {
			continue
		}
		// check if the dev is using mpio and if so mount it via the dm-XX device
		if mappedDevicePath := b.deviceUtil.FindMultipathDeviceForDevice(path); mappedDevicePath != "" {
			devicePath = mappedDevicePath
			break
		}
	}
	err = b.mounter.FormatAndMount(devicePath, globalPDPath, b.fsType, nil)
	if err != nil {
		glog.Errorf("iscsi: failed to mount iscsi volume %s [%s] to %s, error %v", devicePath, b.fsType, globalPDPath, err)
	}

	return devicePath, err
}

func (util *ISCSIUtil) DetachDisk(c iscsiDiskUnmounter, mntPath string) error {
	_, cnt, err := mount.GetDeviceNameFromMount(c.mounter, mntPath)
	if err != nil {
		glog.Errorf("iscsi detach disk: failed to get device from mnt: %s\nError: %v", mntPath, err)
		return err
	}
	if pathExists, pathErr := volumeutil.PathExists(mntPath); pathErr != nil {
		return fmt.Errorf("Error checking if path exists: %v", pathErr)
	} else if !pathExists {
		glog.Warningf("Warning: Unmount skipped because path does not exist: %v", mntPath)
		return nil
	}
	if err = c.mounter.Unmount(mntPath); err != nil {
		glog.Errorf("iscsi detach disk: failed to unmount: %s\nError: %v", mntPath, err)
		return err
	}
	cnt--
	if cnt != 0 {
		return nil
	}
	// if device is no longer used, see if need to logout the target
	device, prefix, err := extractDeviceAndPrefix(mntPath)
	if err != nil {
		return err
	}
	refCount, err := getDevicePrefixRefCount(c.mounter, prefix)
	if err != nil || refCount != 0 {
		return nil
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
	portals := removeDuplicate(bkpPortal)
	if len(portals) == 0 {
		return fmt.Errorf("iscsi detach disk: failed to detach iscsi disk. Couldn't get connected portals from configurations.")
	}

	for _, portal := range portals {
		logoutArgs := []string{"-m", "node", "-p", portal, "-T", iqn, "--logout"}
		deleteArgs := []string{"-m", "node", "-p", portal, "-T", iqn, "-o", "delete"}
		if found {
			logoutArgs = append(logoutArgs, []string{"-I", iface}...)
			deleteArgs = append(deleteArgs, []string{"-I", iface}...)
		}
		glog.Infof("iscsi: log out target %s iqn %s iface %s", portal, iqn, iface)
		out, err := c.exec.Run("iscsiadm", logoutArgs...)
		if err != nil {
			glog.Errorf("iscsi: failed to detach disk Error: %s", string(out))
		}
		// Delete the node record
		glog.Infof("iscsi: delete node record target %s iqn %s", portal, iqn)
		out, err = c.exec.Run("iscsiadm", deleteArgs...)
		if err != nil {
			glog.Errorf("iscsi: failed to delete node record Error: %s", string(out))
		}
	}
	// Delete the iface after all sessions have logged out
	// If the iface is not created via iscsi plugin, skip to delete
	if initiatorName != "" && found && iface == (portals[0]+":"+volName) {
		deleteArgs := []string{"-m", "iface", "-I", iface, "-o", "delete"}
		out, err := c.exec.Run("iscsiadm", deleteArgs...)
		if err != nil {
			glog.Errorf("iscsi: failed to delete iface Error: %s", string(out))
		}
	}

	return nil
}

func extractTransportname(ifaceOutput string) (iscsiTransport string) {
	re := regexp.MustCompile(`iface.transport_name = (.*)\n`)

	rexOutput := re.FindStringSubmatch(ifaceOutput)
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
	re := regexp.MustCompile(`.+/iface-([^/]+)/.+`)

	reOutput := re.FindStringSubmatch(mntPath)
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
