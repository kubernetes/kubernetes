/*
Copyright 2014 The Kubernetes Authors.

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

//
// utility functions to setup rbd volume
// mainly implement diskManager interface
//

package rbd

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"strconv"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog"
	fileutil "k8s.io/kubernetes/pkg/util/file"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/volume"
	volutil "k8s.io/kubernetes/pkg/volume/util"
)

const (
	imageWatcherStr = "watcher="
	imageSizeStr    = "size "
	kubeLockMagic   = "kubelet_lock_magic_"
	// The following three values are used for 30 seconds timeout
	// while waiting for RBD Watcher to expire.
	rbdImageWatcherInitDelay = 1 * time.Second
	rbdImageWatcherFactor    = 1.4
	rbdImageWatcherSteps     = 10
	rbdImageSizeUnitMiB      = 1024 * 1024
)

func getDevFromImageAndPool(pool, image string) (string, bool) {
	device, found := getRbdDevFromImageAndPool(pool, image)
	if found {
		return device, true
	}
	device, found = getNbdDevFromImageAndPool(pool, image)
	if found {
		return device, true
	}
	return "", false
}

// Search /sys/bus for rbd device that matches given pool and image.
func getRbdDevFromImageAndPool(pool string, image string) (string, bool) {
	// /sys/bus/rbd/devices/X/name and /sys/bus/rbd/devices/X/pool
	sys_path := "/sys/bus/rbd/devices"
	if dirs, err := ioutil.ReadDir(sys_path); err == nil {
		for _, f := range dirs {
			// Pool and name format:
			// see rbd_pool_show() and rbd_name_show() at
			// https://github.com/torvalds/linux/blob/master/drivers/block/rbd.c
			name := f.Name()
			// First match pool, then match name.
			poolFile := path.Join(sys_path, name, "pool")
			poolBytes, err := ioutil.ReadFile(poolFile)
			if err != nil {
				klog.V(4).Infof("error reading %s: %v", poolFile, err)
				continue
			}
			if strings.TrimSpace(string(poolBytes)) != pool {
				klog.V(4).Infof("device %s is not %q: %q", name, pool, string(poolBytes))
				continue
			}
			imgFile := path.Join(sys_path, name, "name")
			imgBytes, err := ioutil.ReadFile(imgFile)
			if err != nil {
				klog.V(4).Infof("error reading %s: %v", imgFile, err)
				continue
			}
			if strings.TrimSpace(string(imgBytes)) != image {
				klog.V(4).Infof("device %s is not %q: %q", name, image, string(imgBytes))
				continue
			}
			// Found a match, check if device exists.
			devicePath := "/dev/rbd" + name
			if _, err := os.Lstat(devicePath); err == nil {
				return devicePath, true
			}
		}
	}
	return "", false
}

func getMaxNbds() (int, error) {

	// the max number of nbd devices may be found in maxNbdsPath
	// we will check sysfs for possible nbd devices even if this is not available
	maxNbdsPath := "/sys/module/nbd/parameters/nbds_max"
	_, err := os.Lstat(maxNbdsPath)
	if err != nil {
		return 0, fmt.Errorf("rbd-nbd: failed to retrieve max_nbds from %s err: %q", maxNbdsPath, err)
	}

	klog.V(4).Infof("found nbds max parameters file at %s", maxNbdsPath)

	maxNbdBytes, err := ioutil.ReadFile(maxNbdsPath)
	if err != nil {
		return 0, fmt.Errorf("rbd-nbd: failed to read max_nbds from %s err: %q", maxNbdsPath, err)
	}

	maxNbds, err := strconv.Atoi(strings.TrimSpace(string(maxNbdBytes)))
	if err != nil {
		return 0, fmt.Errorf("rbd-nbd: failed to read max_nbds err: %q", err)
	}

	klog.V(4).Infof("rbd-nbd: max_nbds: %d", maxNbds)
	return maxNbds, nil
}

// Locate any existing rbd-nbd process mapping given a <pool, image>.
// Recent versions of rbd-nbd tool can correctly provide this info using list-mapped
// but older versions of list-mapped don't.
// The implementation below peeks at the command line of nbd bound processes
// to figure out any mapped images.
func getNbdDevFromImageAndPool(pool string, image string) (string, bool) {
	// nbd module exports the pid of serving process in sysfs
	basePath := "/sys/block/nbd"
	// Do not change imgPath format - some tools like rbd-nbd are strict about it.
	imgPath := fmt.Sprintf("%s/%s", pool, image)

	maxNbds, maxNbdsErr := getMaxNbds()
	if maxNbdsErr != nil {
		klog.V(4).Infof("error reading nbds_max %v", maxNbdsErr)
		return "", false
	}

	for i := 0; i < maxNbds; i++ {
		nbdPath := basePath + strconv.Itoa(i)
		_, err := os.Lstat(nbdPath)
		if err != nil {
			klog.V(4).Infof("error reading nbd info directory %s: %v", nbdPath, err)
			continue
		}
		pidBytes, err := ioutil.ReadFile(path.Join(nbdPath, "pid"))
		if err != nil {
			klog.V(5).Infof("did not find valid pid file in dir %s: %v", nbdPath, err)
			continue
		}
		cmdlineFileName := path.Join("/proc", strings.TrimSpace(string(pidBytes)), "cmdline")
		rawCmdline, err := ioutil.ReadFile(cmdlineFileName)
		if err != nil {
			klog.V(4).Infof("failed to read cmdline file %s: %v", cmdlineFileName, err)
			continue
		}
		cmdlineArgs := strings.FieldsFunc(string(rawCmdline), func(r rune) bool {
			return r == '\u0000'
		})
		// Check if this process is mapping a rbd device.
		// Only accepted pattern of cmdline is from execRbdMap:
		// rbd-nbd map pool/image ...
		if len(cmdlineArgs) < 3 || cmdlineArgs[0] != "rbd-nbd" || cmdlineArgs[1] != "map" {
			klog.V(4).Infof("nbd device %s is not used by rbd", nbdPath)
			continue
		}
		if cmdlineArgs[2] != imgPath {
			klog.V(4).Infof("rbd-nbd device %s did not match expected image path: %s with path found: %s",
				nbdPath, imgPath, cmdlineArgs[2])
			continue
		}
		devicePath := path.Join("/dev", "nbd"+strconv.Itoa(i))
		if _, err := os.Lstat(devicePath); err != nil {
			klog.Warningf("Stat device %s for imgpath %s failed %v", devicePath, imgPath, err)
			continue
		}
		return devicePath, true
	}
	return "", false
}

// Stat a path, if it doesn't exist, retry maxRetries times.
func waitForPath(pool, image string, maxRetries int, useNbdDriver bool) (string, bool) {
	for i := 0; i < maxRetries; i++ {
		if i != 0 {
			time.Sleep(time.Second)
		}
		if useNbdDriver {
			if devicePath, found := getNbdDevFromImageAndPool(pool, image); found {
				return devicePath, true
			}
		} else {
			if devicePath, found := getRbdDevFromImageAndPool(pool, image); found {
				return devicePath, true
			}
		}
	}
	return "", false
}

// Execute command to map a rbd device for mounter.
// rbdCmd is driver dependent and either "rbd" or "rbd-nbd".
func execRbdMap(b rbdMounter, rbdCmd string, mon string) ([]byte, error) {
	// Commandline: rbdCmd map imgPath ...
	// do not change this format - some tools like rbd-nbd are strict about it.
	imgPath := fmt.Sprintf("%s/%s", b.Pool, b.Image)
	if b.Secret != "" {
		return b.exec.Run(rbdCmd,
			"map", imgPath, "--id", b.Id, "-m", mon, "--key="+b.Secret)
	} else {
		return b.exec.Run(rbdCmd,
			"map", imgPath, "--id", b.Id, "-m", mon, "-k", b.Keyring)
	}
}

// Check if rbd-nbd tools are installed.
func checkRbdNbdTools(e mount.Exec) bool {
	_, err := e.Run("modprobe", "nbd")
	if err != nil {
		klog.V(5).Infof("rbd-nbd: nbd modprobe failed with error %v", err)
		return false
	}
	if _, err := e.Run("rbd-nbd", "--version"); err != nil {
		klog.V(5).Infof("rbd-nbd: getting rbd-nbd version failed with error %v", err)
		return false
	}
	klog.V(3).Infof("rbd-nbd tools were found.")
	return true
}

// Make a directory like /var/lib/kubelet/plugins/kubernetes.io/rbd/mounts/pool-image-image.
func makePDNameInternal(host volume.VolumeHost, pool string, image string) string {
	// Backward compatibility for the deprecated format: /var/lib/kubelet/plugins/kubernetes.io/rbd/rbd/pool-image-image.
	deprecatedDir := path.Join(host.GetPluginDir(rbdPluginName), "rbd", pool+"-image-"+image)
	info, err := os.Stat(deprecatedDir)
	if err == nil && info.IsDir() {
		// The device mount path has already been created with the deprecated format, return it.
		klog.V(5).Infof("Deprecated format path %s found", deprecatedDir)
		return deprecatedDir
	}
	// Return the canonical format path.
	return path.Join(host.GetPluginDir(rbdPluginName), mount.MountsInGlobalPDPath, pool+"-image-"+image)
}

// Make a directory like /var/lib/kubelet/plugins/kubernetes.io/rbd/volumeDevices/pool-image-image.
func makeVDPDNameInternal(host volume.VolumeHost, pool string, image string) string {
	return path.Join(host.GetVolumeDevicePluginDir(rbdPluginName), pool+"-image-"+image)
}

// RBDUtil implements diskManager interface.
type RBDUtil struct{}

var _ diskManager = &RBDUtil{}

func (util *RBDUtil) MakeGlobalPDName(rbd rbd) string {
	return makePDNameInternal(rbd.plugin.host, rbd.Pool, rbd.Image)
}

func (util *RBDUtil) MakeGlobalVDPDName(rbd rbd) string {
	return makeVDPDNameInternal(rbd.plugin.host, rbd.Pool, rbd.Image)
}

func rbdErrors(runErr, resultErr error) error {
	if err, ok := runErr.(*exec.Error); ok {
		if err.Err == exec.ErrNotFound {
			return fmt.Errorf("rbd: rbd cmd not found")
		}
	}
	return resultErr
}

// 'rbd' utility builds a comma-separated list of monitor addresses from '-m' /
// '--mon_host` parameter (comma, semi-colon, or white-space delimited monitor
// addresses) and send it to kernel rbd/libceph modules, which can accept
// comma-seprated list of monitor addresses (e.g. ip1[:port1][,ip2[:port2]...])
// in their first version in linux (see
// https://github.com/torvalds/linux/blob/602adf400201636e95c3fed9f31fba54a3d7e844/net/ceph/ceph_common.c#L239).
// Also, libceph module chooses monitor randomly, so we can simply pass all
// addresses without randomization (see
// https://github.com/torvalds/linux/blob/602adf400201636e95c3fed9f31fba54a3d7e844/net/ceph/mon_client.c#L132).
func (util *RBDUtil) kernelRBDMonitorsOpt(mons []string) string {
	return strings.Join(mons, ",")
}

// rbdUnlock releases a lock on image if found.
func (util *RBDUtil) rbdUnlock(b rbdMounter) error {
	var err error
	var output, locker string
	var cmd []byte
	var secret_opt []string

	if b.Secret != "" {
		secret_opt = []string{"--key=" + b.Secret}
	} else {
		secret_opt = []string{"-k", b.Keyring}
	}
	if len(b.adminId) == 0 {
		b.adminId = b.Id
	}
	if len(b.adminSecret) == 0 {
		b.adminSecret = b.Secret
	}

	// Construct lock id using host name and a magic prefix.
	hostName, err := node.GetHostname("")
	if err != nil {
		return err
	}
	lock_id := kubeLockMagic + hostName

	mon := util.kernelRBDMonitorsOpt(b.Mon)

	// Get the locker name, something like "client.1234".
	args := []string{"lock", "list", b.Image, "--pool", b.Pool, "--id", b.Id, "-m", mon}
	args = append(args, secret_opt...)
	cmd, err = b.exec.Run("rbd", args...)
	output = string(cmd)
	klog.V(4).Infof("lock list output %q", output)
	if err != nil {
		return err
	}
	ind := strings.LastIndex(output, lock_id) - 1
	for i := ind; i >= 0; i-- {
		if output[i] == '\n' {
			locker = output[(i + 1):ind]
			break
		}
	}

	// Remove a lock if found: rbd lock remove.
	if len(locker) > 0 {
		args := []string{"lock", "remove", b.Image, lock_id, locker, "--pool", b.Pool, "--id", b.Id, "-m", mon}
		args = append(args, secret_opt...)
		cmd, err = b.exec.Run("rbd", args...)
		if err == nil {
			klog.V(4).Infof("rbd: successfully remove lock (locker_id: %s) on image: %s/%s with id %s mon %s", lock_id, b.Pool, b.Image, b.Id, mon)
		} else {
			klog.Warningf("rbd: failed to remove lock (lock_id: %s) on image: %s/%s with id %s mon %s: %v", lock_id, b.Pool, b.Image, b.Id, mon, err)
		}
	}

	return err
}

// AttachDisk attaches the disk on the node.
func (util *RBDUtil) AttachDisk(b rbdMounter) (string, error) {
	var output []byte

	globalPDPath := util.MakeGlobalPDName(*b.rbd)
	if pathExists, pathErr := volutil.PathExists(globalPDPath); pathErr != nil {
		return "", fmt.Errorf("Error checking if path exists: %v", pathErr)
	} else if !pathExists {
		if err := os.MkdirAll(globalPDPath, 0750); err != nil {
			return "", err
		}
	}

	// Evalute whether this device was mapped with rbd.
	devicePath, mapped := waitForPath(b.Pool, b.Image, 1 /*maxRetries*/, false /*useNbdDriver*/)

	// If rbd-nbd tools are found, we will fallback to it should the default krbd driver fail.
	nbdToolsFound := false

	if !mapped {
		nbdToolsFound = checkRbdNbdTools(b.exec)
		if nbdToolsFound {
			devicePath, mapped = waitForPath(b.Pool, b.Image, 1 /*maxRetries*/, true /*useNbdDriver*/)
		}
	}

	if !mapped {
		// Currently, we don't acquire advisory lock on image, but for backward
		// compatibility, we need to check if the image is being used by nodes running old kubelet.
		// osd_client_watch_timeout defaults to 30 seconds, if the watcher stays active longer than 30 seconds,
		// rbd image does not get mounted and failure message gets generated.
		backoff := wait.Backoff{
			Duration: rbdImageWatcherInitDelay,
			Factor:   rbdImageWatcherFactor,
			Steps:    rbdImageWatcherSteps,
		}
		needValidUsed := true
		if b.accessModes != nil {
			// If accessModes only contains ReadOnlyMany, we don't need check rbd status of being used.
			if len(b.accessModes) == 1 && b.accessModes[0] == v1.ReadOnlyMany {
				needValidUsed = false
			}
		}
		// If accessModes is nil, the volume is referenced by in-line volume.
		// We can assume the AccessModes to be {"RWO" and "ROX"}, which is what the volume plugin supports.
		// We do not need to consider ReadOnly here, because it is used for VolumeMounts.

		if needValidUsed {
			err := wait.ExponentialBackoff(backoff, func() (bool, error) {
				used, rbdOutput, err := util.rbdStatus(&b)
				if err != nil {
					return false, fmt.Errorf("fail to check rbd image status with: (%v), rbd output: (%s)", err, rbdOutput)
				}
				return !used, nil
			})
			// Return error if rbd image has not become available for the specified timeout.
			if err == wait.ErrWaitTimeout {
				return "", fmt.Errorf("rbd image %s/%s is still being used", b.Pool, b.Image)
			}
			// Return error if any other errors were encountered during waiting for the image to become available.
			if err != nil {
				return "", err
			}
		}

		mon := util.kernelRBDMonitorsOpt(b.Mon)
		klog.V(1).Infof("rbd: map mon %s", mon)

		_, err := b.exec.Run("modprobe", "rbd")
		if err != nil {
			klog.Warningf("rbd: failed to load rbd kernel module:%v", err)
		}
		output, err = execRbdMap(b, "rbd", mon)
		if err != nil {
			if !nbdToolsFound {
				klog.V(1).Infof("rbd: map error %v, rbd output: %s", err, string(output))
				return "", fmt.Errorf("rbd: map failed %v, rbd output: %s", err, string(output))
			}
			klog.V(3).Infof("rbd: map failed with %v, %s. Retrying with rbd-nbd", err, string(output))
			errList := []error{err}
			outputList := output
			output, err = execRbdMap(b, "rbd-nbd", mon)
			if err != nil {
				errList = append(errList, err)
				outputList = append(outputList, output...)
				return "", fmt.Errorf("rbd: map failed %v, rbd output: %s", errors.NewAggregate(errList), string(outputList))
			}
			devicePath, mapped = waitForPath(b.Pool, b.Image, 10 /*maxRetries*/, true /*useNbdDrive*/)
		} else {
			devicePath, mapped = waitForPath(b.Pool, b.Image, 10 /*maxRetries*/, false /*useNbdDriver*/)
		}
		if !mapped {
			return "", fmt.Errorf("Could not map image %s/%s, Timeout after 10s", b.Pool, b.Image)
		}
	}
	return devicePath, nil
}

// DetachDisk detaches the disk from the node.
// It detaches device from the node if device is provided, and removes the lock
// if there is persisted RBD info under deviceMountPath.
func (util *RBDUtil) DetachDisk(plugin *rbdPlugin, deviceMountPath string, device string) error {
	if len(device) == 0 {
		return fmt.Errorf("DetachDisk failed , device is empty")
	}

	exec := plugin.host.GetExec(plugin.GetPluginName())

	var rbdCmd string

	// Unlike map, we cannot fallthrough for unmap
	// the tool to unmap is based on device type
	if strings.HasPrefix(device, "/dev/nbd") {
		rbdCmd = "rbd-nbd"
	} else {
		rbdCmd = "rbd"
	}

	// rbd unmap
	output, err := exec.Run(rbdCmd, "unmap", device)
	if err != nil {
		return rbdErrors(err, fmt.Errorf("rbd: failed to unmap device %s, error %v, rbd output: %v", device, err, output))
	}
	klog.V(3).Infof("rbd: successfully unmap device %s", device)

	// Currently, we don't persist rbd info on the disk, but for backward
	// compatbility, we need to clean it if found.
	rbdFile := path.Join(deviceMountPath, "rbd.json")
	exists, err := fileutil.FileExists(rbdFile)
	if err != nil {
		return err
	}
	if exists {
		klog.V(3).Infof("rbd: old rbd.json is found under %s, cleaning it", deviceMountPath)
		err = util.cleanOldRBDFile(plugin, rbdFile)
		if err != nil {
			klog.Errorf("rbd: failed to clean %s", rbdFile)
			return err
		}
		klog.V(3).Infof("rbd: successfully remove %s", rbdFile)
	}
	return nil
}

// DetachBlockDisk detaches the disk from the node.
func (util *RBDUtil) DetachBlockDisk(disk rbdDiskUnmapper, mapPath string) error {

	if pathExists, pathErr := volutil.PathExists(mapPath); pathErr != nil {
		return fmt.Errorf("Error checking if path exists: %v", pathErr)
	} else if !pathExists {
		klog.Warningf("Warning: Unmap skipped because path does not exist: %v", mapPath)
		return nil
	}
	// If we arrive here, device is no longer used, see if we need to logout of the target
	device, err := getBlockVolumeDevice(mapPath)
	if err != nil {
		return err
	}

	if len(device) == 0 {
		return fmt.Errorf("DetachDisk failed , device is empty")
	}

	exec := disk.plugin.host.GetExec(disk.plugin.GetPluginName())

	var rbdCmd string

	// Unlike map, we cannot fallthrough here.
	// Any nbd device must be unmapped by rbd-nbd
	if strings.HasPrefix(device, "/dev/nbd") {
		rbdCmd = "rbd-nbd"
		klog.V(4).Infof("rbd: using rbd-nbd for unmap function")
	} else {
		rbdCmd = "rbd"
		klog.V(4).Infof("rbd: using rbd for unmap function")
	}

	// rbd unmap
	output, err := exec.Run(rbdCmd, "unmap", device)
	if err != nil {
		return rbdErrors(err, fmt.Errorf("rbd: failed to unmap device %s, error %v, rbd output: %s", device, err, string(output)))
	}
	klog.V(3).Infof("rbd: successfully unmap device %s", device)

	return nil
}

// cleanOldRBDFile read rbd info from rbd.json file and removes lock if found.
// At last, it removes rbd.json file.
func (util *RBDUtil) cleanOldRBDFile(plugin *rbdPlugin, rbdFile string) error {
	mounter := &rbdMounter{
		// util.rbdUnlock needs it to run command.
		rbd: newRBD("", "", "", "", false, plugin, util),
	}
	fp, err := os.Open(rbdFile)
	if err != nil {
		return fmt.Errorf("rbd: open err %s/%s", rbdFile, err)
	}
	defer fp.Close()

	decoder := json.NewDecoder(fp)
	if err = decoder.Decode(mounter); err != nil {
		return fmt.Errorf("rbd: decode err: %v.", err)
	}

	if err != nil {
		klog.Errorf("failed to load rbd info from %s: %v", rbdFile, err)
		return err
	}
	// Remove rbd lock if found.
	// The disk is not attached to this node anymore, so the lock on image
	// for this node can be removed safely.
	err = util.rbdUnlock(*mounter)
	if err == nil {
		os.Remove(rbdFile)
	}
	return err
}

func (util *RBDUtil) CreateImage(p *rbdVolumeProvisioner) (r *v1.RBDPersistentVolumeSource, size int, err error) {
	var output []byte
	capacity := p.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	volSizeBytes := capacity.Value()
	// Convert to MB that rbd defaults on.
	sz, err := volutil.RoundUpSizeInt(volSizeBytes, 1024*1024)
	if err != nil {
		return nil, 0, err
	}
	volSz := fmt.Sprintf("%d", sz)
	mon := util.kernelRBDMonitorsOpt(p.Mon)
	if p.rbdMounter.imageFormat == rbdImageFormat2 {
		klog.V(4).Infof("rbd: create %s size %s format %s (features: %s) using mon %s, pool %s id %s key %s", p.rbdMounter.Image, volSz, p.rbdMounter.imageFormat, p.rbdMounter.imageFeatures, mon, p.rbdMounter.Pool, p.rbdMounter.adminId, p.rbdMounter.adminSecret)
	} else {
		klog.V(4).Infof("rbd: create %s size %s format %s using mon %s, pool %s id %s key %s", p.rbdMounter.Image, volSz, p.rbdMounter.imageFormat, mon, p.rbdMounter.Pool, p.rbdMounter.adminId, p.rbdMounter.adminSecret)
	}
	args := []string{"create", p.rbdMounter.Image, "--size", volSz, "--pool", p.rbdMounter.Pool, "--id", p.rbdMounter.adminId, "-m", mon, "--key=" + p.rbdMounter.adminSecret, "--image-format", p.rbdMounter.imageFormat}
	if p.rbdMounter.imageFormat == rbdImageFormat2 {
		// If no image features is provided, it results in empty string
		// which disable all RBD image format 2 features as expected.
		features := strings.Join(p.rbdMounter.imageFeatures, ",")
		args = append(args, "--image-feature", features)
	}
	output, err = p.exec.Run("rbd", args...)

	if err != nil {
		klog.Warningf("failed to create rbd image, output %v", string(output))
		return nil, 0, fmt.Errorf("failed to create rbd image: %v, command output: %s", err, string(output))
	}

	return &v1.RBDPersistentVolumeSource{
		CephMonitors: p.rbdMounter.Mon,
		RBDImage:     p.rbdMounter.Image,
		RBDPool:      p.rbdMounter.Pool,
	}, sz, nil
}

func (util *RBDUtil) DeleteImage(p *rbdVolumeDeleter) error {
	var output []byte
	found, rbdOutput, err := util.rbdStatus(p.rbdMounter)
	if err != nil {
		return fmt.Errorf("error %v, rbd output: %v", err, rbdOutput)
	}
	if found {
		klog.Info("rbd is still being used ", p.rbdMounter.Image)
		return fmt.Errorf("rbd image %s/%s is still being used, rbd output: %v", p.rbdMounter.Pool, p.rbdMounter.Image, rbdOutput)
	}
	// rbd rm.
	mon := util.kernelRBDMonitorsOpt(p.rbdMounter.Mon)
	klog.V(4).Infof("rbd: rm %s using mon %s, pool %s id %s key %s", p.rbdMounter.Image, mon, p.rbdMounter.Pool, p.rbdMounter.adminId, p.rbdMounter.adminSecret)
	output, err = p.exec.Run("rbd",
		"rm", p.rbdMounter.Image, "--pool", p.rbdMounter.Pool, "--id", p.rbdMounter.adminId, "-m", mon, "--key="+p.rbdMounter.adminSecret)
	if err == nil {
		return nil
	}

	klog.Errorf("failed to delete rbd image: %v, command output: %s", err, string(output))
	return fmt.Errorf("error %v, rbd output: %v", err, string(output))
}

// ExpandImage runs rbd resize command to resize the specified image.
func (util *RBDUtil) ExpandImage(rbdExpander *rbdVolumeExpander, oldSize resource.Quantity, newSize resource.Quantity) (resource.Quantity, error) {
	var output []byte
	var err error
	volSizeBytes := newSize.Value()
	// Convert to MB that rbd defaults on.
	sz := int(volutil.RoundUpSize(volSizeBytes, 1024*1024))
	newVolSz := fmt.Sprintf("%d", sz)
	newSizeQuant := resource.MustParse(fmt.Sprintf("%dMi", sz))

	// Check the current size of rbd image, if equals to or greater that the new request size, do nothing.
	curSize, infoErr := util.rbdInfo(rbdExpander.rbdMounter)
	if infoErr != nil {
		return oldSize, fmt.Errorf("rbd info failed, error: %v", infoErr)
	}
	if curSize >= sz {
		return newSizeQuant, nil
	}

	// rbd resize.
	mon := util.kernelRBDMonitorsOpt(rbdExpander.rbdMounter.Mon)
	klog.V(4).Infof("rbd: resize %s using mon %s, pool %s id %s key %s", rbdExpander.rbdMounter.Image, mon, rbdExpander.rbdMounter.Pool, rbdExpander.rbdMounter.adminId, rbdExpander.rbdMounter.adminSecret)
	output, err = rbdExpander.exec.Run("rbd",
		"resize", rbdExpander.rbdMounter.Image, "--size", newVolSz, "--pool", rbdExpander.rbdMounter.Pool, "--id", rbdExpander.rbdMounter.adminId, "-m", mon, "--key="+rbdExpander.rbdMounter.adminSecret)
	if err == nil {
		return newSizeQuant, nil
	}

	klog.Errorf("failed to resize rbd image: %v, command output: %s", err, string(output))
	return oldSize, err
}

// rbdInfo runs `rbd info` command to get the current image size in MB.
func (util *RBDUtil) rbdInfo(b *rbdMounter) (int, error) {
	var err error
	var output []byte

	// If we don't have admin id/secret (e.g. attaching), fallback to user id/secret.
	id := b.adminId
	secret := b.adminSecret
	if id == "" {
		id = b.Id
		secret = b.Secret
	}

	mon := util.kernelRBDMonitorsOpt(b.Mon)
	// cmd "rbd info" get the image info with the following output:
	//
	// # image exists (exit=0)
	// rbd info volume-4a5bcc8b-2b55-46da-ba04-0d3dc5227f08
	//    size 1024 MB in 256 objects
	//    order 22 (4096 kB objects)
	// 	  block_name_prefix: rbd_data.1253ac238e1f29
	//    format: 2
	//    ...
	//
	//  rbd info volume-4a5bcc8b-2b55-46da-ba04-0d3dc5227f08 --format json
	// {"name":"volume-4a5bcc8b-2b55-46da-ba04-0d3dc5227f08","size":1073741824,"objects":256,"order":22,"object_size":4194304,"block_name_prefix":"rbd_data.1253ac238e1f29","format":2,"features":["layering","exclusive-lock","object-map","fast-diff","deep-flatten"],"flags":[]}
	//
	//
	// # image does not exist (exit=2)
	// rbd: error opening image 1234: (2) No such file or directory
	//
	klog.V(4).Infof("rbd: info %s using mon %s, pool %s id %s key %s", b.Image, mon, b.Pool, id, secret)
	output, err = b.exec.Run("rbd",
		"info", b.Image, "--pool", b.Pool, "-m", mon, "--id", id, "--key="+secret, "--format=json")

	if err, ok := err.(*exec.Error); ok {
		if err.Err == exec.ErrNotFound {
			klog.Errorf("rbd cmd not found")
			// fail fast if rbd command is not found.
			return 0, err
		}
	}

	// If command never succeed, returns its last error.
	if err != nil {
		return 0, err
	}

	if len(output) == 0 {
		return 0, fmt.Errorf("can not get image size info %s: %s", b.Image, string(output))
	}

	return getRbdImageSize(output)
}

func getRbdImageSize(output []byte) (int, error) {
	info := struct {
		Size int64 `json:"size"`
	}{}
	if err := json.Unmarshal(output, &info); err != nil {
		return 0, fmt.Errorf("parse rbd info output failed: %s, %v", string(output), err)
	}
	return int(info.Size / rbdImageSizeUnitMiB), nil
}

// rbdStatus runs `rbd status` command to check if there is watcher on the image.
func (util *RBDUtil) rbdStatus(b *rbdMounter) (bool, string, error) {
	var err error
	var output string
	var cmd []byte

	// If we don't have admin id/secret (e.g. attaching), fallback to user id/secret.
	id := b.adminId
	secret := b.adminSecret
	if id == "" {
		id = b.Id
		secret = b.Secret
	}

	mon := util.kernelRBDMonitorsOpt(b.Mon)
	// cmd "rbd status" list the rbd client watch with the following output:
	//
	// # there is a watcher (exit=0)
	// Watchers:
	//   watcher=10.16.153.105:0/710245699 client.14163 cookie=1
	//
	// # there is no watcher (exit=0)
	// Watchers: none
	//
	// Otherwise, exit is non-zero, for example:
	//
	// # image does not exist (exit=2)
	// rbd: error opening image kubernetes-dynamic-pvc-<UUID>: (2) No such file or directory
	//
	klog.V(4).Infof("rbd: status %s using mon %s, pool %s id %s key %s", b.Image, mon, b.Pool, id, secret)
	cmd, err = b.exec.Run("rbd",
		"status", b.Image, "--pool", b.Pool, "-m", mon, "--id", id, "--key="+secret)
	output = string(cmd)

	if err, ok := err.(*exec.Error); ok {
		if err.Err == exec.ErrNotFound {
			klog.Errorf("rbd cmd not found")
			// fail fast if command not found
			return false, output, err
		}
	}

	// If command never succeed, returns its last error.
	if err != nil {
		return false, output, err
	}

	if strings.Contains(output, imageWatcherStr) {
		klog.V(4).Infof("rbd: watchers on %s: %s", b.Image, output)
		return true, output, nil
	} else {
		klog.Warningf("rbd: no watchers on %s", b.Image)
		return false, output, nil
	}
}
