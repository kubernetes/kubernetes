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
	"errors"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"path"
	"regexp"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	imageWatcherStr = "watcher="
	kubeLockMagic   = "kubelet_lock_magic_"
	rbdCmdErr       = "executable file not found in $PATH"
)

// search /sys/bus for rbd device that matches given pool and image
func getDevFromImageAndPool(pool, image string) (string, bool) {
	// /sys/bus/rbd/devices/X/name and /sys/bus/rbd/devices/X/pool
	sys_path := "/sys/bus/rbd/devices"
	if dirs, err := ioutil.ReadDir(sys_path); err == nil {
		for _, f := range dirs {
			// pool and name format:
			// see rbd_pool_show() and rbd_name_show() at
			// https://github.com/torvalds/linux/blob/master/drivers/block/rbd.c
			name := f.Name()
			// first match pool, then match name
			poolFile := path.Join(sys_path, name, "pool")
			poolBytes, err := ioutil.ReadFile(poolFile)
			if err != nil {
				glog.V(4).Infof("Error reading %s: %v", poolFile, err)
				continue
			}
			if strings.TrimSpace(string(poolBytes)) != pool {
				glog.V(4).Infof("Device %s is not %q: %q", name, pool, string(poolBytes))
				continue
			}
			imgFile := path.Join(sys_path, name, "name")
			imgBytes, err := ioutil.ReadFile(imgFile)
			if err != nil {
				glog.V(4).Infof("Error reading %s: %v", imgFile, err)
				continue
			}
			if strings.TrimSpace(string(imgBytes)) != image {
				glog.V(4).Infof("Device %s is not %q: %q", name, image, string(imgBytes))
				continue
			}
			// found a match, check if device exists
			devicePath := "/dev/rbd" + name
			if _, err := os.Lstat(devicePath); err == nil {
				return devicePath, true
			}
		}
	}
	return "", false
}

// stat a path, if not exists, retry maxRetries times
func waitForPath(pool, image string, maxRetries int) (string, bool) {
	for i := 0; i < maxRetries; i++ {
		devicePath, found := getDevFromImageAndPool(pool, image)
		if found {
			return devicePath, true
		}
		if i == maxRetries-1 {
			break
		}
		time.Sleep(time.Second)
	}
	return "", false
}

// make a directory like /var/lib/kubelet/plugins/kubernetes.io/pod/rbd/pool-image-image
func makePDNameInternal(host volume.VolumeHost, pool string, image string) string {
	return path.Join(host.GetPluginDir(rbdPluginName), "rbd", pool+"-image-"+image)
}

type RBDUtil struct{}

func (util *RBDUtil) MakeGlobalPDName(rbd rbd) string {
	return makePDNameInternal(rbd.plugin.host, rbd.Pool, rbd.Image)
}
func rbdErrors(runErr, resultErr error) error {
	if runErr.Error() == rbdCmdErr {
		return fmt.Errorf("rbd: rbd cmd not found")
	}
	return resultErr
}

func (util *RBDUtil) rbdLock(b rbdMounter, lock bool) error {
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

	// construct lock id using host name and a magic prefix
	lock_id := kubeLockMagic + node.GetHostname("")

	l := len(b.Mon)
	// avoid mount storm, pick a host randomly
	start := rand.Int() % l
	// iterate all hosts until mount succeeds.
	for i := start; i < start+l; i++ {
		mon := b.Mon[i%l]
		// cmd "rbd lock list" serves two purposes:
		// for fencing, check if lock already held for this host
		// this edge case happens if host crashes in the middle of acquiring lock and mounting rbd
		// for defencing, get the locker name, something like "client.1234"
		args := []string{"lock", "list", b.Image, "--pool", b.Pool, "--id", b.Id, "-m", mon}
		args = append(args, secret_opt...)
		cmd, err = b.exec.Run("rbd", args...)
		output = string(cmd)
		glog.Infof("lock list output %q", output)
		if err != nil {
			continue
		}

		if lock {
			// check if lock is already held for this host by matching lock_id and rbd lock id
			if strings.Contains(output, lock_id) {
				// this host already holds the lock, exit
				glog.V(1).Infof("rbd: lock already held for %s", lock_id)
				return nil
			}
			// clean up orphaned lock if no watcher on the image
			used, statusErr := util.rbdStatus(&b)
			if statusErr == nil && !used {
				re := regexp.MustCompile("client.* " + kubeLockMagic + ".*")
				locks := re.FindAllStringSubmatch(output, -1)
				for _, v := range locks {
					if len(v) > 0 {
						lockInfo := strings.Split(v[0], " ")
						if len(lockInfo) > 2 {
							args := []string{"lock", "remove", b.Image, lockInfo[1], lockInfo[0], "--pool", b.Pool, "--id", b.Id, "-m", mon}
							args = append(args, secret_opt...)
							cmd, err = b.exec.Run("rbd", args...)
							glog.Infof("remove orphaned locker %s from client %s: err %v, output: %s", lockInfo[1], lockInfo[0], err, string(cmd))
						}
					}
				}
			}

			// hold a lock: rbd lock add
			args := []string{"lock", "add", b.Image, lock_id, "--pool", b.Pool, "--id", b.Id, "-m", mon}
			args = append(args, secret_opt...)
			cmd, err = b.exec.Run("rbd", args...)
		} else {
			// defencing, find locker name
			ind := strings.LastIndex(output, lock_id) - 1
			for i := ind; i >= 0; i-- {
				if output[i] == '\n' {
					locker = output[(i + 1):ind]
					break
				}
			}
			// remove a lock: rbd lock remove
			args := []string{"lock", "remove", b.Image, lock_id, locker, "--pool", b.Pool, "--id", b.Id, "-m", mon}
			args = append(args, secret_opt...)
			cmd, err = b.exec.Run("rbd", args...)
		}

		if err == nil {
			//lock is acquired
			break
		}
	}
	return err
}

func (util *RBDUtil) persistRBD(rbd rbdMounter, mnt string) error {
	file := path.Join(mnt, "rbd.json")
	fp, err := os.Create(file)
	if err != nil {
		return fmt.Errorf("rbd: create err %s/%s", file, err)
	}
	defer fp.Close()

	encoder := json.NewEncoder(fp)
	if err = encoder.Encode(rbd); err != nil {
		return fmt.Errorf("rbd: encode err: %v.", err)
	}

	return nil
}

func (util *RBDUtil) loadRBD(mounter *rbdMounter, mnt string) error {
	file := path.Join(mnt, "rbd.json")
	fp, err := os.Open(file)
	if err != nil {
		return fmt.Errorf("rbd: open err %s/%s", file, err)
	}
	defer fp.Close()

	decoder := json.NewDecoder(fp)
	if err = decoder.Decode(mounter); err != nil {
		return fmt.Errorf("rbd: decode err: %v.", err)
	}

	return nil
}

func (util *RBDUtil) fencing(b rbdMounter) error {
	// no need to fence readOnly
	if (&b).GetAttributes().ReadOnly {
		return nil
	}
	return util.rbdLock(b, true)
}

func (util *RBDUtil) defencing(c rbdUnmounter) error {
	// no need to fence readOnly
	if c.ReadOnly {
		return nil
	}

	return util.rbdLock(*c.rbdMounter, false)
}

func (util *RBDUtil) AttachDisk(b rbdMounter) error {
	var err error
	var output []byte

	// create mount point
	globalPDPath := b.manager.MakeGlobalPDName(*b.rbd)
	notMnt, err := b.mounter.IsLikelyNotMountPoint(globalPDPath)
	// in the first time, the path shouldn't exist and IsLikelyNotMountPoint is expected to get NotExist
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("rbd: %s failed to check mountpoint", globalPDPath)
	}
	if !notMnt {
		return nil
	}
	if err = os.MkdirAll(globalPDPath, 0750); err != nil {
		return fmt.Errorf("rbd: failed to mkdir %s, error", globalPDPath)
	}

	devicePath, found := waitForPath(b.Pool, b.Image, 1)
	if !found {
		_, err = b.exec.Run("modprobe", "rbd")
		if err != nil {
			glog.Warningf("rbd: failed to load rbd kernel module:%v", err)
		}

		// fence off other mappers
		if err = util.fencing(b); err != nil {
			return rbdErrors(err, fmt.Errorf("rbd: failed to lock image %s (maybe locked by other nodes), error %v", b.Image, err))
		}
		// rbd lock remove needs ceph and image config
		// but kubelet doesn't get them from apiserver during teardown
		// so persit rbd config so upon disk detach, rbd lock can be removed
		// since rbd json is persisted in the same local directory that is used as rbd mountpoint later,
		// the json file remains invisible during rbd mount and thus won't be removed accidentally.
		util.persistRBD(b, globalPDPath)

		// rbd map
		l := len(b.Mon)
		// avoid mount storm, pick a host randomly
		start := rand.Int() % l
		// iterate all hosts until mount succeeds.
		for i := start; i < start+l; i++ {
			mon := b.Mon[i%l]
			glog.V(1).Infof("rbd: map mon %s", mon)
			if b.Secret != "" {
				output, err = b.exec.Run("rbd",
					"map", b.Image, "--pool", b.Pool, "--id", b.Id, "-m", mon, "--key="+b.Secret)
			} else {
				output, err = b.exec.Run("rbd",
					"map", b.Image, "--pool", b.Pool, "--id", b.Id, "-m", mon, "-k", b.Keyring)
			}
			if err == nil {
				break
			}
			glog.V(1).Infof("rbd: map error %v %s", err, string(output))
		}
		if err != nil {
			return fmt.Errorf("rbd: map failed %v %s", err, string(output))
		}
		devicePath, found = waitForPath(b.Pool, b.Image, 10)
		if !found {
			return errors.New("Could not map image: Timeout after 10s")
		}
		glog.V(3).Infof("rbd: successfully map image %s/%s to %s", b.Pool, b.Image, devicePath)
	}

	// mount it
	if err = b.mounter.FormatAndMount(devicePath, globalPDPath, b.fsType, nil); err != nil {
		err = fmt.Errorf("rbd: failed to mount rbd volume %s [%s] to %s, error %v", devicePath, b.fsType, globalPDPath, err)
	}
	glog.V(3).Infof("rbd: successfully mount image %s/%s at %s", b.Pool, b.Image, globalPDPath)
	return err
}

func (util *RBDUtil) DetachDisk(c rbdUnmounter, mntPath string) error {
	device, cnt, err := mount.GetDeviceNameFromMount(c.mounter, mntPath)
	if err != nil {
		return fmt.Errorf("rbd detach disk: failed to get device from mnt: %s\nError: %v", mntPath, err)
	}
	if err = c.mounter.Unmount(mntPath); err != nil {
		return fmt.Errorf("rbd detach disk: failed to umount: %s\nError: %v", mntPath, err)
	}
	glog.V(3).Infof("rbd: successfully umount mountpoint %s", mntPath)
	// if device is no longer used, see if can unmap
	if cnt <= 1 {
		// rbd unmap
		_, err = c.exec.Run("rbd", "unmap", device)
		if err != nil {
			return rbdErrors(err, fmt.Errorf("rbd: failed to unmap device %s:Error: %v", device, err))
		}

		// load ceph and image/pool info to remove fencing
		if err := util.loadRBD(c.rbdMounter, mntPath); err == nil {
			// remove rbd lock
			util.defencing(c)
		}

		glog.V(3).Infof("rbd: successfully unmap device %s", device)
	}
	return nil
}

func (util *RBDUtil) CreateImage(p *rbdVolumeProvisioner) (r *v1.RBDVolumeSource, size int, err error) {
	var output []byte
	capacity := p.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	volSizeBytes := capacity.Value()
	// convert to MB that rbd defaults on
	sz := int(volume.RoundUpSize(volSizeBytes, 1024*1024))
	volSz := fmt.Sprintf("%d", sz)
	// rbd create
	l := len(p.rbdMounter.Mon)
	// pick a mon randomly
	start := rand.Int() % l
	// iterate all monitors until create succeeds.
	for i := start; i < start+l; i++ {
		mon := p.Mon[i%l]
		if p.rbdMounter.imageFormat == rbdImageFormat2 {
			glog.V(4).Infof("rbd: create %s size %s format %s (features: %s) using mon %s, pool %s id %s key %s", p.rbdMounter.Image, volSz, p.rbdMounter.imageFormat, p.rbdMounter.imageFeatures, mon, p.rbdMounter.Pool, p.rbdMounter.adminId, p.rbdMounter.adminSecret)
		} else {
			glog.V(4).Infof("rbd: create %s size %s format %s using mon %s, pool %s id %s key %s", p.rbdMounter.Image, volSz, p.rbdMounter.imageFormat, mon, p.rbdMounter.Pool, p.rbdMounter.adminId, p.rbdMounter.adminSecret)
		}
		args := []string{"create", p.rbdMounter.Image, "--size", volSz, "--pool", p.rbdMounter.Pool, "--id", p.rbdMounter.adminId, "-m", mon, "--key=" + p.rbdMounter.adminSecret, "--image-format", p.rbdMounter.imageFormat}
		if p.rbdMounter.imageFormat == rbdImageFormat2 {
			// if no image features is provided, it results in empty string
			// which disable all RBD image format 2 features as we expected
			features := strings.Join(p.rbdMounter.imageFeatures, ",")
			args = append(args, "--image-feature", features)
		}
		output, err = p.exec.Run("rbd", args...)
		if err == nil {
			break
		} else {
			glog.Warningf("failed to create rbd image, output %v", string(output))
		}
	}

	if err != nil {
		return nil, 0, fmt.Errorf("failed to create rbd image: %v, command output: %s", err, string(output))
	}

	return &v1.RBDVolumeSource{
		CephMonitors: p.rbdMounter.Mon,
		RBDImage:     p.rbdMounter.Image,
		RBDPool:      p.rbdMounter.Pool,
	}, sz, nil
}

func (util *RBDUtil) DeleteImage(p *rbdVolumeDeleter) error {
	var output []byte
	found, err := util.rbdStatus(p.rbdMounter)
	if err != nil {
		return err
	}
	if found {
		glog.Info("rbd is still being used ", p.rbdMounter.Image)
		return fmt.Errorf("rbd %s is still being used", p.rbdMounter.Image)
	}
	// rbd rm
	l := len(p.rbdMounter.Mon)
	// pick a mon randomly
	start := rand.Int() % l
	// iterate all monitors until rm succeeds.
	for i := start; i < start+l; i++ {
		mon := p.rbdMounter.Mon[i%l]
		glog.V(4).Infof("rbd: rm %s using mon %s, pool %s id %s key %s", p.rbdMounter.Image, mon, p.rbdMounter.Pool, p.rbdMounter.adminId, p.rbdMounter.adminSecret)
		output, err = p.exec.Run("rbd",
			"rm", p.rbdMounter.Image, "--pool", p.rbdMounter.Pool, "--id", p.rbdMounter.adminId, "-m", mon, "--key="+p.rbdMounter.adminSecret)
		if err == nil {
			return nil
		} else {
			glog.Errorf("failed to delete rbd image: %v, command output: %s", err, string(output))
		}
	}
	return err
}

// run rbd status command to check if there is watcher on the image
func (util *RBDUtil) rbdStatus(b *rbdMounter) (bool, error) {
	var err error
	var output string
	var cmd []byte

	l := len(b.Mon)
	start := rand.Int() % l
	// iterate all hosts until mount succeeds.
	for i := start; i < start+l; i++ {
		mon := b.Mon[i%l]
		// cmd "rbd status" list the rbd client watch with the following output:
		// Watchers:
		//   watcher=10.16.153.105:0/710245699 client.14163 cookie=1
		glog.V(4).Infof("rbd: status %s using mon %s, pool %s id %s key %s", b.Image, mon, b.Pool, b.adminId, b.adminSecret)
		cmd, err = b.exec.Run("rbd",
			"status", b.Image, "--pool", b.Pool, "-m", mon, "--id", b.adminId, "--key="+b.adminSecret)
		output = string(cmd)

		if err != nil {
			if err.Error() == rbdCmdErr {
				glog.Errorf("rbd cmd not found")
			} else {
				// ignore error code, just checkout output for watcher string
				glog.Warningf("failed to execute rbd status on mon %s", mon)
			}
		}

		if strings.Contains(output, imageWatcherStr) {
			glog.V(4).Infof("rbd: watchers on %s: %s", b.Image, output)
			return true, nil
		} else {
			glog.Warningf("rbd: no watchers on %s", b.Image)
			return false, nil
		}
	}
	return false, nil
}
