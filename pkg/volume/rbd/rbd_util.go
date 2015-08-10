/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"math/rand"
	"os"
	"path"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/volume"
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

// make a directory like /var/lib/kubelet/plugins/kubernetes.io/pod/rbd/pool-image-image
func makePDNameInternal(host volume.VolumeHost, pool string, image string) string {
	return path.Join(host.GetPluginDir(rbdPluginName), "rbd", pool+"-image-"+image)
}

type RBDUtil struct{}

func (util *RBDUtil) MakeGlobalPDName(rbd rbd) string {
	return makePDNameInternal(rbd.plugin.host, rbd.Pool, rbd.Image)
}

func (util *RBDUtil) rbdLock(b rbdBuilder, lock bool) error {
	var err error
	var output, locker string
	var cmd []byte
	var secret_opt []string

	if b.Secret != "" {
		secret_opt = []string{"--key=" + b.Secret}
	} else {
		secret_opt = []string{"-k", b.Keyring}
	}
	// construct lock id using host name and a magic prefix
	lock_id := "kubelet_lock_magic_" + node.GetHostname("")

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
		cmd, err = b.plugin.execCommand("rbd",
			append([]string{"lock", "list", b.Image, "--pool", b.Pool, "--id", b.Id, "-m", mon}, secret_opt...))
		output = string(cmd)

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
			// hold a lock: rbd lock add
			cmd, err = b.plugin.execCommand("rbd",
				append([]string{"lock", "add", b.Image, lock_id, "--pool", b.Pool, "--id", b.Id, "-m", mon}, secret_opt...))
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
			cmd, err = b.plugin.execCommand("rbd",
				append([]string{"lock", "remove", b.Image, lock_id, locker, "--pool", b.Pool, "--id", b.Id, "-m", mon}, secret_opt...))
		}

		if err == nil {
			//lock is acquired
			break
		}
	}
	return err
}

func (util *RBDUtil) persistRBD(rbd rbdBuilder, mnt string) error {
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

func (util *RBDUtil) loadRBD(builder *rbdBuilder, mnt string) error {
	file := path.Join(mnt, "rbd.json")
	fp, err := os.Open(file)
	if err != nil {
		return fmt.Errorf("rbd: open err %s/%s", file, err)
	}
	defer fp.Close()

	decoder := json.NewDecoder(fp)
	if err = decoder.Decode(builder); err != nil {
		return fmt.Errorf("rbd: decode err: %v.", err)
	}

	return nil
}

func (util *RBDUtil) fencing(b rbdBuilder) error {
	// no need to fence readOnly
	if b.IsReadOnly() {
		return nil
	}
	return util.rbdLock(b, true)
}

func (util *RBDUtil) defencing(c rbdCleaner) error {
	// no need to fence readOnly
	if c.ReadOnly {
		return nil
	}

	return util.rbdLock(*c.rbdBuilder, false)
}

func (util *RBDUtil) AttachDisk(b rbdBuilder) error {
	var err error
	devicePath := strings.Join([]string{"/dev/rbd", b.Pool, b.Image}, "/")
	exist := waitForPathToExist(devicePath, 1)
	if !exist {
		// modprobe
		_, err = b.plugin.execCommand("modprobe", []string{"rbd"})
		if err != nil {
			return fmt.Errorf("rbd: failed to modprobe rbd error:%v", err)
		}
		// rbd map
		l := len(b.Mon)
		// avoid mount storm, pick a host randomly
		start := rand.Int() % l
		// iterate all hosts until mount succeeds.
		for i := start; i < start+l; i++ {
			mon := b.Mon[i%l]
			glog.V(1).Infof("rbd: map mon %s", mon)
			if b.Secret != "" {
				_, err = b.plugin.execCommand("rbd",
					[]string{"map", b.Image, "--pool", b.Pool, "--id", b.Id, "-m", mon, "--key=" + b.Secret})
			} else {
				_, err = b.plugin.execCommand("rbd",
					[]string{"map", b.Image, "--pool", b.Pool, "--id", b.Id, "-m", mon, "-k", b.Keyring})
			}
			if err == nil {
				break
			}
		}
	}
	if err != nil {
		return err
	}
	exist = waitForPathToExist(devicePath, 10)
	if !exist {
		return errors.New("Could not map image: Timeout after 10s")
	}
	// mount it
	globalPDPath := b.manager.MakeGlobalPDName(*b.rbd)
	mountpoint, err := b.mounter.IsMountPoint(globalPDPath)
	// in the first time, the path shouldn't exist and IsMountPoint is expected to get NotExist
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("rbd: %s failed to check mountpoint", globalPDPath)
	}
	if mountpoint {
		return nil
	}

	if err := os.MkdirAll(globalPDPath, 0750); err != nil {
		return fmt.Errorf("rbd: failed to mkdir %s, error", globalPDPath)
	}

	// fence off other mappers
	if err := util.fencing(b); err != nil {
		return fmt.Errorf("rbd: image %s is locked by other nodes", b.Image)
	}
	// rbd lock remove needs ceph and image config
	// but kubelet doesn't get them from apiserver during teardown
	// so persit rbd config so upon disk detach, rbd lock can be removed
	// since rbd json is persisted in the same local directory that is used as rbd mountpoint later,
	// the json file remains invisible during rbd mount and thus won't be removed accidentally.
	util.persistRBD(b, globalPDPath)

	if err = b.mounter.Mount(devicePath, globalPDPath, b.fsType, nil); err != nil {
		err = fmt.Errorf("rbd: failed to mount rbd volume %s [%s] to %s, error %v", devicePath, b.fsType, globalPDPath, err)
	}
	return err
}

func (util *RBDUtil) DetachDisk(c rbdCleaner, mntPath string) error {
	device, cnt, err := mount.GetDeviceNameFromMount(c.mounter, mntPath)
	if err != nil {
		return fmt.Errorf("rbd detach disk: failed to get device from mnt: %s\nError: %v", mntPath, err)
	}
	if err = c.mounter.Unmount(mntPath); err != nil {
		return fmt.Errorf("rbd detach disk: failed to umount: %s\nError: %v", mntPath, err)
	}
	// if device is no longer used, see if can unmap
	if cnt <= 1 {
		// rbd unmap
		_, err = c.plugin.execCommand("rbd", []string{"unmap", device})
		if err != nil {
			return fmt.Errorf("rbd: failed to unmap device %s:Error: %v", device, err)
		}

		// load ceph and image/pool info to remove fencing
		if err := util.loadRBD(c.rbdBuilder, mntPath); err == nil {
			// remove rbd lock
			util.defencing(c)
		}

		glog.Infof("rbd: successfully unmap device %s", device)
	}
	return nil
}
