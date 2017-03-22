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
// utility functions to setup rbd volume using the kernel RBD (krbd) client
// mainly implement diskMapper interface
//

package rbd

import (
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
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/node"
)

const (
	kubeLockMagic = "kubelet_lock_magic_"
)

func (rk *RBDKernel) IsSupported(b rbdMounter) bool {
	// let's check whether we have the kernel plugin available
	if _, err := b.plugin.execCommand("modprobe", []string{"rbd"}); err != nil {
		return false
	}
	if _, err := os.Stat("/sys/bus/rbd/devices"); os.IsNotExist(err) {
		return false
	}
	return true
}

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
			po := path.Join(sys_path, name, "pool")
			img := path.Join(sys_path, name, "name")
			exe := exec.New()
			out, err := exe.Command("cat", po, img).CombinedOutput()
			if err != nil {
				continue
			}
			matched, err := regexp.MatchString("^"+pool+"\n"+image+"\n$", string(out))
			if err != nil || !matched {
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

type RBDKernel struct {
	RBDUtil
}

func (rk *RBDKernel) fencing(b rbdMounter) error {
	// no need to fence readOnly
	if (&b).GetAttributes().ReadOnly {
		return nil
	}
	return rk.rbdLock(b, true)
}

func (rk *RBDKernel) defencing(b rbdMounter) error {
	// no need to fence readOnly
	if b.ReadOnly {
		return nil
	}

	return rk.rbdLock(b, false)
}

func (rk *RBDKernel) MapDisk(b rbdMounter) (string, error) {
	var err error
	var output []byte

	devicePath, found := waitForPath(b.Pool, b.Image, 1)
	if !found {
		// modprobe
		_, err = b.plugin.execCommand("modprobe", []string{"rbd"})
		if err != nil {
			return "", fmt.Errorf("rbd: failed to modprobe rbd error:%v", err)
		}

		// fence off other mappers
		if err := rk.fencing(b); err != nil {
			return "", fmt.Errorf("rbd: image %s is locked by other nodes", b.Image)
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
				output, err = b.plugin.execCommand("rbd",
					[]string{"map", b.Image, "--pool", b.Pool, "--id", b.Id, "-m", mon, "--key=" + b.Secret})
			} else {
				output, err = b.plugin.execCommand("rbd",
					[]string{"map", b.Image, "--pool", b.Pool, "--id", b.Id, "-m", mon, "-k", b.Keyring})
			}
			if err == nil {
				break
			}
			glog.V(1).Infof("rbd: map error %v %s", err, string(output))
		}
		if err != nil {
			return "", fmt.Errorf("rbd: map failed %v %s", err, string(output))
		}
		devicePath, found = waitForPath(b.Pool, b.Image, 10)
		if !found {
			return "", errors.New("Could not map image: Timeout after 10s")
		}
	}

	return devicePath, nil
}

func (rk *RBDKernel) UnmapDisk(b rbdMounter, device string) error {
	// rbd unmap
	_, err := b.plugin.execCommand("rbd", []string{"unmap", device})
	if err != nil {
		return fmt.Errorf("rbd: failed to unmap device %s:Error: %v", device, err)
	}

	rk.defencing(b)

	glog.Infof("rbd: successfully unmap device %s", device)
	return nil
}

func (rk *RBDKernel) rbdLock(b rbdMounter, lock bool) error {
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
			used, statusErr := rk.rbdStatus(&b)
			if statusErr == nil && !used {
				re := regexp.MustCompile("client.* " + kubeLockMagic + ".*")
				locks := re.FindAllStringSubmatch(output, -1)
				for _, v := range locks {
					if len(v) > 0 {
						lockInfo := strings.Split(v[0], " ")
						if len(lockInfo) > 2 {
							cmd, err = b.plugin.execCommand("rbd",
								append([]string{"lock", "remove", b.Image, lockInfo[1], lockInfo[0], "--pool", b.Pool, "--id", b.Id, "-m", mon}, secret_opt...))
							glog.Infof("remove orphaned locker %s from client %s: err %v, output: %s", lockInfo[1], lockInfo[0], err, string(cmd))
						}
					}
				}
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
