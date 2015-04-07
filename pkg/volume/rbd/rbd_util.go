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
	"errors"
	"fmt"
	"math/rand"
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

// make a directory like /var/lib/kubelet/plugins/kubernetes.io/pod/rbd/pool-image-image
func makePDNameInternal(host volume.VolumeHost, pool string, image string) string {
	return path.Join(host.GetPluginDir(RBDPluginName), "rbd", pool+"-image-"+image)
}

type RBDUtil struct{}

func (util *RBDUtil) MakeGlobalPDName(rbd rbd) string {
	return makePDNameInternal(rbd.plugin.host, rbd.pool, rbd.image)
}

func (util *RBDUtil) AttachDisk(rbd rbd) error {
	var err error
	devicePath := strings.Join([]string{"/dev/rbd", rbd.pool, rbd.image}, "/")
	exist := waitForPathToExist(devicePath, 1)
	if !exist {
		// modprobe
		_, err = rbd.plugin.execCommand("modprobe", []string{"rbd"})
		if err != nil {
			return fmt.Errorf("rbd: failed to modprobe rbd error:%v", err)
		}
		// rbd map
		l := len(rbd.mon)
		// avoid mount storm, pick a host randomly
		start := rand.Int() % l
		// iterate all hosts until mount succeeds.
		for i := start; i < start+l; i++ {
			mon := rbd.mon[i%l]
			glog.V(1).Infof("rbd: map mon %s", mon)
			if rbd.secret != "" {
				_, err = rbd.plugin.execCommand("rbd",
					[]string{"map", rbd.image, "--pool", rbd.pool, "--id", rbd.id, "-m", mon, "--key=" + rbd.secret})
			} else {
				_, err = rbd.plugin.execCommand("rbd",
					[]string{"map", rbd.image, "--pool", rbd.pool, "--id", rbd.id, "-m", mon, "-k", rbd.keyring})
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
	globalPDPath := rbd.manager.MakeGlobalPDName(rbd)
	mountpoint, err := rbd.mounter.IsMountPoint(globalPDPath)
	if err != nil {
		return fmt.Errorf("rbd: %s failed to check mountpoint", globalPDPath)
	}
	if mountpoint {
		return nil
	}

	if err := os.MkdirAll(globalPDPath, 0750); err != nil {
		return fmt.Errorf("rbd: failed to mkdir %s, error", globalPDPath)
	}

	if err = rbd.mounter.Mount(devicePath, globalPDPath, rbd.fsType, nil); err != nil {
		err = fmt.Errorf("rbd: failed to mount rbd volume %s [%s] to %s, error %v", devicePath, rbd.fsType, globalPDPath, err)
	}

	return err
}

func (util *RBDUtil) DetachDisk(rbd rbd, mntPath string) error {
	device, cnt, err := mount.GetDeviceNameFromMount(rbd.mounter, mntPath)
	if err != nil {
		return fmt.Errorf("rbd detach disk: failed to get device from mnt: %s\nError: %v", mntPath, err)
	}
	if err = rbd.mounter.Unmount(mntPath); err != nil {
		return fmt.Errorf("rbd detach disk: failed to umount: %s\nError: %v", mntPath, err)
	}
	// if device is no longer used, see if can unmap
	if cnt <= 1 {
		// rbd unmap
		_, err = rbd.plugin.execCommand("rbd", []string{"unmap", device})
		if err != nil {
			return fmt.Errorf("rbd: failed to unmap device %s:Error: %v", device, err)
		}
		glog.Infof("rbd: successfully unmap device %s", device)
	}
	return nil
}
