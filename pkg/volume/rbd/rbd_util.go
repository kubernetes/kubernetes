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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	imageWatcherStr = "watcher="
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
		time.Sleep(time.Second)
	}
	return "", false
}

// make a directory like /var/lib/kubelet/plugins/kubernetes.io/pod/rbd/pool-image-image
func makePDNameInternal(host volume.VolumeHost, pool string, image string) string {
	return path.Join(host.GetPluginDir(rbdPluginName), pool+"-image-"+image)
}

type RBDUtil struct{}

func (util *RBDUtil) AttachDisk(b rbdMounter) (string, error) {
	var err error
	var output []byte

	devicePath, found := waitForPath(b.Pool, b.Image, 1)
	if !found {
		// modprobe
		_, err = b.plugin.execCommand("modprobe", []string{"rbd"})
		if err != nil {
			return "", fmt.Errorf("rbd: failed to modprobe rbd error:%v", err)
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

	return devicePath, err
}

func (util *RBDUtil) DetachDisk(plugin *rbdPlugin, device string) error {
	// rbd unmap
	_, err := plugin.execCommand("rbd", []string{"unmap", device})
	if err != nil {
		return fmt.Errorf("rbd: failed to unmap device %s:Error: %v", device, err)
	}

	glog.Infof("rbd: successfully unmap device %s", device)
	return nil
}

func (util *RBDUtil) CreateImage(p *rbdVolumeProvisioner) (r *api.RBDVolumeSource, size int, err error) {
	capacity := p.options.PVC.Spec.Resources.Requests[api.ResourceName(api.ResourceStorage)]
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
		glog.V(4).Infof("rbd: create %s size %s using mon %s, pool %s id %s key %s", p.rbdMounter.Image, volSz, mon, p.rbdMounter.Pool, p.rbdMounter.adminId, p.rbdMounter.adminSecret)
		var output []byte
		output, err = p.rbdMounter.plugin.execCommand("rbd",
			[]string{"create", p.rbdMounter.Image, "--size", volSz, "--pool", p.rbdMounter.Pool, "--id", p.rbdMounter.adminId, "-m", mon, "--key=" + p.rbdMounter.adminSecret, "--image-format", "1"})
		if err == nil {
			break
		} else {
			glog.Warningf("failed to create rbd image, output %v", string(output))
		}
	}

	if err != nil {
		glog.Errorf("rbd: Error creating rbd image: %v", err)
		return nil, 0, err
	}

	return &api.RBDVolumeSource{
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
		output, err = p.plugin.execCommand("rbd",
			[]string{"rm", p.rbdMounter.Image, "--pool", p.rbdMounter.Pool, "--id", p.rbdMounter.adminId, "-m", mon, "--key=" + p.rbdMounter.adminSecret})
		if err == nil {
			return nil
		} else {
			glog.Errorf("failed to delete rbd image, error %v output %v", err, string(output))
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
		cmd, err = b.plugin.execCommand("rbd",
			[]string{"status", b.Image, "--pool", b.Pool, "-m", mon, "--id", b.adminId, "--key=" + b.adminSecret})
		output = string(cmd)

		if err != nil {
			// ignore error code, just checkout output for watcher string
			glog.Warningf("failed to execute rbd status on mon %s", mon)
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
