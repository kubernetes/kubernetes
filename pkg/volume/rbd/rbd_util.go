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
	"math/rand"
	"os"
	"path"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	imageWatcherStr = "watcher="
)

// make a directory like /var/lib/kubelet/plugins/kubernetes.io/pod/rbd/pool-image-image
func makePDNameInternal(host volume.VolumeHost, pool string, image string) string {
	return path.Join(host.GetPluginDir(rbdPluginName), "rbd", pool+"-image-"+image)
}

type RBDUtil struct{}

func (util *RBDUtil) MakeGlobalPDName(rbd rbd) string {
	return makePDNameInternal(rbd.plugin.host, rbd.Pool, rbd.Image)
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

type diskMapper interface {
	MapDisk(disk rbdMounter) (string, error)
	UnmapDisk(disk rbdMounter, mntPath string) error
}

func createDiskMapper(backendType string) (diskMapper, error) {
	glog.V(1).Infof("rbd: creating diskMapper for backendType %s", backendType)

	switch strings.ToLower(backendType) {
	case "krbd":
		return &RBDKernel{}, nil
	case "nbd":
		return &RBDNbd{}, nil
	}
	return nil, fmt.Errorf("unsupported backendType %s", backendType)
}

func (util *RBDUtil) AttachDisk(b rbdMounter) error {
	// check the mount point first
	globalPDPath := b.manager.MakeGlobalPDName(*b.rbd)
	notMnt, err := b.mounter.IsLikelyNotMountPoint(globalPDPath)
	// in the first time, the path shouldn't exist and IsLikelyNotMountPoint is expected to get NotExist
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("rbd: %s failed to check mountpoint", globalPDPath)
	}
	if !notMnt {
		// it looks the specified directory is already a mount point,
		// so there is no sense in mapping a new rbd device
		return nil
	}
	if err = os.MkdirAll(globalPDPath, 0750); err != nil {
		return fmt.Errorf("rbd: failed to mkdir %s, error", globalPDPath)
	}

	// map a block device
	mapper, err := createDiskMapper(b.BackendType)
	if err != nil {
		return fmt.Errorf("rbd: cannot create diskMapper: %v", err)
	}
	devicePath, err := mapper.MapDisk(b)
	if err != nil {
		return fmt.Errorf("rbd: cannot map block device")
	}

	// rbd lock remove needs ceph and image config while the whole detach requires BackendType
	// but kubelet doesn't get them from apiserver during teardown
	// so persit rbd config to let disk detaching as well as rbd unlocking
	// since rbd json is persisted in the same local directory that is used as rbd mountpoint later,
	// the json file remains invisible during rbd mount and thus won't be removed accidentally.
	util.persistRBD(b, globalPDPath)

	// mount it
	if err := b.mounter.FormatAndMount(devicePath, globalPDPath, b.fsType, nil); err != nil {
		if err := mapper.UnmapDisk(b, devicePath); err != nil {
			return fmt.Errorf("rbd detach disk: failed to unmap: %s\nError: %v", devicePath, err)
		}
		return fmt.Errorf("rbd: failed to mount rbd volume %s [%s] to %s, error %v", devicePath, b.fsType, globalPDPath, err)
	}
	return nil
}

func (util *RBDUtil) DetachDisk(c rbdUnmounter, mntPath string) error {
	device, cnt, err := mount.GetDeviceNameFromMount(c.mounter, mntPath)
	if err != nil {
		return fmt.Errorf("rbd detach disk: failed to get device from mnt: %s\nError: %v", mntPath, err)
	}
	if err = c.mounter.Unmount(mntPath); err != nil {
		return fmt.Errorf("rbd detach disk: failed to umount: %s\nError: %v", mntPath, err)
	}
	// if device is no longer used, see if can unmap
	if cnt <= 1 {
		// load ceph and image/pool info to remove fencing
		if err := util.loadRBD(c.rbdMounter, mntPath); err != nil {
			return fmt.Errorf("rbd detach disk: failed to load the persisted metadata\nError: %v", err)
		}

		mapper, err := createDiskMapper(c.rbdMounter.BackendType)
		if err != nil {
			return fmt.Errorf("rbd: cannot create diskMapper: %v", err)
		}
		if err := mapper.UnmapDisk(*c.rbdMounter, device); err != nil {
			return fmt.Errorf("rbd detach disk: failed to unmap: %s\nError: %v", device, err)
		}

		glog.Infof("rbd: successfully unmap device %s", device)
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
		glog.V(4).Infof("rbd: create %s size %s using mon %s, pool %s id %s key %s", p.rbdMounter.Image, volSz, mon, p.rbdMounter.Pool, p.rbdMounter.adminId, p.rbdMounter.adminSecret)

		// TODO: krbd in 4.9 does support images in the v2 format with the exclusive-lock
		// feature enabled. Its proliferation would hopefully allow us to remove this "if".
		var imgFormatOpts []string
		if p.rbdMounter.BackendType == "krbd" {
			imgFormatOpts = []string{"--image-format", "1"}
		}
		output, err = p.rbdMounter.plugin.execCommand("rbd",
			append([]string{"create", p.rbdMounter.Image, "--size", volSz, "--pool", p.rbdMounter.Pool, "--id", p.rbdMounter.adminId, "-m", mon, "--key=" + p.rbdMounter.adminSecret}, imgFormatOpts...))
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
		output, err = p.plugin.execCommand("rbd",
			[]string{"rm", p.rbdMounter.Image, "--pool", p.rbdMounter.Pool, "--id", p.rbdMounter.adminId, "-m", mon, "--key=" + p.rbdMounter.adminSecret})
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
