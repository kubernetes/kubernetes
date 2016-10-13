/*
Copyright 2016 The Kubernetes Authors.

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

package drivers

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"regexp"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/exec"
)

type rbdDriver struct{}

func init() {
	RegisterDriver("rbd", &rbdDriver{})
}

// TODO: copied from ../../rbd/rbd_util.go -- needs refactor
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

func extractPoolAndImage(name string) (string, string, error) {
	parts := strings.Split(name, "/")
	if len(parts) != 2 {
		return "", "", fmt.Errorf("connection_info.data.name (%v) is not in pool/volume format", name)
	}

	return parts[0], parts[1], nil
}

func rbdMap(pool string, image string) error {
	exe := exec.New()
	output, err := exe.Command("rbd", "map", image, "--pool", pool).CombinedOutput()
	if err != nil {
		return fmt.Errorf("rbd: map failed %v %s", err, string(output))
	}
	return nil
}

func (_ *rbdDriver) AttachDisk(connInfo ConnectionInfo) (string, error) {
	name := connInfo.Data.Name

	pool, image, err := extractPoolAndImage(name)
	if err != nil {
		return "", err
	}

	devicePath, found := waitForPath(pool, image, 1)
	if !found {
		// modprobe
		exe := exec.New()
		output, err := exe.Command("modprobe", "rbd").CombinedOutput()
		if err != nil {
			glog.Warningf("rbd: failed to modprobe rbd error: %v", string(output))
		}

		if err = rbdMap(pool, image); err != nil {
			return "", err
		}

		devicePath, found = waitForPath(pool, image, 10)
		if !found {
			return "", errors.New("Could not map image: Timeout after 10s")
		}
	}

	return devicePath, nil
}

func (_ *rbdDriver) DetachDisk(connInfo ConnectionInfo) error {
	name := connInfo.Data.Name

	pool, image, err := extractPoolAndImage(name)
	if err != nil {
		return err
	}
	device, ok := getDevFromImageAndPool(pool, image)
	if !ok {
		return fmt.Errorf("failed to find rbd device for %v/%v", pool, image)
	}

	// rbd unmap
	exe := exec.New()
	output, err := exe.Command("rbd", "unmap", device).CombinedOutput()
	if err != nil {
		return fmt.Errorf("rbd: failed to unmap device %s:Error: %v", device, output)
	}

	return nil
}

func (_ *rbdDriver) IsAttached(connInfo ConnectionInfo) (string, bool, error) {
	name := connInfo.Data.Name

	pool, image, err := extractPoolAndImage(name)
	if err != nil {
		return "", false, err
	}
	devicePath, found := waitForPath(pool, image, 1)

	return devicePath, found, nil
}
