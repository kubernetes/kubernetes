/*
Copyright 2014 Google Inc. All rights reserved.

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

package gce_pd

import (
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/gce"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
)

const partitionRegex = "[a-z][a-z]*(?P<partition>[0-9][0-9]*)?"

var regexMatcher = regexp.MustCompile(partitionRegex)

type GCEDiskUtil struct{}

// Attaches a disk specified by a volume.GCEPersistentDisk to the current kubelet.
// Mounts the disk to it's global path.
func (util *GCEDiskUtil) AttachDisk(pd *gcePersistentDisk) error {
	gce, err := cloudprovider.GetCloudProvider("gce", nil)
	if err != nil {
		return err
	}
	flags := uintptr(0)
	if pd.readOnly {
		flags = mount.FlagReadOnly
	}
	if err := gce.(*gce_cloud.GCECloud).AttachDisk(pd.pdName, pd.readOnly); err != nil {
		return err
	}
	devicePath := path.Join("/dev/disk/by-id/", "google-"+pd.pdName)
	if pd.partition != "" {
		devicePath = devicePath + "-part" + pd.partition
	}
	//TODO(jonesdl) There should probably be better method than busy-waiting here.
	numTries := 0
	for {
		_, err := os.Stat(devicePath)
		if err == nil {
			break
		}
		if err != nil && !os.IsNotExist(err) {
			return err
		}
		numTries++
		if numTries == 10 {
			return errors.New("Could not attach disk: Timeout after 10s")
		}
		time.Sleep(time.Second)
	}
	globalPDPath := makeGlobalPDName(pd.plugin.host, pd.pdName, pd.readOnly)
	// Only mount the PD globally once.
	mountpoint, err := isMountPoint(globalPDPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(globalPDPath, 0750); err != nil {
				return err
			}
			mountpoint = false
		} else {
			return err
		}
	}
	if !mountpoint {
		err = pd.mounter.Mount(devicePath, globalPDPath, pd.fsType, flags, "")
		if err != nil {
			os.Remove(globalPDPath)
			return err
		}
	}
	return nil
}

func getDeviceName(devicePath, canonicalDevicePath string) (string, error) {
	isMatch := regexMatcher.MatchString(path.Base(canonicalDevicePath))
	if !isMatch {
		return "", fmt.Errorf("unexpected device: %s", canonicalDevicePath)
	}
	if isMatch {
		result := make(map[string]string)
		substrings := regexMatcher.FindStringSubmatch(path.Base(canonicalDevicePath))
		for i, label := range regexMatcher.SubexpNames() {
			result[label] = substrings[i]
		}
		partition := result["partition"]
		devicePath = strings.TrimSuffix(devicePath, "-part"+partition)
	}
	return strings.TrimPrefix(path.Base(devicePath), "google-"), nil
}

// Unmounts the device and detaches the disk from the kubelet's host machine.
// Expects a GCE device path symlink. Ex: /dev/disk/by-id/google-mydisk-part1
func (util *GCEDiskUtil) DetachDisk(pd *gcePersistentDisk, devicePath string) error {
	// Follow the symlink to the actual device path.
	canonicalDevicePath, err := filepath.EvalSymlinks(devicePath)
	if err != nil {
		return err
	}
	deviceName, err := getDeviceName(devicePath, canonicalDevicePath)
	if err != nil {
		return err
	}
	globalPDPath := makeGlobalPDName(pd.plugin.host, deviceName, pd.readOnly)
	if err := pd.mounter.Unmount(globalPDPath, 0); err != nil {
		return err
	}
	if err := os.Remove(globalPDPath); err != nil {
		return err
	}
	gce, err := cloudprovider.GetCloudProvider("gce", nil)
	if err != nil {
		return err
	}
	if err := gce.(*gce_cloud.GCECloud).DetachDisk(deviceName); err != nil {
		return err
	}
	return nil
}
