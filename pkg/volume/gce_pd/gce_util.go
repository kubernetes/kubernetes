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

package gce_pd

import (
	"errors"
	"fmt"
	"os"
	"path"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/gce"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/golang/glog"
)

type GCEDiskUtil struct{}

// Attaches a disk specified by a volume.GCEPersistentDisk to the current kubelet.
// Mounts the disk to it's global path.
func (util *GCEDiskUtil) AttachAndMountDisk(pd *gcePersistentDisk, globalPDPath string) error {
	gce, err := cloudprovider.GetCloudProvider("gce", nil)
	if err != nil {
		return err
	}
	if err := gce.(*gce_cloud.GCECloud).AttachDisk(pd.pdName, pd.readOnly); err != nil {
		return err
	}

	devicePaths := []string{
		path.Join("/dev/disk/by-id/", "google-"+pd.pdName),
		path.Join("/dev/disk/by-id/", "scsi-0Google_PersistentDisk_"+pd.pdName),
	}

	if pd.partition != "" {
		for i, path := range devicePaths {
			devicePaths[i] = path + "-part" + pd.partition
		}
	}
	//TODO(jonesdl) There should probably be better method than busy-waiting here.
	numTries := 0
	devicePath := ""
	// Wait for the disk device to be created
	for {
		for _, path := range devicePaths {
			_, err := os.Stat(path)
			if err == nil {
				devicePath = path
				break
			}
			if err != nil && !os.IsNotExist(err) {
				return err
			}
		}
		if devicePath != "" {
			break
		}
		numTries++
		if numTries == 10 {
			return errors.New("Could not attach disk: Timeout after 10s")
		}
		time.Sleep(time.Second)
	}

	// Only mount the PD globally once.
	mountpoint, err := pd.mounter.IsMountPoint(globalPDPath)
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
	options := []string{}
	if pd.readOnly {
		options = append(options, "ro")
	}
	if !mountpoint {
		err = pd.diskMounter.Mount(devicePath, globalPDPath, pd.fsType, options)
		if err != nil {
			os.Remove(globalPDPath)
			return err
		}
	}
	return nil
}

// Unmounts the device and detaches the disk from the kubelet's host machine.
func (util *GCEDiskUtil) DetachDisk(pd *gcePersistentDisk) error {
	// Unmount the global PD mount, which should be the only one.
	globalPDPath := makeGlobalPDName(pd.plugin.host, pd.pdName)
	if err := pd.mounter.Unmount(globalPDPath); err != nil {
		return err
	}
	if err := os.Remove(globalPDPath); err != nil {
		return err
	}
	// Detach the disk
	gce, err := cloudprovider.GetCloudProvider("gce", nil)
	if err != nil {
		return err
	}
	if err := gce.(*gce_cloud.GCECloud).DetachDisk(pd.pdName); err != nil {
		return err
	}
	return nil
}

// safe_format_and_mount is a utility script on GCE VMs that probes a persistent disk, and if
// necessary formats it before mounting it.
// This eliminates the necesisty to format a PD before it is used with a Pod on GCE.
// TODO: port this script into Go and use it for all Linux platforms
type gceSafeFormatAndMount struct {
	mount.Interface
	runner exec.Interface
}

// uses /usr/share/google/safe_format_and_mount to optionally mount, and format a disk
func (mounter *gceSafeFormatAndMount) Mount(source string, target string, fstype string, options []string) error {
	// Don't attempt to format if mounting as readonly. Go straight to mounting.
	for _, option := range options {
		if option == "ro" {
			return mounter.Interface.Mount(source, target, fstype, options)
		}
	}
	args := []string{}
	// ext4 is the default for safe_format_and_mount
	if len(fstype) > 0 && fstype != "ext4" {
		args = append(args, "-m", fmt.Sprintf("mkfs.%s", fstype))
	}
	args = append(args, options...)
	args = append(args, source, target)
	glog.V(5).Infof("exec-ing: /usr/share/google/safe_format_and_mount %v", args)
	cmd := mounter.runner.Command("/usr/share/google/safe_format_and_mount", args...)
	dataOut, err := cmd.CombinedOutput()
	if err != nil {
		glog.V(5).Infof("error running /usr/share/google/safe_format_and_mount\n%s", string(dataOut))
	}
	return err
}
