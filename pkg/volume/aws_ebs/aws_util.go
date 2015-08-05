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

package aws_ebs

import (
	"errors"
	"fmt"
	"os"
	"time"

	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"github.com/golang/glog"
)

type AWSDiskUtil struct{}

// Attaches a disk specified by a volume.AWSElasticBlockStore to the current kubelet.
// Mounts the disk to it's global path.
func (util *AWSDiskUtil) AttachAndMountDisk(b *awsElasticBlockStoreBuilder, globalPDPath string) error {
	volumes, err := b.getVolumeProvider()
	if err != nil {
		return err
	}
	devicePath, err := volumes.AttachDisk("", b.volumeID, b.readOnly)
	if err != nil {
		return err
	}
	if b.partition != "" {
		devicePath = devicePath + b.partition
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
			return errors.New("Could not attach disk: Timeout after 10s (" + devicePath + ")")
		}
		time.Sleep(time.Second)
	}

	// Only mount the PD globally once.
	mountpoint, err := b.mounter.IsMountPoint(globalPDPath)
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
	if b.readOnly {
		options = append(options, "ro")
	}
	if !mountpoint {
		err = b.diskMounter.Mount(devicePath, globalPDPath, b.fsType, options)
		if err != nil {
			os.Remove(globalPDPath)
			return err
		}
	}
	return nil
}

// Unmounts the device and detaches the disk from the kubelet's host machine.
func (util *AWSDiskUtil) DetachDisk(c *awsElasticBlockStoreCleaner) error {
	// Unmount the global PD mount, which should be the only one.
	globalPDPath := makeGlobalPDPath(c.plugin.host, c.volumeID)
	if err := c.mounter.Unmount(globalPDPath); err != nil {
		glog.V(2).Info("Error unmount dir ", globalPDPath, ": ", err)
		return err
	}
	if err := os.Remove(globalPDPath); err != nil {
		glog.V(2).Info("Error removing dir ", globalPDPath, ": ", err)
		return err
	}
	// Detach the disk
	volumes, err := c.getVolumeProvider()
	if err != nil {
		glog.V(2).Info("Error getting volume provider for volumeID ", c.volumeID, ": ", err)
		return err
	}
	if err := volumes.DetachDisk("", c.volumeID); err != nil {
		glog.V(2).Info("Error detaching disk ", c.volumeID, ": ", err)
		return err
	}
	return nil
}

// safe_format_and_mount is a utility script on AWS VMs that probes a persistent disk, and if
// necessary formats it before mounting it.
// This eliminates the necessity to format a PD before it is used with a Pod on AWS.
// TODO: port this script into Go and use it for all Linux platforms
type awsSafeFormatAndMount struct {
	mount.Interface
	runner exec.Interface
}

// uses /usr/share/google/safe_format_and_mount to optionally mount, and format a disk
func (mounter *awsSafeFormatAndMount) Mount(source string, target string, fstype string, options []string) error {
	// Don't attempt to format if mounting as readonly. Go straight to mounting.
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
