/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package cinder

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

type CinderDiskUtil struct{}

// Attaches a disk specified by a volume.CinderPersistenDisk to the current kubelet.
// Mounts the disk to it's global path.
func (util *CinderDiskUtil) AttachDisk(b *cinderVolumeBuilder, globalPDPath string) error {
	options := []string{}
	if b.readOnly {
		options = append(options, "ro")
	}
	cloud, err := b.plugin.getCloudProvider()
	if err != nil {
		return err
	}
	diskid, err := cloud.AttachDisk(b.pdName)
	if err != nil {
		return err
	}

	var devicePath string
	numTries := 0
	for {
		devicePath = makeDevicePath(diskid)
		// probe the attached vol so that symlink in /dev/disk/by-id is created
		probeAttachedVolume()

		_, err := os.Stat(devicePath)
		if err == nil {
			break
		}
		if err != nil && !os.IsNotExist(err) {
			return err
		}
		numTries++
		if numTries == 10 {
			return errors.New("Could not attach disk: Timeout after 60s")
		}
		time.Sleep(time.Second * 6)
	}

	notmnt, err := b.mounter.IsLikelyNotMountPoint(globalPDPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(globalPDPath, 0750); err != nil {
				return err
			}
			notmnt = true
		} else {
			return err
		}
	}
	if notmnt {
		err = b.blockDeviceMounter.Mount(devicePath, globalPDPath, b.fsType, options)
		if err != nil {
			os.Remove(globalPDPath)
			return err
		}
		glog.V(2).Infof("Safe mount successful: %q\n", devicePath)
	}
	return nil
}

func makeDevicePath(diskid string) string {
	files, _ := ioutil.ReadDir("/dev/disk/by-id/")
	for _, f := range files {
		if strings.Contains(f.Name(), "virtio-") {
			devid_prefix := f.Name()[len("virtio-"):len(f.Name())]
			if strings.Contains(diskid, devid_prefix) {
				glog.V(4).Infof("Found disk attached as %q; full devicepath: %s\n", f.Name(), path.Join("/dev/disk/by-id/", f.Name()))
				return path.Join("/dev/disk/by-id/", f.Name())
			}
		}
	}
	glog.Warningf("Failed to find device for the diskid: %q\n", diskid)
	return ""
}

// Unmounts the device and detaches the disk from the kubelet's host machine.
func (util *CinderDiskUtil) DetachDisk(cd *cinderVolumeCleaner) error {
	globalPDPath := makeGlobalPDName(cd.plugin.host, cd.pdName)
	if err := cd.mounter.Unmount(globalPDPath); err != nil {
		return err
	}
	if err := os.Remove(globalPDPath); err != nil {
		return err
	}
	glog.V(2).Infof("Successfully unmounted main device: %s\n", globalPDPath)

	cloud, err := cd.plugin.getCloudProvider()
	if err != nil {
		return err
	}

	if err = cloud.DetachDisk(cd.pdName); err != nil {
		return err
	}
	glog.V(2).Infof("Successfully detached cinder volume %s", cd.pdName)
	return nil
}

func (util *CinderDiskUtil) DeleteVolume(cd *cinderVolumeDeleter) error {
	cloud, err := cd.plugin.getCloudProvider()
	if err != nil {
		return err
	}

	if err = cloud.DeleteVolume(cd.pdName); err != nil {
		glog.V(2).Infof("Error deleting cinder volume %s: %v", cd.pdName, err)
		return err
	}
	glog.V(2).Infof("Successfully deleted cinder volume %s", cd.pdName)
	return nil
}

func (util *CinderDiskUtil) CreateVolume(c *cinderVolumeProvisioner) (volumeID string, volumeSizeGB int, err error) {
	cloud, err := c.plugin.getCloudProvider()
	if err != nil {
		return "", 0, err
	}

	volSizeBytes := c.options.Capacity.Value()
	// Cinder works with gigabytes, convert to GiB with rounding up
	volSizeGB := int(volume.RoundUpSize(volSizeBytes, 1024*1024*1024))
	name, err := cloud.CreateVolume(volSizeGB)
	if err != nil {
		glog.V(2).Infof("Error creating cinder volume: %v", err)
		return "", 0, err
	}
	glog.V(2).Infof("Successfully created cinder volume %s", name)
	return name, volSizeGB, nil
}

type cinderSafeFormatAndMount struct {
	mount.Interface
	runner exec.Interface
}

/*
The functions below depend on the following executables; This will have to be ported to more generic implementations
/bin/lsblk
/sbin/mkfs.ext3 or /sbin/mkfs.ext4
/usr/bin/udevadm
*/
func (diskmounter *cinderSafeFormatAndMount) Mount(device string, target string, fstype string, options []string) error {
	fmtRequired, err := isFormatRequired(device, fstype, diskmounter)
	if err != nil {
		glog.Warningf("Failed to determine if formating is required: %v\n", err)
		//return err
	}
	if fmtRequired {
		glog.V(2).Infof("Formatting of the vol required")
		if _, err := formatVolume(device, fstype, diskmounter); err != nil {
			glog.Warningf("Failed to format volume: %v\n", err)
			return err
		}
	}
	return diskmounter.Interface.Mount(device, target, fstype, options)
}

func isFormatRequired(devicePath string, fstype string, exec *cinderSafeFormatAndMount) (bool, error) {
	args := []string{"-f", devicePath}
	glog.V(4).Infof("exec-ing: /bin/lsblk %v\n", args)
	cmd := exec.runner.Command("/bin/lsblk", args...)
	dataOut, err := cmd.CombinedOutput()
	if err != nil {
		glog.Warningf("error running /bin/lsblk\n%s", string(dataOut))
		return false, err
	}
	if len(string(dataOut)) > 0 {
		if strings.Contains(string(dataOut), fstype) {
			return false, nil
		} else {
			return true, nil
		}
	} else {
		glog.Warningf("Failed to get any response from /bin/lsblk")
		return false, errors.New("Failed to get reponse from /bin/lsblk")
	}
	glog.Warningf("Unknown error occured executing /bin/lsblk")
	return false, errors.New("Unknown error occured executing /bin/lsblk")
}

func formatVolume(devicePath string, fstype string, exec *cinderSafeFormatAndMount) (bool, error) {
	if "ext4" != fstype && "ext3" != fstype {
		glog.Warningf("Unsupported format type: %q\n", fstype)
		return false, errors.New(fmt.Sprint("Unsupported format type: %q\n", fstype))
	}
	args := []string{devicePath}
	cmd := exec.runner.Command(fmt.Sprintf("/sbin/mkfs.%s", fstype), args...)
	dataOut, err := cmd.CombinedOutput()
	if err != nil {
		glog.Warningf("error running /sbin/mkfs for fstype: %q \n%s", fstype, string(dataOut))
		return false, err
	}
	glog.V(2).Infof("Successfully formated device: %q with fstype %q; output:\n %q\n,", devicePath, fstype, string(dataOut))
	return true, err
}

func probeAttachedVolume() error {
	executor := exec.New()
	args := []string{"trigger"}
	cmd := executor.Command("/usr/bin/udevadm", args...)
	_, err := cmd.CombinedOutput()
	if err != nil {
		glog.Errorf("error running udevadm trigger %v\n", err)
		return err
	}
	glog.V(4).Infof("Successfully probed all attachments")
	return nil
}
