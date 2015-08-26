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

// TODO(thockin): This whole pkg is pretty linux-centric.  As soon as we have
// an alternate platform, we will need to abstract further.
package mount

import (
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/exec"
)

type Interface interface {
	// Mount mounts source to target as fstype with given options.
	Mount(source string, target string, fstype string, options []string) error
	// Unmount unmounts given target.
	Unmount(target string) error
	// List returns a list of all mounted filesystems.  This can be large.
	// On some platforms, reading mounts is not guaranteed consistent (i.e.
	// it could change between chunked reads). This is guaranteed to be
	// consistent.
	List() ([]MountPoint, error)
	// IsLikelyNotMountPoint determines if a directory is a mountpoint.
	IsLikelyNotMountPoint(file string) (bool, error)
}

// This represents a single line in /proc/mounts or /etc/fstab.
type MountPoint struct {
	Device string
	Path   string
	Type   string
	Opts   []string
	Freq   int
	Pass   int
}

// SafeFormatAndMount probes a device to see if it is formatted. If
// so it mounts it otherwise it formats it and mounts it
type SafeFormatAndMount struct {
	Interface
	Runner exec.Interface
}

// Mount mounts the given disk. If the disk is not formatted and the disk is not being mounted as read only
// it will format the disk first then mount it.
func (mounter *SafeFormatAndMount) Mount(source string, target string, fstype string, options []string) error {
	// Don't attempt to format if mounting as readonly. Go straight to mounting.
	for _, option := range options {
		if option == "ro" {
			return mounter.Interface.Mount(source, target, fstype, options)
		}
	}
	return mounter.formatAndMount(source, target, fstype, options)
}

// formatAndMount uses unix utils to format and mount the given disk
func (mounter *SafeFormatAndMount) formatAndMount(source string, target string, fstype string, options []string) error {
	options = append(options, "defaults")

	// Try to mount the disk
	err := mounter.Interface.Mount(source, target, fstype, options)
	if err != nil {
		// It is possible that this disk is not formatted. Double check using 'file'
		notFormatted, err := mounter.diskLooksUnformatted(source)
		if err == nil && notFormatted {
			// Disk is unformatted so format it.
			// Use 'ext4' as the default
			if len(fstype) == 0 {
				fstype = "ext4"
			}
			args := []string{"-E", "lazy_itable_init=0,lazy_journal_init=0", "-F", source}
			cmd := mounter.Runner.Command("mkfs."+fstype, args...)
			_, err := cmd.CombinedOutput()
			if err == nil {
				// the disk has been formatted sucessfully try to mount it again.
				return mounter.Interface.Mount(source, target, fstype, options)
			}
			return err
		}
	}
	return err
}

// diskLooksUnformatted uses 'file' to see if the given disk is unformated
func (mounter *SafeFormatAndMount) diskLooksUnformatted(disk string) (bool, error) {
	args := []string{"-L", "--special-files", disk}
	cmd := mounter.Runner.Command("file", args...)
	dataOut, err := cmd.CombinedOutput()

	// TODO (#13212): check if this disk has partitions and return false, and
	// an error if so.

	if err != nil {
		return false, err
	}

	return !strings.Contains(string(dataOut), "filesystem"), nil
}

// New returns a mount.Interface for the current system.
func New() Interface {
	return &Mounter{}
}

// GetMountRefs finds all other references to the device referenced
// by mountPath; returns a list of paths.
func GetMountRefs(mounter Interface, mountPath string) ([]string, error) {
	mps, err := mounter.List()
	if err != nil {
		return nil, err
	}

	// Find the device name.
	deviceName := ""
	for i := range mps {
		if mps[i].Path == mountPath {
			deviceName = mps[i].Device
			break
		}
	}

	// Find all references to the device.
	var refs []string
	if deviceName == "" {
		glog.Warningf("could not determine device for path: %q", mountPath)
	} else {
		for i := range mps {
			if mps[i].Device == deviceName && mps[i].Path != mountPath {
				refs = append(refs, mps[i].Path)
			}
		}
	}
	return refs, nil
}

// GetDeviceNameFromMount: given a mnt point, find the device from /proc/mounts
// returns the device name, reference count, and error code
func GetDeviceNameFromMount(mounter Interface, mountPath string) (string, int, error) {
	mps, err := mounter.List()
	if err != nil {
		return "", 0, err
	}

	// Find the device name.
	// FIXME if multiple devices mounted on the same mount path, only the first one is returned
	device := ""
	for i := range mps {
		if mps[i].Path == mountPath {
			device = mps[i].Device
			break
		}
	}

	// Find all references to the device.
	refCount := 0
	for i := range mps {
		if mps[i].Device == device {
			refCount++
		}
	}
	return device, refCount, nil
}
