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

// TODO(thockin): This whole pkg is pretty linux-centric.  As soon as we have
// an alternate platform, we will need to abstract further.
package mount

import (
	"fmt"
	"path"
	"path/filepath"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/exec"
)

const (
	// Default mount command if mounter path is not specified
	defaultMountCommand  = "mount"
	MountsInGlobalPDPath = "mounts"
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
	// It should return ErrNotExist when the directory does not exist.
	IsLikelyNotMountPoint(file string) (bool, error)
	// DeviceOpened determines if the device is in use elsewhere
	// on the system, i.e. still mounted.
	DeviceOpened(pathname string) (bool, error)
	// PathIsDevice determines if a path is a device.
	PathIsDevice(pathname string) (bool, error)
	// GetDeviceNameFromMount finds the device name by checking the mount path
	// to get the global mount path which matches its plugin directory
	GetDeviceNameFromMount(mountPath, pluginDir string) (string, error)
}

// Compile-time check to ensure all Mounter implementations satisfy
// the mount interface
var _ Interface = &Mounter{}

// This represents a single line in /proc/mounts or /etc/fstab.
type MountPoint struct {
	Device string
	Path   string
	Type   string
	Opts   []string
	Freq   int
	Pass   int
}

// SafeFormatAndMount probes a device to see if it is formatted.
// Namely it checks to see if a file system is present. If so it
// mounts it otherwise the device is formatted first then mounted.
type SafeFormatAndMount struct {
	Interface
	Runner exec.Interface
}

// FormatAndMount formats the given disk, if needed, and mounts it.
// That is if the disk is not formatted and it is not being mounted as
// read-only it will format it first then mount it. Otherwise, if the
// disk is already formatted or it is being mounted as read-only, it
// will be mounted without formatting.
func (mounter *SafeFormatAndMount) FormatAndMount(source string, target string, fstype string, options []string) error {
	// Don't attempt to format if mounting as readonly. Go straight to mounting.
	for _, option := range options {
		if option == "ro" {
			return mounter.Interface.Mount(source, target, fstype, options)
		}
	}
	return mounter.formatAndMount(source, target, fstype, options)
}

// New returns a mount.Interface for the current system.
// It provides options to override the default mounter behavior.
// mounterPath allows using an alternative to `/bin/mount` for mounting.
func New(mounterPath string) Interface {
	// If mounter-path flag is not set, use default mount path
	if mounterPath == "" {
		mounterPath = defaultMountCommand
	}

	return &Mounter{
		mounterPath: mounterPath,
	}
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
	// If mountPath is symlink, need get its target path.
	slTarget, err := filepath.EvalSymlinks(mountPath)
	if err != nil {
		slTarget = mountPath
	}
	for i := range mps {
		if mps[i].Path == slTarget {
			deviceName = mps[i].Device
			break
		}
	}

	// TODO: this is a workaround for the unmount device issue caused by gci mounter.
	// In GCI cluster, if gci mounter is used for mounting, the container started by mounter
	// script will cause additional mounts created in the container. Since these mounts are
	// irrelavant to the original mounts, they should be not considered when checking the
	// mount references. Current solution is to filter out those mount paths that contain
	// the string of original mount path.
	// Plan to work on better approach to solve this issue.

	// Find all references to the device.
	var refs []string
	if deviceName == "" {
		glog.Warningf("could not determine device for path: %q", mountPath)
	} else {
		for i := range mps {
			if mps[i].Device == deviceName && !strings.Contains(mps[i].Path, slTarget) {
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
	// If mountPath is symlink, need get its target path.
	slTarget, err := filepath.EvalSymlinks(mountPath)
	if err != nil {
		slTarget = mountPath
	}
	for i := range mps {
		if mps[i].Path == slTarget {
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

// getDeviceNameFromMount find the device name from /proc/mounts in which
// the mount path reference should match the given plugin directory. In case no mount path reference
// matches, returns the volume name taken from its given mountPath
func getDeviceNameFromMount(mounter Interface, mountPath, pluginDir string) (string, error) {
	refs, err := GetMountRefs(mounter, mountPath)
	if err != nil {
		glog.V(4).Infof("GetMountRefs failed for mount path %q: %v", mountPath, err)
		return "", err
	}
	if len(refs) == 0 {
		glog.V(4).Infof("Directory %s is not mounted", mountPath)
		return "", fmt.Errorf("directory %s is not mounted", mountPath)
	}
	basemountPath := path.Join(pluginDir, MountsInGlobalPDPath)
	for _, ref := range refs {
		if strings.HasPrefix(ref, basemountPath) {
			volumeID, err := filepath.Rel(basemountPath, ref)
			if err != nil {
				glog.Errorf("Failed to get volume id from mount %s - %v", mountPath, err)
				return "", err
			}
			return volumeID, nil
		}
	}

	return path.Base(mountPath), nil
}
