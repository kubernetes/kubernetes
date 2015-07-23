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
	"fmt"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/golang/glog"
	"strings"
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
	// IsMountPoint determines if a directory is a mountpoint.
	IsMountPoint(file string) (bool, error)
	SetRunner(executor ContainerExecutor)
}
type ContainerExecutor interface {
	RunInContainerBySelector(selector labels.Selector, containerName string, cmd []string) ([]byte, error)
}

type MounterType int

const (
	MounterMount MounterType = iota
	MounterNsenter
	MounterContainer
)

// This represents a single line in /proc/mounts or /etc/fstab.
type MountPoint struct {
	Device string
	Path   string
	Type   string
	Opts   []string
	Freq   int
	Pass   int
}

type MountContainerConfig struct {
	Selector      labels.Selector
	ContainerName string
}

// MounterConfig contains configuration of Volumes mounter
type MountConfig struct {
	// What mounter to use
	Mounter MounterType

	// Configuration of mount.ContainerMounter:
	// Map filesystem type -> MounterContainerConfig
	MountContainers map[string]*MountContainerConfig
}

func ParseMountConfig(volumeMounter string, mountContainers []string, containerized bool) (*MountConfig, error) {
	glog.V(5).Infof("ParseMountConfig: parsing mounter: '%s', containers: '%s', containerized: '%v'", volumeMounter, mountContainers, containerized)

	cfg := &MountConfig{}

	switch volumeMounter {
	case "":
		// nothing on command line, use the default one
		if containerized {
			glog.V(5).Infof("ParseMountConfig: using default Mounter")
			cfg.Mounter = MounterNsenter
		} else {
			glog.V(5).Infof("ParseMountConfig: using default NsenterMounter")
			cfg.Mounter = MounterMount
		}
	case "mount":
		glog.V(5).Infof("ParseMountConfig: using Mounter")
		cfg.Mounter = MounterMount
	case "nsenter":
		glog.V(5).Infof("ParseMountConfig: using NsenterMounter")
		cfg.Mounter = MounterNsenter
	case "container":
		glog.V(5).Infof("ParseMountConfig: using ContainerMounter")
		cfg.Mounter = MounterContainer
	default:
		return nil, fmt.Errorf("Unknown volume-mounter value: '%v'", volumeMounter)
	}
	if cfg.Mounter != MounterContainer {
		return cfg, nil
	}

	// for MounterContainer, we need to parse mountContainers
	cfg.MountContainers = make(map[string]*MountContainerConfig)

	for _, container := range mountContainers {
		parts := strings.Split(container, ":")
		if len(parts) != 3 {
			return nil, fmt.Errorf("Cannot parse --mount-container specification '%v': expected 3 parts separated by ':'", container)
		}

		selector, err := labels.Parse(parts[1])
		if err != nil {
			return nil, fmt.Errorf("Error parsing selector '%s': %v", parts[1], err)
		}

		cfg.MountContainers[parts[0]] = &MountContainerConfig{
			Selector:      selector,
			ContainerName: parts[2],
		}
	}
	return cfg, nil

}

// New returns a mount.Interface for the current system.
func New(cfg *MountConfig) Interface {
	if cfg == nil {
		glog.V(3).Infof("mount.new: no cfg specified, using Mounter")
		return &Mounter{}
	}

	switch cfg.Mounter {
	case MounterMount:
		glog.V(3).Infof("mount.new: using Mounter")
		return &Mounter{}
	case MounterNsenter:
		glog.V(3).Infof("mount.new: using NsenterMounter")
		return &NsenterMounter{}
	case MounterContainer:
		glog.V(3).Infof("mount.new: using ContainerMounter")
		return &ContainerMounter{
			config: cfg,
		}
	}
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
