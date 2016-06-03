/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/volume"
)

type gcePersistentDiskAttacher struct {
	host volume.VolumeHost
}

var _ volume.Attacher = &gcePersistentDiskAttacher{}

var _ volume.AttachableVolumePlugin = &gcePersistentDiskPlugin{}

func (plugin *gcePersistentDiskPlugin) NewAttacher() (volume.Attacher, error) {
	return &gcePersistentDiskAttacher{host: plugin.host}, nil
}

func (plugin *gcePersistentDiskPlugin) GetDeviceName(spec *volume.Spec) (string, error) {
	volumeSource, _ := getVolumeSource(spec)
	if volumeSource == nil {
		return "", fmt.Errorf("Spec does not reference a GCE volume type")
	}

	return volumeSource.PDName, nil
}

// Attach checks with the GCE cloud provider if the specified volume is already
// attached to the specified node. If the volume is attached, it succeeds
// (returns nil). If it is not, Attach issues a call to the GCE cloud provider
// to attach it.
// Callers are responsible for retryinging on failure.
// Callers are responsible for thread safety between concurrent attach and
// detach operations.
func (attacher *gcePersistentDiskAttacher) Attach(spec *volume.Spec, hostName string) error {
	volumeSource, readOnly := getVolumeSource(spec)
	pdName := volumeSource.PDName

	gceCloud, err := getCloudProvider(attacher.host.GetCloudProvider())
	if err != nil {
		return err
	}

	attached, err := gceCloud.DiskIsAttached(pdName, hostName)
	if err != nil {
		// Log error and continue with attach
		glog.Errorf(
			"Error checking if PD (%q) is already attached to current node (%q). Will continue and try attach anyway. err=%v",
			pdName, hostName, err)
	}

	if attached {
		// Volume is already attached to node.
		glog.Infof("Attach operation is successful. PD %q is already attached to node %q.", pdName, hostName)
		return nil
	}

	if err = gceCloud.AttachDisk(pdName, hostName, readOnly); err != nil {
		glog.Errorf("Error attaching PD %q to node %q: %+v", pdName, hostName, err)
		return err
	}

	return nil
}

func (attacher *gcePersistentDiskAttacher) WaitForAttach(spec *volume.Spec, timeout time.Duration) (string, error) {
	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	volumeSource, _ := getVolumeSource(spec)
	pdName := volumeSource.PDName
	partition := ""
	if volumeSource.Partition != 0 {
		partition = strconv.Itoa(int(volumeSource.Partition))
	}

	sdBefore, err := filepath.Glob(diskSDPattern)
	if err != nil {
		glog.Errorf("Error filepath.Glob(\"%s\"): %v\r\n", diskSDPattern, err)
	}
	sdBeforeSet := sets.NewString(sdBefore...)

	devicePaths := getDiskByIdPaths(pdName, partition)
	for {
		select {
		case <-ticker.C:
			glog.V(5).Infof("Checking GCE PD %q is attached.", pdName)
			path, err := verifyDevicePath(devicePaths, sdBeforeSet)
			if err != nil {
				// Log error, if any, and continue checking periodically. See issue #11321
				glog.Errorf("Error verifying GCE PD (%q) is attached: %v", pdName, err)
			} else if path != "" {
				// A device path has successfully been created for the PD
				glog.Infof("Successfully found attached GCE PD %q.", pdName)
				return path, nil
			}
		case <-timer.C:
			return "", fmt.Errorf("Could not find attached GCE PD %q. Timeout waiting for mount paths to be created.", pdName)
		}
	}
}

func (attacher *gcePersistentDiskAttacher) GetDeviceMountPath(spec *volume.Spec) string {
	volumeSource, _ := getVolumeSource(spec)
	return makeGlobalPDName(attacher.host, volumeSource.PDName)
}

func (attacher *gcePersistentDiskAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string, mounter mount.Interface) error {
	// Only mount the PD globally once.
	notMnt, err := mounter.IsLikelyNotMountPoint(deviceMountPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(deviceMountPath, 0750); err != nil {
				return err
			}
			notMnt = true
		} else {
			return err
		}
	}

	volumeSource, readOnly := getVolumeSource(spec)

	options := []string{}
	if readOnly {
		options = append(options, "ro")
	}
	if notMnt {
		diskMounter := &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()}
		err = diskMounter.FormatAndMount(devicePath, deviceMountPath, volumeSource.FSType, options)
		if err != nil {
			os.Remove(deviceMountPath)
			return err
		}
	}
	return nil
}

type gcePersistentDiskDetacher struct {
	host volume.VolumeHost
}

var _ volume.Detacher = &gcePersistentDiskDetacher{}

func (plugin *gcePersistentDiskPlugin) NewDetacher() (volume.Detacher, error) {
	return &gcePersistentDiskDetacher{host: plugin.host}, nil
}

// Detach checks with the GCE cloud provider if the specified volume is already
// attached to the specified node. If the volume is not attached, it succeeds
// (returns nil). If it is attached, Detach issues a call to the GCE cloud
// provider to attach it.
// Callers are responsible for retryinging on failure.
// Callers are responsible for thread safety between concurrent attach and detach
// operations.
func (detacher *gcePersistentDiskDetacher) Detach(deviceMountPath string, hostName string) error {
	pdName := path.Base(deviceMountPath)

	gceCloud, err := getCloudProvider(detacher.host.GetCloudProvider())
	if err != nil {
		return err
	}

	attached, err := gceCloud.DiskIsAttached(pdName, hostName)
	if err != nil {
		// Log error and continue with detach
		glog.Errorf(
			"Error checking if PD (%q) is already attached to current node (%q). Will continue and try detach anyway. err=%v",
			pdName, hostName, err)
	}

	if !attached {
		// Volume is not attached to node. Success!
		glog.Infof("Detach operation is successful. PD %q was not attached to node %q.", pdName, hostName)
		return nil
	}

	if err = gceCloud.DetachDisk(pdName, hostName); err != nil {
		glog.Errorf("Error detaching PD %q from node %q: %v", pdName, hostName, err)
		return err
	}

	return nil
}

func (detacher *gcePersistentDiskDetacher) WaitForDetach(devicePath string, timeout time.Duration) error {
	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			glog.V(5).Infof("Checking device %q is detached.", devicePath)
			if pathExists, err := pathExists(devicePath); err != nil {
				return fmt.Errorf("Error checking if device path exists: %v", err)
			} else if !pathExists {
				return nil
			}
		case <-timer.C:
			return fmt.Errorf("Timeout reached; PD Device %v is still attached", devicePath)
		}
	}
}

func (detacher *gcePersistentDiskDetacher) UnmountDevice(deviceMountPath string, mounter mount.Interface) error {
	return unmountPDAndRemoveGlobalPath(deviceMountPath, mounter)
}
