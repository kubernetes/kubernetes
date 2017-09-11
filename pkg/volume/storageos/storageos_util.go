/*
Copyright 2017 The Kubernetes Authors.

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

package storageos

import (
	"errors"
	"fmt"
	"os"
	"path"
	"strings"

	"k8s.io/kubernetes/pkg/util/mount"

	"github.com/golang/glog"
	storageosapi "github.com/storageos/go-api"
	storageostypes "github.com/storageos/go-api/types"
)

const (
	losetupPath = "losetup"

	modeBlock deviceType = iota
	modeFile
	modeUnsupported

	ErrDeviceNotFound     = "device not found"
	ErrDeviceNotSupported = "device not supported"
	ErrNotAvailable       = "not available"
)

type deviceType int

// storageosVolume describes a provisioned volume
type storageosVolume struct {
	ID          string
	Name        string
	Namespace   string
	Description string
	Pool        string
	SizeGB      int
	Labels      map[string]string
	FSType      string
}

type storageosAPIConfig struct {
	apiAddr    string
	apiUser    string
	apiPass    string
	apiVersion string
}

type apiImplementer interface {
	Volume(namespace string, ref string) (*storageostypes.Volume, error)
	VolumeCreate(opts storageostypes.VolumeCreateOptions) (*storageostypes.Volume, error)
	VolumeMount(opts storageostypes.VolumeMountOptions) error
	VolumeUnmount(opts storageostypes.VolumeUnmountOptions) error
	VolumeDelete(opt storageostypes.DeleteOptions) error
}

// storageosUtil is the utility structure to interact with the StorageOS API.
type storageosUtil struct {
	api apiImplementer
}

func (u *storageosUtil) NewAPI(apiCfg *storageosAPIConfig) error {
	if u.api != nil {
		return nil
	}
	if apiCfg == nil {
		apiCfg = &storageosAPIConfig{
			apiAddr:    defaultAPIAddress,
			apiUser:    defaultAPIUser,
			apiPass:    defaultAPIPassword,
			apiVersion: defaultAPIVersion,
		}
		glog.V(4).Infof("Using default StorageOS API settings: addr %s, version: %s", apiCfg.apiAddr, defaultAPIVersion)
	}

	api, err := storageosapi.NewVersionedClient(apiCfg.apiAddr, defaultAPIVersion)
	if err != nil {
		return err
	}
	api.SetAuth(apiCfg.apiUser, apiCfg.apiPass)
	u.api = api
	return nil
}

// Creates a new StorageOS volume and makes it available as a device within
// /var/lib/storageos/volumes.
func (u *storageosUtil) CreateVolume(p *storageosProvisioner) (*storageosVolume, error) {
	if err := u.NewAPI(p.apiCfg); err != nil {
		return nil, err
	}

	if p.labels == nil {
		p.labels = make(map[string]string)
	}
	opts := storageostypes.VolumeCreateOptions{
		Name:        p.volName,
		Size:        p.sizeGB,
		Description: p.description,
		Pool:        p.pool,
		FSType:      p.fsType,
		Namespace:   p.volNamespace,
		Labels:      p.labels,
	}

	vol, err := u.api.VolumeCreate(opts)
	if err != nil {
		glog.Errorf("volume create failed for volume %q (%v)", opts.Name, err)
		return nil, err
	}
	return &storageosVolume{
		ID:          vol.ID,
		Name:        vol.Name,
		Namespace:   vol.Namespace,
		Description: vol.Description,
		Pool:        vol.Pool,
		FSType:      vol.FSType,
		SizeGB:      int(vol.Size),
		Labels:      vol.Labels,
	}, nil
}

// Attach exposes a volume on the host as a block device.  StorageOS uses a
// global namespace, so if the volume exists, it should already be available as
// a device within `/var/lib/storageos/volumes/<id>`.
//
// Depending on the host capabilities, the device may be either a block device
// or a file device.  Block devices can be used directly, but file devices must
// be made accessible as a block device before using.
func (u *storageosUtil) AttachVolume(b *storageosMounter) (string, error) {
	if err := u.NewAPI(b.apiCfg); err != nil {
		return "", err
	}

	vol, err := u.api.Volume(b.volNamespace, b.volName)
	if err != nil {
		glog.Warningf("volume retrieve failed for volume %q with namespace %q (%v)", b.volName, b.volNamespace, err)
		return "", err
	}

	// Clear any existing mount reference from the API.  These may be leftover
	// from previous mounts where the unmount operation couldn't get access to
	// the API credentials.
	if vol.Mounted {
		opts := storageostypes.VolumeUnmountOptions{
			Name:      vol.Name,
			Namespace: vol.Namespace,
		}
		if err := u.api.VolumeUnmount(opts); err != nil {
			glog.Warningf("Couldn't clear existing StorageOS mount reference: %v", err)
		}
	}

	srcPath := path.Join(b.devicePath, vol.ID)
	dt, err := pathDeviceType(srcPath)
	if err != nil {
		glog.Warningf("volume source path %q for volume %q not ready (%v)", srcPath, b.volName, err)
		return "", err
	}
	switch dt {
	case modeBlock:
		return srcPath, nil
	case modeFile:
		return attachFileDevice(srcPath, b.exec)
	default:
		return "", fmt.Errorf(ErrDeviceNotSupported)
	}
}

// Detach detaches a volume from the host.  This is only needed when NBD is not
// enabled and loop devices are used to simulate a block device.
func (u *storageosUtil) DetachVolume(b *storageosUnmounter, devicePath string) error {
	if !isLoopDevice(devicePath) {
		return nil
	}
	if _, err := os.Stat(devicePath); os.IsNotExist(err) {
		return nil
	}
	return removeLoopDevice(devicePath, b.exec)
}

// Mount mounts the volume on the host.
func (u *storageosUtil) MountVolume(b *storageosMounter, mntDevice, deviceMountPath string) error {
	notMnt, err := b.mounter.IsLikelyNotMountPoint(deviceMountPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err = os.MkdirAll(deviceMountPath, 0750); err != nil {
				return err
			}
			notMnt = true
		} else {
			return err
		}
	}
	if err = os.MkdirAll(deviceMountPath, 0750); err != nil {
		glog.Errorf("mkdir failed on disk %s (%v)", deviceMountPath, err)
		return err
	}
	options := []string{}
	if b.readOnly {
		options = append(options, "ro")
	}
	if notMnt {
		err = b.diskMounter.FormatAndMount(mntDevice, deviceMountPath, b.fsType, options)
		if err != nil {
			os.Remove(deviceMountPath)
			return err
		}
	}
	if err != nil {
		return err
	}

	if err := u.NewAPI(b.apiCfg); err != nil {
		return err
	}

	opts := storageostypes.VolumeMountOptions{
		Name:       b.volName,
		Namespace:  b.volNamespace,
		FsType:     b.fsType,
		Mountpoint: deviceMountPath,
		Client:     b.plugin.host.GetHostName(),
	}
	return u.api.VolumeMount(opts)
}

// Unmount removes the mount reference from the volume allowing it to be
// re-mounted elsewhere.
func (u *storageosUtil) UnmountVolume(b *storageosUnmounter) error {
	if err := u.NewAPI(b.apiCfg); err != nil {
		// We can't always get the config we need, so allow the unmount to
		// succeed even if we can't remove the mount reference from the API.
		glog.V(4).Infof("Could not remove mount reference in the StorageOS API as no credentials available to the unmount operation")
		return nil
	}

	opts := storageostypes.VolumeUnmountOptions{
		Name:      b.volName,
		Namespace: b.volNamespace,
		Client:    b.plugin.host.GetHostName(),
	}
	return u.api.VolumeUnmount(opts)
}

// Deletes a StorageOS volume.  Assumes it has already been unmounted and detached.
func (u *storageosUtil) DeleteVolume(d *storageosDeleter) error {
	if err := u.NewAPI(d.apiCfg); err != nil {
		return err
	}

	// Deletes must be forced as the StorageOS API will not normally delete
	// volumes that it thinks are mounted.  We can't be sure the unmount was
	// registered via the API so we trust k8s to only delete volumes it knows
	// are unmounted.
	opts := storageostypes.DeleteOptions{
		Name:      d.volName,
		Namespace: d.volNamespace,
		Force:     true,
	}
	return u.api.VolumeDelete(opts)
}

// pathMode returns the FileMode for a path.
func pathDeviceType(path string) (deviceType, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return modeUnsupported, err
	}
	switch mode := fi.Mode(); {
	case mode&os.ModeDevice != 0:
		return modeBlock, nil
	case mode.IsRegular():
		return modeFile, nil
	default:
		return modeUnsupported, nil
	}
}

// attachFileDevice takes a path to a regular file and makes it available as an
// attached block device.
func attachFileDevice(path string, exec mount.Exec) (string, error) {
	blockDevicePath, err := getLoopDevice(path, exec)
	if err != nil && err.Error() != ErrDeviceNotFound {
		return "", err
	}

	// If no existing loop device for the path, create one
	if blockDevicePath == "" {
		glog.V(4).Infof("Creating device for path: %s", path)
		blockDevicePath, err = makeLoopDevice(path, exec)
		if err != nil {
			return "", err
		}
	}
	return blockDevicePath, nil
}

// Returns the full path to the loop device associated with the given path.
func getLoopDevice(path string, exec mount.Exec) (string, error) {
	_, err := os.Stat(path)
	if os.IsNotExist(err) {
		return "", errors.New(ErrNotAvailable)
	}
	if err != nil {
		return "", fmt.Errorf("not attachable: %v", err)
	}

	args := []string{"-j", path}
	out, err := exec.Run(losetupPath, args...)
	if err != nil {
		glog.V(2).Infof("Failed device discover command for path %s: %v", path, err)
		return "", err
	}
	return parseLosetupOutputForDevice(out)
}

func makeLoopDevice(path string, exec mount.Exec) (string, error) {
	args := []string{"-f", "--show", path}
	out, err := exec.Run(losetupPath, args...)
	if err != nil {
		glog.V(2).Infof("Failed device create command for path %s: %v", path, err)
		return "", err
	}
	return parseLosetupOutputForDevice(out)
}

func removeLoopDevice(device string, exec mount.Exec) error {
	args := []string{"-d", device}
	out, err := exec.Run(losetupPath, args...)
	if err != nil {
		if !strings.Contains(string(out), "No such device or address") {
			return err
		}
	}
	return nil
}

func isLoopDevice(device string) bool {
	return strings.HasPrefix(device, "/dev/loop")
}

func parseLosetupOutputForDevice(output []byte) (string, error) {
	if len(output) == 0 {
		return "", errors.New(ErrDeviceNotFound)
	}

	// losetup returns device in the format:
	// /dev/loop1: [0073]:148662 (/var/lib/storageos/volumes/308f14af-cf0a-08ff-c9c3-b48104318e05)
	device := strings.TrimSpace(strings.SplitN(string(output), ":", 2)[0])
	if len(device) == 0 {
		return "", errors.New(ErrDeviceNotFound)
	}
	return device, nil
}
