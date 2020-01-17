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
	"path/filepath"
	"strings"

	storageosapi "github.com/storageos/go-api"
	storageostypes "github.com/storageos/go-api/types"
	"k8s.io/klog"
	utilexec "k8s.io/utils/exec"
)

const (
	losetupPath = "losetup"

	modeBlock deviceType = iota
	modeFile
	modeUnsupported

	//ErrDeviceNotFound defines "device not found"
	ErrDeviceNotFound = "device not found"
	//ErrDeviceNotSupported defines "device not supported"
	ErrDeviceNotSupported = "device not supported"
	//ErrNotAvailable defines "not available"
	ErrNotAvailable = "not available"
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
	Node(ref string) (*storageostypes.Node, error)
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
		klog.V(4).Infof("using default StorageOS API settings: addr %s, version: %s", apiCfg.apiAddr, defaultAPIVersion)
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

	klog.V(4).Infof("creating StorageOS volume %q with namespace %q", p.volName, p.volNamespace)

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
		klog.Errorf("volume create failed for volume %q (%v)", opts.Name, err)
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

	klog.V(4).Infof("attaching StorageOS volume %q with namespace %q", b.volName, b.volNamespace)

	if err := u.NewAPI(b.apiCfg); err != nil {
		return "", err
	}

	// Get the node's device path from the API, falling back to the default if
	// not set on the node.
	if b.deviceDir == "" {
		b.deviceDir = u.DeviceDir(b)
	}

	vol, err := u.api.Volume(b.volNamespace, b.volName)
	if err != nil {
		klog.Warningf("volume retrieve failed for volume %q with namespace %q (%v)", b.volName, b.volNamespace, err)
		return "", err
	}

	srcPath := filepath.Join(b.deviceDir, vol.ID)
	dt, err := pathDeviceType(srcPath)
	if err != nil {
		klog.Warningf("volume source path %q for volume %q not ready (%v)", srcPath, b.volName, err)
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

	klog.V(4).Infof("detaching StorageOS volume %q with namespace %q", b.volName, b.volNamespace)

	if !isLoopDevice(devicePath) {
		return nil
	}
	if _, err := os.Stat(devicePath); os.IsNotExist(err) {
		return nil
	}
	return removeLoopDevice(devicePath, b.exec)
}

// AttachDevice attaches the volume device to the host at a given mount path.
func (u *storageosUtil) AttachDevice(b *storageosMounter, deviceMountPath string) error {

	klog.V(4).Infof("attaching StorageOS device for volume %q with namespace %q", b.volName, b.volNamespace)

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
	if err := u.api.VolumeMount(opts); err != nil {
		return err
	}
	return nil
}

// Mount mounts the volume on the host.
func (u *storageosUtil) MountVolume(b *storageosMounter, mntDevice, deviceMountPath string) error {

	klog.V(4).Infof("mounting StorageOS volume %q with namespace %q", b.volName, b.volNamespace)

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
		klog.Errorf("mkdir failed on disk %s (%v)", deviceMountPath, err)
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
	return err
}

// Unmount removes the mount reference from the volume allowing it to be
// re-mounted elsewhere.
func (u *storageosUtil) UnmountVolume(b *storageosUnmounter) error {

	klog.V(4).Infof("clearing StorageOS mount reference for volume %q with namespace %q", b.volName, b.volNamespace)

	if err := u.NewAPI(b.apiCfg); err != nil {
		// We can't always get the config we need, so allow the unmount to
		// succeed even if we can't remove the mount reference from the API.
		klog.Warningf("could not remove mount reference in the StorageOS API as no credentials available to the unmount operation")
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

// Get the node's device path from the API, falling back to the default if not
// specified.
func (u *storageosUtil) DeviceDir(b *storageosMounter) string {

	ctrl, err := u.api.Node(b.plugin.host.GetHostName())
	if err != nil {
		klog.Warningf("node device path lookup failed: %v", err)
		return defaultDeviceDir
	}
	if ctrl == nil || ctrl.DeviceDir == "" {
		klog.Warningf("node device path not set, using default: %s", defaultDeviceDir)
		return defaultDeviceDir
	}
	return ctrl.DeviceDir
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
func attachFileDevice(path string, exec utilexec.Interface) (string, error) {
	blockDevicePath, err := getLoopDevice(path, exec)
	if err != nil && err.Error() != ErrDeviceNotFound {
		return "", err
	}

	// If no existing loop device for the path, create one
	if blockDevicePath == "" {
		klog.V(4).Infof("Creating device for path: %s", path)
		blockDevicePath, err = makeLoopDevice(path, exec)
		if err != nil {
			return "", err
		}
	}
	return blockDevicePath, nil
}

// Returns the full path to the loop device associated with the given path.
func getLoopDevice(path string, exec utilexec.Interface) (string, error) {
	_, err := os.Stat(path)
	if os.IsNotExist(err) {
		return "", errors.New(ErrNotAvailable)
	}
	if err != nil {
		return "", fmt.Errorf("not attachable: %v", err)
	}

	args := []string{"-j", path}
	out, err := exec.Command(losetupPath, args...).CombinedOutput()
	if err != nil {
		klog.V(2).Infof("Failed device discover command for path %s: %v", path, err)
		return "", err
	}
	return parseLosetupOutputForDevice(out)
}

func makeLoopDevice(path string, exec utilexec.Interface) (string, error) {
	args := []string{"-f", "-P", "--show", path}
	out, err := exec.Command(losetupPath, args...).CombinedOutput()
	if err != nil {
		klog.V(2).Infof("Failed device create command for path %s: %v", path, err)
		return "", err
	}
	return parseLosetupOutputForDevice(out)
}

func removeLoopDevice(device string, exec utilexec.Interface) error {
	args := []string{"-d", device}
	out, err := exec.Command(losetupPath, args...).CombinedOutput()
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
