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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/volume"

	"github.com/golang/glog"
	storageosapi "github.com/storageos/go-api"
	storageostypes "github.com/storageos/go-api/types"
)

const (
	losetupPath   = "losetup"
	apiSecretName = "storageos-api"

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
	Size        int
	Labels      map[string]string
	FSType      string
}

type storageosAPIConfig struct {
	apiAddr    string
	apiUser    string
	apiPass    string
	apiVersion string
}

type apiConfigGetter interface {
	GetAPIConfig() *storageosAPIConfig
}

type namespacedSecret struct {
	namespace  string
	secretName string
	host       volume.VolumeHost
}

// GetAPIConfig checks the namespace for the api config, then default namespace.
// Returns default config if no secrets found.
func (a *namespacedSecret) GetAPIConfig() *storageosAPIConfig {
	c := &storageosAPIConfig{
		apiAddr:    defaultAPIAddress,
		apiUser:    defaultAPIUser,
		apiPass:    defaultAPIPassword,
		apiVersion: defaultAPIVersion,
	}

	keys, err := a.host.GetKubeClient().Core().Secrets(a.namespace).Get(apiSecretName, metav1.GetOptions{})
	if err != nil && a.namespace == "default" {
		return c
	}
	if err != nil {
		keys, err = a.host.GetKubeClient().Core().Secrets("default").Get(apiSecretName, metav1.GetOptions{})
		if err != nil {
			return c
		}
	}
	for name, data := range keys.Data {
		if name == "apiAddress" {
			c.apiAddr = string(data)
		}
		if name == "apiUsername" {
			c.apiUser = string(data)
		}
		if name == "apiPassword" {
			c.apiPass = string(data)
		}
		if name == "apiVersion" {
			c.apiVersion = string(data)
		}
	}
	return c
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

// client returns a StorageOS API Client.
func (u *storageosUtil) client(cfg apiConfigGetter) apiImplementer {
	if u.api == nil {
		apiCfg := cfg.GetAPIConfig()
		api, err := storageosapi.NewVersionedClient(apiCfg.apiAddr, defaultAPIVersion)
		if err != nil {
			return nil
		}
		api.SetAuth(apiCfg.apiUser, apiCfg.apiPass)
		u.api = api
	}
	return u.api
}

// Creates a new StorageOS volume and makes it available as a device within
// /var/lib/storageos/volumes.
func (u *storageosUtil) CreateVolume(p *storageosProvisioner) (*storageosVolume, error) {
	if p.labels == nil {
		p.labels = make(map[string]string)
	}
	opts := storageostypes.VolumeCreateOptions{
		Name:        p.volName,
		Size:        p.size,
		Description: p.description,
		Pool:        p.pool,
		FSType:      p.fsType,
		Namespace:   p.namespace,
		Labels:      p.labels,
	}

	apiCfg := &namespacedSecret{
		secretName: apiSecretName,
		namespace:  p.namespace,
		host:       p.plugin.host,
	}
	vol, err := u.client(apiCfg).VolumeCreate(opts)
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
		Size:        int(vol.Size),
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
	apiCfg := &namespacedSecret{
		secretName: apiSecretName,
		namespace:  b.namespace,
		host:       b.plugin.host,
	}
	vol, err := u.client(apiCfg).Volume(b.namespace, b.volName)
	if err != nil {
		glog.Warningf("volume retrieve failed for volume %q with namespace %q (%v)", b.volName, b.namespace, err)
		return "", err
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
		return attachFileDevice(srcPath)
	default:
		return "", fmt.Errorf(ErrDeviceNotSupported)
	}
}

// Detach detaches a volume from the host.
func (u *storageosUtil) DetachVolume(b *storageosUnmounter, loopDevice string) error {
	if _, err := os.Stat(loopDevice); os.IsNotExist(err) {
		return nil
	}
	if err := removeLoopDevice(loopDevice); err != nil {
		return err
	}
	opts := storageostypes.VolumeUnmountOptions{
		Name:      b.volName,
		Namespace: b.namespace,
		Client:    b.plugin.host.GetHostName(),
	}
	apiCfg := &namespacedSecret{
		secretName: apiSecretName,
		namespace:  b.namespace,
		host:       b.plugin.host,
	}
	return u.client(apiCfg).VolumeUnmount(opts)
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

	opts := storageostypes.VolumeMountOptions{
		Name:       b.volName,
		Namespace:  b.namespace,
		FsType:     b.fsType,
		Mountpoint: deviceMountPath,
		Client:     b.plugin.host.GetHostName(),
	}
	apiCfg := &namespacedSecret{
		secretName: apiSecretName,
		namespace:  b.namespace,
		host:       b.plugin.host,
	}
	return u.client(apiCfg).VolumeMount(opts)
}

// Unmount unmounts the volume on the host.
func (u *storageosUtil) UnmountVolume(b *storageosUnmounter) error {
	// Nothing to update in the API, we only need to update on detach.
	return nil
}

// Deletes a StorageOS volume.  Assumes it has already been unmounted and detached.
func (u *storageosUtil) DeleteVolume(d *storageosDeleter) error {
	opts := storageostypes.DeleteOptions{
		Name:      d.volName,
		Namespace: d.namespace,
	}
	apiCfg := &namespacedSecret{
		secretName: apiSecretName,
		namespace:  d.namespace,
		host:       d.plugin.host,
	}
	return u.client(apiCfg).VolumeDelete(opts)
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
func attachFileDevice(path string) (string, error) {
	blockDevicePath, err := getLoopDevice(path)
	if err != nil && err.Error() != ErrDeviceNotFound {
		return "", err
	}

	// If no existing loop device for the path, create one
	if blockDevicePath == "" {
		glog.V(4).Infof("Creating device for path: %s", path)
		blockDevicePath, err = makeLoopDevice(path)
		if err != nil {
			return "", err
		}
	}
	return blockDevicePath, nil
}

// Returns the full path to the loop device associated with the given path.
func getLoopDevice(path string) (string, error) {
	_, err := os.Stat(path)
	if os.IsNotExist(err) {
		return "", errors.New(ErrNotAvailable)
	}
	if err != nil {
		return "", fmt.Errorf("not attachable: %v", err)
	}

	exec := exec.New()
	args := []string{"-j", path}
	out, err := exec.Command(losetupPath, args...).CombinedOutput()
	if err != nil {
		glog.V(2).Infof("Failed device discover command for path %s: %v", path, err)
		return "", err
	}
	return parseLosetupOutputForDevice(out)
}

func makeLoopDevice(path string) (string, error) {
	exec := exec.New()
	args := []string{"-f", "--show", path}
	out, err := exec.Command(losetupPath, args...).CombinedOutput()
	if err != nil {
		glog.V(2).Infof("Failed device create command for path %s: %v", path, err)
		return "", err
	}
	return parseLosetupOutputForDevice(out)
}

func removeLoopDevice(device string) error {
	exec := exec.New()
	args := []string{"-d", device}
	out, err := exec.Command(losetupPath, args...).CombinedOutput()
	if err != nil {
		if !strings.Contains(string(out), "No such device or address") {
			return err
		}
	}
	return nil
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
