/*
Copyright 2016 The Kubernetes Authors.

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

// RBD Attach and Detach implementation
// Attach: lock RBD image
// Detach: release RBD image

package rbd

import (
	"fmt"
	"os"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

type rbdAttacher struct {
	plugin  *rbdPlugin
	locker  Locker
	manager diskManager
}

var _ volume.Attacher = &rbdAttacher{}

var _ volume.AttachableVolumePlugin = &rbdPlugin{}

func (plugin *rbdPlugin) NewAttacher() (volume.Attacher, error) {
	return &rbdAttacher{
		plugin:  plugin,
		locker:  &RBDLocker{},
		manager: &RBDUtil{},
	}, nil
}

func (plugin *rbdPlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	mounter := plugin.host.GetMounter()
	return mount.GetMountRefs(mounter, deviceMountPath)
}

func (attacher *rbdAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	mounter, err := volumeSpecToMounter(spec, attacher.plugin)
	if err != nil {
		glog.Warningf("failed to get rbd mounter: %v", err)
		return "", err
	}
	id := "/dev/" + mounter.Pool + "/" + mounter.Image
	return id, attacher.locker.Fencing(*mounter, string(nodeName))
}

func (attacher *rbdAttacher) WaitForAttach(spec *volume.Spec, _ string, timeout time.Duration) (string, error) {
	mounter, err := volumeSpecToMounter(spec, attacher.plugin)
	if err != nil {
		glog.Warningf("failed to get rbd mounter: %v", err)
		return "", err
	}
	return attacher.manager.AttachDisk(*mounter)
}

func (attacher *rbdAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	volumesAttachedCheck := make(map[*volume.Spec]bool)
	for _, spec := range specs {
		mounter, err := volumeSpecToMounter(spec, attacher.plugin)
		if err != nil {
			glog.Warningf("failed to get rbd mounter: %v", err)
			continue
		}
		volumesAttachedCheck[spec] = true
		attached, _ := attacher.locker.IsLocked(*mounter, string(nodeName))
		volumesAttachedCheck[spec] = attached
	}
	return volumesAttachedCheck, nil
}

func (attacher *rbdAttacher) GetDeviceMountPath(
	spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		glog.Warningf("failed to get rbd mounter: %v", err)
		return "", err
	}

	return makePDNameInternal(attacher.plugin.host, volumeSource.RBDPool, volumeSource.RBDImage), nil
}

// FIXME: this method can be further pruned.
func (attacher *rbdAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	glog.V(4).Infof("mounting volume device %s to %s", devicePath, deviceMountPath)
	mounter, err := volumeSpecToMounter(spec, attacher.plugin)
	if err != nil {
		glog.Warningf("failed to get rbd mounter for volume: %v", err)
		return err
	}
	notMnt, err := mounter.mounter.IsLikelyNotMountPoint(deviceMountPath)
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
	options := []string{}
	if mounter.rbd.ReadOnly {
		options = append(options, "ro")
	}
	if notMnt {
		if err = mounter.mounter.FormatAndMount(devicePath, deviceMountPath, mounter.fsType, options); err != nil {
			err = fmt.Errorf("rbd: failed to mount rbd device %s [%s] to %s, error %v", devicePath, mounter.fsType, deviceMountPath, err)
		}
	}
	return err
}

type rbdDetacher struct {
	mounter mount.Interface
	plugin  *rbdPlugin
	locker  Locker
	manager diskManager
}

var _ volume.Detacher = &rbdDetacher{}

func (plugin *rbdPlugin) NewDetacher() (volume.Detacher, error) {
	return &rbdDetacher{
		plugin:  plugin,
		mounter: plugin.host.GetMounter(),
		manager: &RBDUtil{},
		locker:  &RBDLocker{},
	}, nil
}

func (detacher *rbdDetacher) Detach(spec *volume.Spec, deviceMountPath string, nodeName types.NodeName) error {
	glog.V(4).Infof("detaching %v from %s", deviceMountPath, nodeName)
	mounter, err := volumeSpecToMounter(spec, detacher.plugin)
	if err != nil {
		glog.Warningf("failed to get rbd mounter: %v", err)
		return err
	}
	return detacher.locker.Defencing(*mounter, string(nodeName))
}

//FIXME: let WaitForDetach DetachDisk once volumemgr uses it
func (detacher *rbdDetacher) WaitForDetach(devicePath string, timeout time.Duration) error {
	return nil
}

func (detacher *rbdDetacher) UnmountDevice(deviceMountPath string) error {
	glog.V(4).Infof("unmount %v", deviceMountPath)
	devicePath, _, _ := mount.GetDeviceNameFromMount(detacher.mounter, deviceMountPath)
	// unmount device
	err := volumeutil.UnmountPath(deviceMountPath, detacher.mounter)
	if err != nil {
		return err
	}
	// unmap (i.e. detach) device
	if len(devicePath) > 0 {
		glog.V(4).Infof("unmap %v", devicePath)
		err = detacher.manager.DetachDisk(detacher.plugin, devicePath)
	}
	return err
}

func volumeSpecToMounter(spec *volume.Spec, plugin *rbdPlugin) (*rbdMounter, error) {
	var secret string
	var err error
	source, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	if source.SecretRef != nil {
		if secret, err = parsePVSecret(spec.NameSpace, source.SecretRef.Name, plugin.host.GetKubeClient()); err != nil {
			glog.Errorf("Couldn't get secret from %v/%v", spec.NameSpace, source.SecretRef)
			return nil, err
		}
	}

	pool := source.RBDPool
	id := source.RadosUser
	keyring := source.Keyring

	return &rbdMounter{
		rbd: &rbd{
			Image:    source.RBDImage,
			Pool:     pool,
			ReadOnly: readOnly,
			mounter:  &mount.SafeFormatAndMount{Interface: plugin.host.GetMounter(), Runner: exec.New()},
			plugin:   plugin,
		},
		Mon:     source.CephMonitors,
		Id:      id,
		Keyring: keyring,
		Secret:  secret,
		fsType:  source.FSType,
	}, nil
}
