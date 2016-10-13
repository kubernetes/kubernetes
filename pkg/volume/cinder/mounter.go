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

package cinder

import (
	"fmt"
	"os"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

type cinderVolumeMounter struct {
	*cinderVolume
	fsType             string
	readOnly           bool
	blockDeviceMounter *mount.SafeFormatAndMount
}

func newMounter(spec *volume.Spec, podUID types.UID, plugin *cinderPlugin, mounter mount.Interface) (volume.Mounter, error) {
	cinder, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	return &cinderVolumeMounter{
		cinderVolume: &cinderVolume{
			podUID:    podUID,
			volName:   spec.Name(),
			pdName:    cinder.VolumeID,
			secretRef: cinder.SecretRef,
			mounter:   mounter,
			plugin:    plugin,
		},
		fsType:             cinder.FSType,
		readOnly:           readOnly,
		blockDeviceMounter: &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()}}, nil
}

func (b *cinderVolumeMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         !b.readOnly,
		SupportsSELinux: true,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *cinderVolumeMounter) CanMount() error {
	return nil
}

func (b *cinderVolumeMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

// SetUp bind mounts to the volume path.
func (b *cinderVolumeMounter) SetUpAt(dir string, fsGroup *int64) error {
	glog.V(5).Infof("Cinder SetUp %s to %s", b.pdName, dir)

	b.plugin.volumeLocks.LockKey(b.pdName)
	defer b.plugin.volumeLocks.UnlockKey(b.pdName)

	// TODO: handle failed mounts here.
	notmnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("Cannot validate mount point: %s %v", dir, err)
		return err
	}
	if !notmnt {
		glog.V(4).Infof("Something is already mounted to target %s", dir)
		return nil
	}
	globalPDPath := makeGlobalPDName(b.plugin.host, b.pdName)

	options := []string{"bind"}
	if b.readOnly {
		options = append(options, "ro")
	}

	if err = os.MkdirAll(dir, 0750); err != nil {
		// TODO: we should really eject the attach/detach out into its own control loop.
		glog.V(4).Infof("Could not create directory %s: %v", dir, err)
		return err
	}

	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	glog.V(4).Infof("Attempting to mount Cinder volume %s to %s with options %v", b.pdName, dir, options)
	err = b.mounter.Mount(globalPDPath, dir, "", options)
	if err != nil {
		return fmt.Errorf("Failed to mount Cinder volume %s to %s: %v", b.pdName, dir, err)
	}

	if !b.readOnly {
		volume.SetVolumeOwnership(b, fsGroup)
	}
	glog.V(3).Infof("Cinder volume %s mounted to %s", b.pdName, dir)

	return nil
}
