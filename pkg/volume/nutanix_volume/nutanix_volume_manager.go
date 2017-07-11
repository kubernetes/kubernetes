/*
Copyright 2015 The Kubernetes Authors.

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

package nutanix_volume

import (
	"fmt"
	"os"
	dstrings "strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

// Abstract interface to disk operations.
type diskManager interface {
	MakeGlobalPDName(volume nutanixVolume, target string) string
	// Attaches the disk to the kubelet's host machine.
	AttachDisk(b nutanixVolumeMounter) error
	// Detaches the disk from the kubelet's host machine.
	DetachDisk(disk nutanixVolumeUnmounter, mntPath string) error
}

// Utility to mount a disk based filesystem.
func diskSetUp(manager diskManager, b nutanixVolumeMounter, volPath string, mounter mount.Interface, fsGroup *int64) error {
	var iqn string

	// We want to keep attach/detach operation on the same volume and host atomic.
	b.plugin.volMutex.LockKey(b.volName)
	defer b.plugin.volMutex.UnlockKey(b.volName)

	// Populate host's iqn in nutanix volume.
	cli, err := NewNutanixClient(b.prismEndPoint, b.secretValue)
	if err != nil {
		return fmt.Errorf("nutanix_volume: failed to create nutanix_volume REST client")
	}
	iqn, err = findHostIscsiInitiator(b.plugin)
	if err != nil {
		return fmt.Errorf("nutanix_volume: error in finding host iqn, %v", err)
	}

	iqnAttached, err := cli.IsIqnAttached(iqn, b.volumeUUID)
	if err != nil {
		return err
	}

	if iqnAttached {
		// Either this is a duplicate call or volume is shared.
		notMnt, err := mounter.IsLikelyNotMountPoint(volPath)
		if err != nil && !os.IsNotExist(err) {
			glog.Errorf("nutanix_volume: cannot validate mountpoint: %s, error: %v", volPath, err)
			return err
		}
		if !notMnt {
			// This is a duplicate call.
			glog.V(4).Infof("nutanix_volume: duplicate disk setup call for %s", volPath)
			return nil
		}

		// Either volume is shared on multiple pods as readonly Or this is a race condition where
		// current pod is killed and a new pod is getting setup for this volume. In either case, we
		// should already have an iscsi session with target.
		glog.V(4).Infof("nutanix_volume: shared disk setup request for volume %s, readOnly: %t",
			b.volName, b.readOnly)
	} else {
		iqnCfg := IscsiClientDTO{ClientAddress: iqn}
		data := &VGAttachDetachDTO{
			IscsiClient: iqnCfg,
		}
		err = cli.AttachVG(b.volumeUUID, data)
		if err != nil {
			glog.Errorf("nutanix_volume: %v", err)
			return err
		}
	}

	target, err := getMatchingTargets(b)
	if err != nil {
		return err
	}
	globalPDPath := manager.MakeGlobalPDName(*b.nutanixVolume, target)

	if err := manager.AttachDisk(b); err != nil {
		glog.Errorf("nutanix_volume: failed to attach disk, error: %v", err)
		return err
	}

	if err := os.MkdirAll(volPath, 0750); err != nil {
		glog.Errorf("nutanix_volume: failed to mkdir: %s, error: %v", volPath, err)
		return err
	}
	// Perform a bind mount to the full path to allow duplicate mounts of the same disk.
	options := []string{"bind"}
	if b.readOnly {
		options = append(options, "ro")
	}
	mountOptions := volume.JoinMountOptions(b.mountOptions, options)
	err = mounter.Mount(globalPDPath, volPath, "", mountOptions)
	if err != nil {
		glog.Errorf("nutanix_volume: failed to bind mount: %s, error: %v", globalPDPath, err)
		return err
	}

	if !b.readOnly {
		volume.SetVolumeOwnership(&b, fsGroup)
	}

	return nil
}

// Utility to tear down a disk based filesystem.
func diskTearDown(manager diskManager, c nutanixVolumeUnmounter, volPath string, mounter mount.Interface) error {
	// We want to keep attach/detach operation on the same volume and host atomic.
	c.plugin.volMutex.LockKey(c.volName)
	defer c.plugin.volMutex.UnlockKey(c.volName)

	// This handles the case where we could not remove volPath in an earlier attempt.
	notMnt, err := mounter.IsLikelyNotMountPoint(volPath)
	if err != nil {
		glog.Errorf("nutanix_volume: cannot validate mountpoint %s, error: %v", volPath, err)
		return err
	}
	if notMnt {
		return os.Remove(volPath)
	}

	refs, err := mount.GetMountRefs(mounter, volPath)
	if err != nil {
		glog.Errorf("nutanix_volume: failed to get reference count %s, error: %v", volPath, err)
		return err
	}
	if len(refs) == 1 {
		// Remove iqn before removing the last mount.
		err = removeIqn(c, volPath)
		if err != nil {
			return err
		}
	}
	if err := mounter.Unmount(volPath); err != nil {
		glog.Errorf("nuatnix_volume: failed to unmount %s, error: %v", volPath, err)
		return err
	}
	for _, mntPath := range refs {
		if dstrings.Contains(mntPath, string(c.podUID)) == false {
			continue
		}
		if err := manager.DetachDisk(c, mntPath); err != nil {
			glog.Errorf("nutanix_volume: failed to detach disk from %s, error: %v", mntPath, err)
			return err
		}
		break
	}

	return os.Remove(volPath)
}

// Find host iscsi initiator name.
func findHostIscsiInitiator(plugin *nutanixVolumePlugin) (string, error) {
	out, err := plugin.execCommand("cat", []string{"/etc/iscsi/initiatorname.iscsi"})
	if err != nil {
		return "", err
	}
	outstr := dstrings.TrimSuffix(string(out), "\n")
	items := dstrings.Split(outstr, "=")
	if len(items) != 2 {
		return "", fmt.Errorf("nutanix_volume: invalid iqn format: %s", outstr)
	}
	iqn := items[1]
	glog.Infof("nutanix_volume: Iqn for the host is %s", iqn)
	return iqn, nil
}

// Remove hosts iqn from volume.
func removeIqn(c nutanixVolumeUnmounter, volPath string) error {
	var vgInfo *VGInfoDTO
	var iqn string

	// Remove host's iqn from nutanix volume.
	cli, err := NewNutanixClient(c.plugin.prismEndPoint, c.plugin.secretValue)
	if err != nil {
		return fmt.Errorf("failed to create nutanix_volume REST client. error: %v", err)
	}
	vgInfo, err = cli.getVolumeFromName(c.volName)
	if err == nil && vgInfo != nil {
		iqn, err = findHostIscsiInitiator(c.plugin)
		if err != nil {
			glog.Errorf("nutanix_volume: %v", err)
			return err
		}
		iqnCfg := IscsiClientDTO{ClientAddress: iqn}
		data := &VGAttachDetachDTO{
			IscsiClient: iqnCfg,
		}
		err = cli.DetachVG(vgInfo.UUID, data)
		if err != nil {
			return err
		}
	}

	return nil
}
