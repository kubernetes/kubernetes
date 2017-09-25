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

package iscsi

import (
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

type iscsiAttacher struct {
	host    volume.VolumeHost
	manager diskManager
}

var _ volume.Attacher = &iscsiAttacher{}

var _ volume.AttachableVolumePlugin = &iscsiPlugin{}

func (plugin *iscsiPlugin) NewAttacher() (volume.Attacher, error) {
	return &iscsiAttacher{
		host:    plugin.host,
		manager: &ISCSIUtil{},
	}, nil
}

func (plugin *iscsiPlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	mounter := plugin.host.GetMounter(iscsiPluginName)
	return mount.GetMountRefs(mounter, deviceMountPath)
}

func (attacher *iscsiAttacher) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	return "", nil
}

func (attacher *iscsiAttacher) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	volumesAttachedCheck := make(map[*volume.Spec]bool)
	for _, spec := range specs {
		volumesAttachedCheck[spec] = true
	}

	return volumesAttachedCheck, nil
}

func (attacher *iscsiAttacher) WaitForAttach(spec *volume.Spec, devicePath string, pod *v1.Pod, timeout time.Duration) (string, error) {
	mounter, err := attacher.volumeSpecToMounter(spec, attacher.host, pod)
	if err != nil {
		glog.Warningf("failed to get iscsi mounter: %v", err)
		return "", err
	}
	return attacher.manager.AttachDisk(*mounter)
}

func (attacher *iscsiAttacher) GetDeviceMountPath(
	spec *volume.Spec) (string, error) {
	mounter, err := attacher.volumeSpecToMounter(spec, attacher.host, nil)
	if err != nil {
		glog.Warningf("failed to get iscsi mounter: %v", err)
		return "", err
	}
	if mounter.InitiatorName != "" {
		// new iface name is <target portal>:<volume name>
		mounter.Iface = mounter.Portals[0] + ":" + mounter.VolName
	}
	return attacher.manager.MakeGlobalPDName(*mounter.iscsiDisk), nil
}

func (attacher *iscsiAttacher) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	mounter := attacher.host.GetMounter(iscsiPluginName)
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
	volumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return err
	}

	options := []string{}
	if readOnly {
		options = append(options, "ro")
	}
	if notMnt {
		diskMounter := &mount.SafeFormatAndMount{Interface: mounter, Exec: attacher.host.GetExec(iscsiPluginName)}
		mountOptions := volume.MountOptionFromSpec(spec, options...)
		err = diskMounter.FormatAndMount(devicePath, deviceMountPath, volumeSource.FSType, mountOptions)
		if err != nil {
			os.Remove(deviceMountPath)
			return err
		}
	}
	return nil
}

type iscsiDetacher struct {
	host    volume.VolumeHost
	mounter mount.Interface
	manager diskManager
}

var _ volume.Detacher = &iscsiDetacher{}

func (plugin *iscsiPlugin) NewDetacher() (volume.Detacher, error) {
	return &iscsiDetacher{
		host:    plugin.host,
		mounter: plugin.host.GetMounter(iscsiPluginName),
		manager: &ISCSIUtil{},
	}, nil
}

func (detacher *iscsiDetacher) Detach(deviceMountPath string, nodeName types.NodeName) error {
	return nil
}

func (detacher *iscsiDetacher) UnmountDevice(deviceMountPath string) error {
	unMounter := detacher.volumeSpecToUnmounter(detacher.mounter)
	err := detacher.manager.DetachDisk(*unMounter, deviceMountPath)
	if err != nil {
		return fmt.Errorf("iscsi: failed to detach disk: %s\nError: %v", deviceMountPath, err)
	}
	glog.V(4).Infof("iscsi: %q is unmounted, deleting the directory", deviceMountPath)
	err = os.RemoveAll(deviceMountPath)
	if err != nil {
		return fmt.Errorf("iscsi: failed to delete the directory: %s\nError: %v", deviceMountPath, err)
	}
	glog.V(4).Infof("iscsi: successfully detached disk: %s", deviceMountPath)
	return nil
}

func (attacher *iscsiAttacher) volumeSpecToMounter(spec *volume.Spec, host volume.VolumeHost, pod *v1.Pod) (*iscsiDiskMounter, error) {
	var secret map[string]string
	var bkportal []string
	iscsi, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}
	// Obtain secret for AttachDisk
	if iscsi.SecretRef != nil && pod != nil {
		if secret, err = volumeutil.GetSecretForPod(pod, iscsi.SecretRef.Name, host.GetKubeClient()); err != nil {
			glog.Errorf("Couldn't get secret from %v/%v", pod.Namespace, iscsi.SecretRef)
			return nil, err
		}
	}
	lun := strconv.Itoa(int(iscsi.Lun))
	portal := portalMounter(iscsi.TargetPortal)
	bkportal = append(bkportal, portal)
	for _, tp := range iscsi.Portals {
		bkportal = append(bkportal, portalMounter(string(tp)))
	}
	iface := iscsi.ISCSIInterface
	exec := attacher.host.GetExec(iscsiPluginName)
	var initiatorName string
	if iscsi.InitiatorName != nil {
		initiatorName = *iscsi.InitiatorName
	}

	return &iscsiDiskMounter{
		iscsiDisk: &iscsiDisk{
			plugin: &iscsiPlugin{
				host: host,
			},
			VolName:        spec.Name(),
			Portals:        bkportal,
			Iqn:            iscsi.IQN,
			lun:            lun,
			Iface:          iface,
			chap_discovery: iscsi.DiscoveryCHAPAuth,
			chap_session:   iscsi.SessionCHAPAuth,
			secret:         secret,
			InitiatorName:  initiatorName,
			manager:        &ISCSIUtil{}},
		fsType:     iscsi.FSType,
		readOnly:   readOnly,
		mounter:    &mount.SafeFormatAndMount{Interface: host.GetMounter(iscsiPluginName), Exec: exec},
		exec:       exec,
		deviceUtil: volumeutil.NewDeviceHandler(volumeutil.NewIOHandler()),
	}, nil
}

func (detacher *iscsiDetacher) volumeSpecToUnmounter(mounter mount.Interface) *iscsiDiskUnmounter {
	exec := detacher.host.GetExec(iscsiPluginName)
	return &iscsiDiskUnmounter{
		iscsiDisk: &iscsiDisk{
			plugin: &iscsiPlugin{},
		},
		mounter: mounter,
		exec:    exec,
	}
}
