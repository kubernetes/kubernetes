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

package iscsi

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	ioutil "k8s.io/kubernetes/pkg/volume/util"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&iscsiPlugin{nil, exec.New()}}
}

type iscsiPlugin struct {
	host volume.VolumeHost
	exe  exec.Interface
}

var _ volume.VolumePlugin = &iscsiPlugin{}
var _ volume.PersistentVolumePlugin = &iscsiPlugin{}

const (
	iscsiPluginName = "kubernetes.io/iscsi"
)

func (plugin *iscsiPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *iscsiPlugin) GetPluginName() string {
	return iscsiPluginName
}

func (plugin *iscsiPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf(
		"%v:%v:%v",
		volumeSource.TargetPortal,
		volumeSource.IQN,
		volumeSource.Lun), nil
}

func (plugin *iscsiPlugin) CanSupport(spec *volume.Spec) bool {
	if (spec.Volume != nil && spec.Volume.ISCSI == nil) || (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.ISCSI == nil) {
		return false
	}

	return true
}

func (plugin *iscsiPlugin) RequiresRemount() bool {
	return false
}

func (plugin *iscsiPlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
		v1.ReadOnlyMany,
	}
}

func (plugin *iscsiPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newMounterInternal(spec, pod.UID, &ISCSIUtil{}, plugin.host.GetMounter())
}

func (plugin *iscsiPlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, manager diskManager, mounter mount.Interface) (volume.Mounter, error) {
	// iscsi volumes used directly in a pod have a ReadOnly flag set by the pod author.
	// iscsi volumes used as a PersistentVolume gets the ReadOnly flag indirectly through the persistent-claim volume used to mount the PV
	iscsi, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	lun := strconv.Itoa(int(iscsi.Lun))
	portal := portalMounter(iscsi.TargetPortal)

	iface := iscsi.ISCSIInterface

	return &iscsiDiskMounter{
		iscsiDisk: &iscsiDisk{
			podUID:  podUID,
			volName: spec.Name(),
			portal:  portal,
			iqn:     iscsi.IQN,
			lun:     lun,
			iface:   iface,
			manager: manager,
			plugin:  plugin},
		fsType:     iscsi.FSType,
		readOnly:   readOnly,
		mounter:    &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()},
		deviceUtil: ioutil.NewDeviceHandler(ioutil.NewIOHandler()),
	}, nil
}

func (plugin *iscsiPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newUnmounterInternal(volName, podUID, &ISCSIUtil{}, plugin.host.GetMounter())
}

func (plugin *iscsiPlugin) newUnmounterInternal(volName string, podUID types.UID, manager diskManager, mounter mount.Interface) (volume.Unmounter, error) {
	return &iscsiDiskUnmounter{
		iscsiDisk: &iscsiDisk{
			podUID:  podUID,
			volName: volName,
			manager: manager,
			plugin:  plugin,
		},
		mounter: mounter,
	}, nil
}

func (plugin *iscsiPlugin) execCommand(command string, args []string) ([]byte, error) {
	cmd := plugin.exe.Command(command, args...)
	return cmd.CombinedOutput()
}

func (plugin *iscsiPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	iscsiVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			ISCSI: &v1.ISCSIVolumeSource{
				TargetPortal: volumeName,
				IQN:          volumeName,
			},
		},
	}
	return volume.NewSpecFromVolume(iscsiVolume), nil
}

type iscsiDisk struct {
	volName string
	podUID  types.UID
	portal  string
	iqn     string
	lun     string
	iface   string
	plugin  *iscsiPlugin
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager diskManager
	volume.MetricsNil
}

func (iscsi *iscsiDisk) GetPath() string {
	name := iscsiPluginName
	// safe to use PodVolumeDir now: volume teardown occurs before pod is cleaned up
	return iscsi.plugin.host.GetPodVolumeDir(iscsi.podUID, utilstrings.EscapeQualifiedNameForDisk(name), iscsi.volName)
}

type iscsiDiskMounter struct {
	*iscsiDisk
	readOnly   bool
	fsType     string
	mounter    *mount.SafeFormatAndMount
	deviceUtil ioutil.DeviceUtil
}

var _ volume.Mounter = &iscsiDiskMounter{}

func (b *iscsiDiskMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         !b.readOnly,
		SupportsSELinux: true,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *iscsiDiskMounter) CanMount() error {
	return nil
}

func (b *iscsiDiskMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

func (b *iscsiDiskMounter) SetUpAt(dir string, fsGroup *int64) error {
	// diskSetUp checks mountpoints and prevent repeated calls
	err := diskSetUp(b.manager, *b, dir, b.mounter, fsGroup)
	if err != nil {
		glog.Errorf("iscsi: failed to setup")
	}
	return err
}

type iscsiDiskUnmounter struct {
	*iscsiDisk
	mounter mount.Interface
}

var _ volume.Unmounter = &iscsiDiskUnmounter{}

// Unmounts the bind mount, and detaches the disk only if the disk
// resource was the last reference to that disk on the kubelet.
func (c *iscsiDiskUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *iscsiDiskUnmounter) TearDownAt(dir string) error {
	return diskTearDown(c.manager, *c, dir, c.mounter)
}

func portalMounter(portal string) string {
	if !strings.Contains(portal, ":") {
		portal = portal + ":3260"
	}
	return portal
}

func getVolumeSource(spec *volume.Spec) (*v1.ISCSIVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.ISCSI != nil {
		return spec.Volume.ISCSI, spec.Volume.ISCSI.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.ISCSI != nil {
		return spec.PersistentVolume.Spec.ISCSI, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference an ISCSI volume type")
}
