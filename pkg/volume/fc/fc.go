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

package fc

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/utils/exec"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&fcPlugin{nil, exec.New()}}
}

type fcPlugin struct {
	host volume.VolumeHost
	exe  exec.Interface
}

var _ volume.VolumePlugin = &fcPlugin{}
var _ volume.PersistentVolumePlugin = &fcPlugin{}

const (
	fcPluginName = "kubernetes.io/fc"
)

func (plugin *fcPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *fcPlugin) GetPluginName() string {
	return fcPluginName
}

func (plugin *fcPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	if len(volumeSource.TargetWWNs) != 0 {
		// TargetWWNs are the FibreChannel target worldwide names
		return fmt.Sprintf("%v", volumeSource.TargetWWNs), nil
	} else if len(volumeSource.WWIDs) != 0 {
		// WWIDs are the FibreChannel World Wide Identifiers
		return fmt.Sprintf("%v", volumeSource.WWIDs), nil
	}

	return "", err
}

func (plugin *fcPlugin) CanSupport(spec *volume.Spec) bool {
	if (spec.Volume != nil && spec.Volume.FC == nil) || (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.FC == nil) {
		return false
	}

	return true
}

func (plugin *fcPlugin) RequiresRemount() bool {
	return false
}

func (plugin *fcPlugin) SupportsMountOption() bool {
	return false
}

func (plugin *fcPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *fcPlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
		v1.ReadOnlyMany,
	}
}

func (plugin *fcPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newMounterInternal(spec, pod.UID, &FCUtil{}, plugin.host.GetMounter())
}

func (plugin *fcPlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, manager diskManager, mounter mount.Interface) (volume.Mounter, error) {
	// fc volumes used directly in a pod have a ReadOnly flag set by the pod author.
	// fc volumes used as a PersistentVolume gets the ReadOnly flag indirectly through the persistent-claim volume used to mount the PV
	fc, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	var lun string
	var wwids []string
	if fc.Lun != nil && len(fc.TargetWWNs) != 0 {
		lun = strconv.Itoa(int(*fc.Lun))
	} else if len(fc.WWIDs) != 0 {
		for _, wwid := range fc.WWIDs {
			wwids = append(wwids, strings.Replace(wwid, " ", "_", -1))
		}
	} else {
		return nil, fmt.Errorf("fc: no fc disk information found. failed to make a new mounter")
	}

	return &fcDiskMounter{
		fcDisk: &fcDisk{
			podUID:  podUID,
			volName: spec.Name(),
			wwns:    fc.TargetWWNs,
			lun:     lun,
			wwids:   wwids,
			manager: manager,
			io:      &osIOHandler{},
			plugin:  plugin},
		fsType:   fc.FSType,
		readOnly: readOnly,
		mounter:  &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()},
	}, nil
}

func (plugin *fcPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newUnmounterInternal(volName, podUID, &FCUtil{}, plugin.host.GetMounter())
}

func (plugin *fcPlugin) newUnmounterInternal(volName string, podUID types.UID, manager diskManager, mounter mount.Interface) (volume.Unmounter, error) {
	return &fcDiskUnmounter{
		fcDisk: &fcDisk{
			podUID:  podUID,
			volName: volName,
			manager: manager,
			plugin:  plugin,
			io:      &osIOHandler{},
		},
		mounter: mounter,
	}, nil
}

func (plugin *fcPlugin) execCommand(command string, args []string) ([]byte, error) {
	cmd := plugin.exe.Command(command, args...)
	return cmd.CombinedOutput()
}

func (plugin *fcPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	fcVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			FC: &v1.FCVolumeSource{},
		},
	}
	return volume.NewSpecFromVolume(fcVolume), nil
}

type fcDisk struct {
	volName string
	podUID  types.UID
	portal  string
	wwns    []string
	lun     string
	wwids   []string
	plugin  *fcPlugin
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager diskManager
	// io handler interface
	io ioHandler
	volume.MetricsNil
}

func (fc *fcDisk) GetPath() string {
	name := fcPluginName
	// safe to use PodVolumeDir now: volume teardown occurs before pod is cleaned up
	return fc.plugin.host.GetPodVolumeDir(fc.podUID, utilstrings.EscapeQualifiedNameForDisk(name), fc.volName)
}

type fcDiskMounter struct {
	*fcDisk
	readOnly bool
	fsType   string
	mounter  *mount.SafeFormatAndMount
}

var _ volume.Mounter = &fcDiskMounter{}

func (b *fcDiskMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         !b.readOnly,
		SupportsSELinux: true,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *fcDiskMounter) CanMount() error {
	return nil
}

func (b *fcDiskMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

func (b *fcDiskMounter) SetUpAt(dir string, fsGroup *int64) error {
	// diskSetUp checks mountpoints and prevent repeated calls
	err := diskSetUp(b.manager, *b, dir, b.mounter, fsGroup)
	if err != nil {
		glog.Errorf("fc: failed to setup")
	}
	return err
}

type fcDiskUnmounter struct {
	*fcDisk
	mounter mount.Interface
}

var _ volume.Unmounter = &fcDiskUnmounter{}

// Unmounts the bind mount, and detaches the disk only if the disk
// resource was the last reference to that disk on the kubelet.
func (c *fcDiskUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *fcDiskUnmounter) TearDownAt(dir string) error {
	if pathExists, pathErr := util.PathExists(dir); pathErr != nil {
		return fmt.Errorf("Error checking if path exists: %v", pathErr)
	} else if !pathExists {
		glog.Warningf("Warning: Unmount skipped because path does not exist: %v", dir)
		return nil
	}
	return diskTearDown(c.manager, *c, dir, c.mounter)
}

func getVolumeSource(spec *volume.Spec) (*v1.FCVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.FC != nil {
		return spec.Volume.FC, spec.Volume.FC.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.FC != nil {
		return spec.PersistentVolume.Spec.FC, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a FibreChannel volume type")
}
