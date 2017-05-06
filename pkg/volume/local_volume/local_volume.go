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

package local_volume

import (
	"fmt"
	"os"
	"syscall"

	"github.com/golang/glog"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&localVolumePlugin{}}
}

type localVolumePlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &localVolumePlugin{}
var _ volume.PersistentVolumePlugin = &localVolumePlugin{}

const (
	localVolumePluginName = "kubernetes.io/local-volume"
)

func (plugin *localVolumePlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *localVolumePlugin) GetPluginName() string {
	return localVolumePluginName
}

func (plugin *localVolumePlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	// This volume is only supported as a PersistentVolumeSource, so the PV name is unique
	return spec.Name(), nil
}

func (plugin *localVolumePlugin) CanSupport(spec *volume.Spec) bool {
	// This volume is only supported as a PersistentVolumeSource
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.LocalVolume != nil)
}

func (plugin *localVolumePlugin) RequiresRemount() bool {
	return false
}

func (plugin *localVolumePlugin) SupportsMountOption() bool {
	return false
}

func (plugin *localVolumePlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *localVolumePlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	// The current meaning of AccessMode is how many nodes can attach to it, not how many pods can mount it
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
	}
}

func getVolumeSource(spec *volume.Spec) (*v1.LocalVolumeSource, bool, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.LocalVolume != nil {
		return spec.PersistentVolume.Spec.LocalVolume, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference an LocalVolume volume type")
}

func (plugin *localVolumePlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	volumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	if volumeSource.Fs == nil {
		return nil, fmt.Errorf("LocalVolume.Fs is nil")
	}

	hostNode := plugin.host.GetNodeName()
	if volumeSource.NodeName != hostNode {
		return nil, fmt.Errorf("LocalVolume NodeName %q does not match this node %q", volumeSource.NodeName, hostNode)
	}

	return &localVolumeMounter{
		localVolume: &localVolume{
			podUID:     pod.UID,
			volName:    spec.Name(),
			mounter:    plugin.host.GetMounter(),
			plugin:     plugin,
			globalPath: volumeSource.Fs.Path,
		},
		readOnly: readOnly,
	}, nil

}

func (plugin *localVolumePlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return &localVolumeUnmounter{
		localVolume: &localVolume{
			podUID:  podUID,
			volName: volName,
			mounter: plugin.host.GetMounter(),
			plugin:  plugin,
		},
	}, nil
}

func (plugin *localVolumePlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	localVolume := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: volumeName,
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				LocalVolume: &v1.LocalVolumeSource{
					NodeName: plugin.host.GetNodeName(),
					Fs:       &v1.LocalFsVolume{Path: ""},
				},
			},
		},
	}
	return volume.NewSpecFromPersistentVolume(localVolume, false), nil
}

// Local volumes represent a local directory on a node.
// The directory at the globalPath will be bind-mounted to the pod's directory
type localVolume struct {
	volName string
	podUID  types.UID
	// Global path to the volume
	globalPath string
	// Mounter interface that provides system calls to mount the global path to the pod local path.
	mounter mount.Interface
	plugin  *localVolumePlugin
	// TODO: add metrics
	volume.MetricsNil
}

func (l *localVolume) GetPath() string {
	return l.plugin.host.GetPodVolumeDir(l.podUID, strings.EscapeQualifiedNameForDisk(localVolumePluginName), l.volName)
}

type localVolumeMounter struct {
	*localVolume
	readOnly bool
}

var _ volume.Mounter = &localVolumeMounter{}

func (m *localVolumeMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        m.readOnly,
		Managed:         false,
		SupportsSELinux: false,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (m *localVolumeMounter) CanMount() error {
	return nil
}

// SetUp bind mounts the directory to the volume path
func (m *localVolumeMounter) SetUp(fsGroup *int64) error {
	return m.SetUpAt(m.GetPath(), fsGroup)
}

// SetUpAt bind mounts the directory to the volume path and sets up volume ownership
func (m *localVolumeMounter) SetUpAt(dir string, fsGroup *int64) error {
	if m.globalPath == "" {
		err := fmt.Errorf("LocalVolume volume %q path is empty", m.volName)
		return err
	}

	// If fsGroup is not nil, we need to make sure the requested fsGroup doesn't
	// conflict with fsGroup already set for previous pod.
	if fsGroup != nil {
		// Check whether fsGroup has been set before.
		bindMounted, err := m.isBindMounted()
		if err != nil {
			return err
		}
		if bindMounted {
			info, err := os.Stat(m.globalPath)
			if err != nil {
				return err
			}
			stat, ok := info.Sys().(*syscall.Stat_t)
			if !ok {
				return fmt.Errorf("error converting Stat_t for path %v while checking volume fsGroup conflict", m.globalPath)
			}

			if stat == nil {
				return fmt.Errorf("got nil stat_t for path %v while checking volume fsGroup conflict", m.globalPath)
			}

			if int64(stat.Gid) != *fsGroup {
				return fmt.Errorf("fsGroup conflict for volume %s: setting fsGroup to %d while exist fsGroup is %d", m.volName, *fsGroup, stat.Gid)
			}
		}
	}

	notMnt, err := m.mounter.IsLikelyNotMountPoint(dir)
	glog.V(4).Infof("LocalVolume mount setup: PodDir(%s) VolDir(%s) Mounted(%t) Error(%v), ReadOnly(%t)", dir, m.globalPath, !notMnt, err, m.readOnly)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("cannot validate mount point: %s %v", dir, err)
		return err
	}
	if !notMnt {
		return nil
	}

	if err := os.MkdirAll(dir, 0750); err != nil {
		glog.Errorf("mkdir failed on disk %s (%v)", dir, err)
		return err
	}

	// Perform a bind mount to the full path to allow duplicate mounts of the same volume.
	options := []string{"bind"}
	if m.readOnly {
		options = append(options, "ro")
	}

	glog.V(4).Infof("attempting to mount %s", dir)
	err = m.mounter.Mount(m.globalPath, dir, "", options)
	if err != nil {
		notMnt, mntErr := m.mounter.IsLikelyNotMountPoint(dir)
		if mntErr != nil {
			glog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
			return err
		}
		if !notMnt {
			if mntErr = m.mounter.Unmount(dir); mntErr != nil {
				glog.Errorf("Failed to unmount: %v", mntErr)
				return err
			}
			notMnt, mntErr := m.mounter.IsLikelyNotMountPoint(dir)
			if mntErr != nil {
				glog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
				return err
			}
			if !notMnt {
				// This is very odd, we don't expect it.  We'll try again next sync loop.
				glog.Errorf("%s is still mounted, despite call to unmount().  Will try again next sync loop.", dir)
				return err
			}
		}
		os.Remove(dir)
		glog.Errorf("Mount of volume %s failed: %v", dir, err)
		return err
	}

	return volume.SetVolumeOwnership(m, fsGroup)
}

// isBindMounted checks whether the volume has beend mounted before.
func (m *localVolumeMounter) isBindMounted() (bool, error) {
	mountpoints, err := m.mounter.List()
	if err != nil {
		return false, err
	}
	for _, mountpoint := range mountpoints {
		_, pluginName, volumeName, _ := m.plugin.host.GetInfoFromPodVolumeDir(mountpoint.Path)
		if volumeName == m.volName && strings.UnescapeQualifiedNameForDisk(pluginName) == m.plugin.GetPluginName() {
			return true, nil
		}
	}
	return false, nil
}

type localVolumeUnmounter struct {
	*localVolume
}

var _ volume.Unmounter = &localVolumeUnmounter{}

// TearDown unmounts the bind mount
func (u *localVolumeUnmounter) TearDown() error {
	return u.TearDownAt(u.GetPath())
}

// TearDownAt unmounts the bind mount
func (u *localVolumeUnmounter) TearDownAt(dir string) error {
	glog.V(4).Infof("Unmounting volume %q at path %q\n", u.volName, dir)
	return util.UnmountPath(dir, u.mounter)
}
