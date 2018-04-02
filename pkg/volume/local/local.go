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

package local

import (
	"fmt"
	"os"
	"path"
	"runtime"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/util/keymutex"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/validation"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&localVolumePlugin{}}
}

type localVolumePlugin struct {
	host        volume.VolumeHost
	volumeLocks keymutex.KeyMutex
	recorder    record.EventRecorder
}

var _ volume.VolumePlugin = &localVolumePlugin{}
var _ volume.PersistentVolumePlugin = &localVolumePlugin{}
var _ volume.BlockVolumePlugin = &localVolumePlugin{}

const (
	localVolumePluginName = "kubernetes.io/local-volume"
)

func (plugin *localVolumePlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	plugin.volumeLocks = keymutex.NewKeyMutex()
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "localvolume"})
	plugin.recorder = recorder
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
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Local != nil)
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
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Local != nil {
		return spec.PersistentVolume.Spec.Local, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a Local volume type")
}

func (plugin *localVolumePlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	volumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	return &localVolumeMounter{
		localVolume: &localVolume{
			pod:             pod,
			podUID:          pod.UID,
			volName:         spec.Name(),
			mounter:         plugin.host.GetMounter(plugin.GetPluginName()),
			plugin:          plugin,
			globalPath:      volumeSource.Path,
			MetricsProvider: volume.NewMetricsStatFS(volumeSource.Path),
		},
		readOnly: readOnly,
	}, nil

}

func (plugin *localVolumePlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return &localVolumeUnmounter{
		localVolume: &localVolume{
			podUID:  podUID,
			volName: volName,
			mounter: plugin.host.GetMounter(plugin.GetPluginName()),
			plugin:  plugin,
		},
	}, nil
}

func (plugin *localVolumePlugin) NewBlockVolumeMapper(spec *volume.Spec, pod *v1.Pod,
	_ volume.VolumeOptions) (volume.BlockVolumeMapper, error) {
	volumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	return &localVolumeMapper{
		localVolume: &localVolume{
			podUID:     pod.UID,
			volName:    spec.Name(),
			globalPath: volumeSource.Path,
			plugin:     plugin,
		},
		readOnly: readOnly,
	}, nil

}

func (plugin *localVolumePlugin) NewBlockVolumeUnmapper(volName string,
	podUID types.UID) (volume.BlockVolumeUnmapper, error) {
	return &localVolumeUnmapper{
		localVolume: &localVolume{
			podUID:  podUID,
			volName: volName,
			plugin:  plugin,
		},
	}, nil
}

// TODO: check if no path and no topology constraints are ok
func (plugin *localVolumePlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	localVolume := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: volumeName,
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				Local: &v1.LocalVolumeSource{
					Path: "",
				},
			},
		},
	}
	return volume.NewSpecFromPersistentVolume(localVolume, false), nil
}

func (plugin *localVolumePlugin) ConstructBlockVolumeSpec(podUID types.UID, volumeName,
	mapPath string) (*volume.Spec, error) {
	block := v1.PersistentVolumeBlock

	localVolume := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: volumeName,
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				Local: &v1.LocalVolumeSource{
					Path: "",
				},
			},
			VolumeMode: &block,
		},
	}

	return volume.NewSpecFromPersistentVolume(localVolume, false), nil
}

// Local volumes represent a local directory on a node.
// The directory at the globalPath will be bind-mounted to the pod's directory
type localVolume struct {
	volName string
	pod     *v1.Pod
	podUID  types.UID
	// Global path to the volume
	globalPath string
	// Mounter interface that provides system calls to mount the global path to the pod local path.
	mounter mount.Interface
	plugin  *localVolumePlugin
	volume.MetricsProvider
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
		Managed:         !m.readOnly,
		SupportsSELinux: true,
	}
}

// CanMount checks prior to mount operations to verify that the required components (binaries, etc.)
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
	m.plugin.volumeLocks.LockKey(m.globalPath)
	defer m.plugin.volumeLocks.UnlockKey(m.globalPath)

	if m.globalPath == "" {
		return fmt.Errorf("LocalVolume volume %q path is empty", m.volName)
	}

	err := validation.ValidatePathNoBacksteps(m.globalPath)
	if err != nil {
		return fmt.Errorf("invalid path: %s %v", m.globalPath, err)
	}

	notMnt, err := m.mounter.IsNotMountPoint(dir)
	glog.V(4).Infof("LocalVolume mount setup: PodDir(%s) VolDir(%s) Mounted(%t) Error(%v), ReadOnly(%t)", dir, m.globalPath, !notMnt, err, m.readOnly)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("cannot validate mount point: %s %v", dir, err)
		return err
	}

	if !notMnt {
		return nil
	}
	refs, err := mount.GetMountRefsByDev(m.mounter, m.globalPath)
	if fsGroup != nil {
		if err != nil {
			glog.Errorf("cannot collect mounting information: %s %v", m.globalPath, err)
			return err
		}

		if len(refs) > 0 {
			fsGroupNew := int64(*fsGroup)
			fsGroupSame, fsGroupOld, err := volume.IsSameFSGroup(m.globalPath, fsGroupNew)
			if err != nil {
				return fmt.Errorf("failed to check fsGroup for %s (%v)", m.globalPath, err)
			}
			if !fsGroupSame {
				m.plugin.recorder.Eventf(m.pod, v1.EventTypeWarning, events.WarnAlreadyMountedVolume, "The requested fsGroup is %d, but the volume %s has GID %d. The volume may not be shareable.", fsGroupNew, m.volName, fsGroupOld)
			}
		}

	}

	if runtime.GOOS != "windows" {
		// skip below MkdirAll for windows since the "bind mount" logic is implemented differently in mount_wiondows.go
		if err := os.MkdirAll(dir, 0750); err != nil {
			glog.Errorf("mkdir failed on disk %s (%v)", dir, err)
			return err
		}
	}
	// Perform a bind mount to the full path to allow duplicate mounts of the same volume.
	options := []string{"bind"}
	if m.readOnly {
		options = append(options, "ro")
	}

	glog.V(4).Infof("attempting to mount %s", dir)
	err = m.mounter.Mount(m.globalPath, dir, "", options)
	if err != nil {
		glog.Errorf("Mount of volume %s failed: %v", dir, err)
		notMnt, mntErr := m.mounter.IsNotMountPoint(dir)
		if mntErr != nil {
			glog.Errorf("IsNotMountPoint check failed: %v", mntErr)
			return err
		}
		if !notMnt {
			if mntErr = m.mounter.Unmount(dir); mntErr != nil {
				glog.Errorf("Failed to unmount: %v", mntErr)
				return err
			}
			notMnt, mntErr = m.mounter.IsNotMountPoint(dir)
			if mntErr != nil {
				glog.Errorf("IsNotMountPoint check failed: %v", mntErr)
				return err
			}
			if !notMnt {
				// This is very odd, we don't expect it.  We'll try again next sync loop.
				glog.Errorf("%s is still mounted, despite call to unmount().  Will try again next sync loop.", dir)
				return err
			}
		}
		os.Remove(dir)
		return err
	}
	if !m.readOnly {
		// Volume owner will be written only once on the first volume mount
		if len(refs) == 0 {
			return volume.SetVolumeOwnership(m, fsGroup)
		}
	}
	return nil
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
	return util.UnmountMountPoint(dir, u.mounter, true) /* extensiveMountPointCheck = true */
}

// localVolumeMapper implements the BlockVolumeMapper interface for local volumes.
type localVolumeMapper struct {
	*localVolume
	readOnly bool
}

var _ volume.BlockVolumeMapper = &localVolumeMapper{}

// SetUpDevice provides physical device path for the local PV.
func (m *localVolumeMapper) SetUpDevice() (string, error) {
	glog.V(4).Infof("SetupDevice returning path %s", m.globalPath)
	return m.globalPath, nil
}

// localVolumeUnmapper implements the BlockVolumeUnmapper interface for local volumes.
type localVolumeUnmapper struct {
	*localVolume
}

var _ volume.BlockVolumeUnmapper = &localVolumeUnmapper{}

// TearDownDevice will undo SetUpDevice procedure. In local PV, all of this already handled by operation_generator.
func (u *localVolumeUnmapper) TearDownDevice(mapPath, devicePath string) error {
	glog.V(4).Infof("local: TearDownDevice completed for: %s", mapPath)
	return nil
}

// GetGlobalMapPath returns global map path and error.
// path: plugins/kubernetes.io/kubernetes.io/local-volume/volumeDevices/{volumeName}
func (lv *localVolume) GetGlobalMapPath(spec *volume.Spec) (string, error) {
	return path.Join(lv.plugin.host.GetVolumeDevicePluginDir(strings.EscapeQualifiedNameForDisk(localVolumePluginName)),
		lv.volName), nil
}

// GetPodDeviceMapPath returns pod device map path and volume name.
// path: pods/{podUid}/volumeDevices/kubernetes.io~local-volume
// volName: local-pv-ff0d6d4
func (lv *localVolume) GetPodDeviceMapPath() (string, string) {
	return lv.plugin.host.GetPodVolumeDeviceDir(lv.podUID,
		strings.EscapeQualifiedNameForDisk(localVolumePluginName)), lv.volName
}
